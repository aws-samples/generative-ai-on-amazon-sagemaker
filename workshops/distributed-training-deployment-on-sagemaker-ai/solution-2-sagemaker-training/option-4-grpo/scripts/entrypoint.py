"""GRPO training entry script, run by ``launcher.py`` on the Ray head node.

The training job's command is ``python launcher.py --entrypoint entrypoint.py``.
``launcher.py`` is the standard Ray-on-SageMaker launcher from
https://github.com/aws-samples/sample-ray-on-amazon-sagemaker-training-jobs, copied
here unmodified. It owns everything about the Ray cluster: it starts the head,
joins the workers, waits for the nodes to connect, keeps worker containers alive
until the head finishes, and tears the cluster down. It then executes this file as
``__main__`` on the head node only, in a process that already holds a connected
Ray driver.

That leaves this script with only the GRPO-specific work, in one fixed order:

1. Read the SageMaker contract -- ``resourceconfig.json``, ``hyperparameters.json``,
   and the ``SM_CHANNEL_*`` mounts.
2. Resolve each channel mount to the concrete data **files** inside it, and refuse
   to continue if any declared channel holds zero rows.
3. Write the resolved configuration and a run record to ``/opt/ml/output/data``.
4. Confirm the Ray cluster the launcher formed registered every GPU the job was
   given, run the GRPO trainer (``run_grpo``), then merge and validate the
   checkpoint into ``/opt/ml/model`` (``export_checkpoint``).

This script runs **only inside the GPU container**. Its only imports beyond the
standard library are the two sibling scripts in this directory, and ``ray``, which
is deferred into the one function that needs it so the pure helpers stay
importable on a workstation.

Two ordering decisions carry real weight.

**Channels are validated before anything expensive happens.** veRL's own failure
on an empty dataset surfaces deep inside a Ray actor, minutes into a run. Checking
the parquet footers first turns a confusing mid-run traceback into an immediate,
named failure.

**Channels resolve to files, never to the mount directory.** veRL's
``RLHFDataset`` dispatches on a file suffix and raises ``Unsupported file format``
for a directory -- it does not expand one. So ``SM_CHANNEL_TRAIN`` pointing at
``/opt/ml/input/data/train`` has to become that directory's ``train.parquet``.
A channel holding several files becomes a Hydra list, which
``run_grpo.render_data_files`` accepts.

The export step runs only after a successful trainer exit. A failed run raises,
which ``launcher.py`` turns into a failure reason for SageMaker; the checkpoints
veRL synced to Amazon S3 are left untouched and ``/opt/ml/model`` stays empty, so
no half-trained model is uploaded and the merge can be retried against the
retained checkpoint without repeating training.
"""

import json
import os
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import export_checkpoint
import run_grpo

CONFIG_DIR = Path("/opt/ml/input/config")
HYPERPARAMETERS_PATH = CONFIG_DIR / "hyperparameters.json"
RESOURCE_CONFIG_PATH = CONFIG_DIR / "resourceconfig.json"
OUTPUT_DATA_DIR = Path("/opt/ml/output/data")
FAILURE_REASON_PATH = Path("/opt/ml/output/failure")
"""SageMaker surfaces this file's contents as the job's ``FailureReason``.

``launcher.py`` reads it when the entry script fails and quotes it in the error it
raises. It only writes the file itself when it is absent, and then only with a
generic message, so the specific reason has to be written here first.
"""
RUN_RECORD_NAME = "run-record.json"
RESOLVED_CONFIG_NAME = "resolved-hyperparameters.json"

#: Declared channels, and the environment variable naming each mount. Both are
#: required: a GRPO run without a validation split cannot report eval metrics, and
#: the notebook always supplies both.
CHANNEL_ENV_VARS: dict[str, str] = {
    run_grpo.TRAIN_CHANNEL: "SM_CHANNEL_TRAIN",
    run_grpo.VALIDATION_CHANNEL: "SM_CHANNEL_VALIDATION",
}

#: Suffixes veRL's dataset loader accepts, in the order preferred when a channel
#: happens to hold more than one kind.
DATA_SUFFIXES: tuple[str, ...] = (".parquet", ".jsonl", ".json")

#: How long to wait for the cluster to register every expected GPU. The launcher
#: has already waited for the nodes to connect, so this is normally satisfied on
#: the first poll; the budget only bounds the failure case.
GPU_REGISTRATION_TIMEOUT_S = 120
GPU_REGISTRATION_POLL_S = 5.0


class EntrypointError(RuntimeError):
    """Base class for every failure raised by this module."""


class ChannelError(EntrypointError):
    """A declared data channel is missing, unreadable, or empty."""


class ClusterError(EntrypointError):
    """The Ray cluster is absent or smaller than the job's resources."""


@dataclass(frozen=True)
class ResourceConfig:
    """The subset of the SageMaker resource contract this script needs.

    ``current_host`` and ``hosts`` come from ``resourceconfig.json``;
    ``gpus_per_node`` comes from ``SM_NUM_GPUS``.
    """

    current_host: str
    hosts: tuple[str, ...]
    gpus_per_node: int

    @property
    def node_count(self) -> int:
        return len(self.hosts)

    @property
    def expected_gpu_count(self) -> int:
        """GPUs the Ray cluster must hold before training may start."""
        return self.node_count * self.gpus_per_node


# --------------------------------------------------------------------------- #
# Pure logic. No filesystem, no Ray, no subprocess.
# --------------------------------------------------------------------------- #


def assert_channels_present(channel_rows: Mapping[str, int]) -> None:
    """Raise unless every declared channel holds at least one row.

    Every channel is inspected before raising, and the error names **all** of the
    offenders rather than only the first. A job with two empty channels should
    take one round trip to diagnose, not two.

    Args:
        channel_rows: Channel name to row count, as counted from the resolved
            files. A channel that resolved to no file at all is expected to have
            been rejected earlier, by :func:`resolve_channel_files`.

    Raises:
        ChannelError: If ``channel_rows`` omits a declared channel, or if any
            channel maps to zero rows or a negative count.
    """
    missing = [name for name in CHANNEL_ENV_VARS if name not in channel_rows]
    if missing:
        raise ChannelError(
            f"no row count was resolved for declared channel(s) {sorted(missing)}; "
            f"resolved channels are {sorted(channel_rows)}"
        )

    empty = sorted(name for name, rows in channel_rows.items() if rows <= 0)
    if empty:
        detail = ", ".join(f"{name}={channel_rows[name]}" for name in empty)
        raise ChannelError(
            f"data channel(s) {empty} resolved to zero rows ({detail}). Training "
            f"cannot proceed, and this is checked up front so the job fails now "
            f"rather than inside a Ray actor. Re-run the data preparation notebook "
            f"and confirm the manifest row counts are non-zero."
        )


def select_data_files(channel: str, names: Sequence[str]) -> list[str]:
    """Pick the data files from one channel's directory listing.

    Only the first matching suffix group is returned. Mixing ``.parquet`` and
    ``.json`` in one channel would make veRL read the same split through two
    different loaders, so the more specific format wins rather than both being
    passed.

    Args:
        channel: Channel name, used only in the error message.
        names: Filenames in the channel directory. Order is irrelevant; the
            result is sorted for determinism.

    Returns:
        A sorted, non-empty list of filenames.

    Raises:
        ChannelError: If no filename carries a suffix veRL accepts.
    """
    for suffix in DATA_SUFFIXES:
        matched = sorted(name for name in names if name.lower().endswith(suffix))
        if matched:
            return matched
    raise ChannelError(
        f"channel {channel!r} holds no file with a suffix veRL accepts "
        f"({list(DATA_SUFFIXES)}); found {sorted(names)!r}. veRL reads the file "
        f"suffix to choose a loader and does not expand a directory."
    )


def render_channel_value(paths: Sequence[str]) -> str:
    """Render resolved paths as the value for veRL's ``data.*_files``.

    A single file is passed as a bare path; several become a Hydra list literal,
    which is what ``run_grpo.render_data_files`` validates and passes through.
    """
    if not paths:
        raise ChannelError("cannot render an empty file list")
    if len(paths) == 1:
        return paths[0]
    return "[" + ",".join(paths) + "]"


def parse_resource_config(payload: Mapping[str, object], gpus_per_node: int) -> ResourceConfig:
    """Build a :class:`ResourceConfig` from parsed ``resourceconfig.json``.

    Kept separate from the file read so the parsing rules are testable without a
    ``/opt/ml`` tree.
    """
    current_host = payload.get("current_host")
    hosts = payload.get("hosts")

    if not isinstance(current_host, str) or not current_host:
        raise EntrypointError(
            f"resourceconfig.json has no usable 'current_host'; got {current_host!r}"
        )
    if not isinstance(hosts, Sequence) or isinstance(hosts, str) or not hosts:
        raise EntrypointError(f"resourceconfig.json has no usable 'hosts' list; got {hosts!r}")
    if not all(isinstance(host, str) and host for host in hosts):
        raise EntrypointError(
            f"resourceconfig.json 'hosts' must be non-empty strings; got {list(hosts)!r}"
        )
    if current_host not in hosts:
        raise EntrypointError(
            f"current_host {current_host!r} is absent from hosts {list(hosts)!r}; the "
            f"resource config and the environment disagree"
        )
    if gpus_per_node < 1:
        raise EntrypointError(
            f"SM_NUM_GPUS must be at least 1 for a GRPO run; got {gpus_per_node}"
        )

    return ResourceConfig(
        current_host=current_host,
        hosts=tuple(hosts),
        gpus_per_node=gpus_per_node,
    )


def render_registration_state(nodes: Sequence[Mapping[str, object]]) -> str:
    """Render one line per Ray node: address, liveness, and registered GPUs.

    Takes the shape ``ray.nodes()`` returns but requires only plain mappings, so
    the failure report is verifiable without a live cluster.
    """
    if not nodes:
        return "  (no nodes registered)"

    lines = []
    for node in nodes:
        address = node.get("NodeManagerAddress") or node.get("NodeID") or "<unknown>"
        hostname = node.get("NodeName") or "<unknown>"
        alive = bool(node.get("Alive", False))
        resources = node.get("Resources") or {}
        gpus = 0.0
        if isinstance(resources, Mapping):
            gpus = float(resources.get("GPU", 0.0) or 0.0)
        lines.append(f"  - {hostname} ({address}): alive={alive} gpus={gpus:g}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# SageMaker contract reads.
# --------------------------------------------------------------------------- #


def _read_json_object(path: Path, what: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EntrypointError(
            f"{what} not found at {path}; this script runs only inside a SageMaker "
            f"training container"
        ) from exc
    except json.JSONDecodeError as exc:
        raise EntrypointError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise EntrypointError(f"{path} must contain a JSON object; got {type(payload).__name__}")
    return payload


def read_hyperparameters(path: Path = HYPERPARAMETERS_PATH) -> dict[str, str]:
    """Read ``hyperparameters.json`` as a mapping of strings.

    SageMaker writes every hyperparameter value as a JSON string, and
    ``run_grpo.build_verl_argv`` parses each one itself. Values are coerced to
    ``str`` here rather than trusted, so a hand-edited file carrying a real JSON
    number still yields the string form the override renderers expect.
    """
    payload = _read_json_object(path, "hyperparameters")
    return {str(key): str(value) for key, value in payload.items()}


def read_resource_config(
    path: Path = RESOURCE_CONFIG_PATH,
    env: Mapping[str, str] | None = None,
) -> ResourceConfig:
    """Read ``resourceconfig.json`` and ``SM_NUM_GPUS`` into a config object."""
    env = os.environ if env is None else env
    payload = _read_json_object(path, "SageMaker resource configuration")

    raw_gpus = env.get("SM_NUM_GPUS", "0")
    try:
        gpus_per_node = int(raw_gpus)
    except (TypeError, ValueError) as exc:
        raise EntrypointError(f"SM_NUM_GPUS is not an integer: {raw_gpus!r}") from exc

    return parse_resource_config(payload, gpus_per_node)


def count_rows(path: Path) -> int:
    """Count rows in one resolved data file.

    Parquet is read through its footer only -- ``pyarrow`` exposes the row count
    from metadata without materialising a column -- so this stays cheap even for a
    large split. JSON Lines is counted by scanning lines, and a JSON array by
    parsing it, because neither format records a count.

    ``pyarrow`` is imported inside the function so that every pure function above
    remains importable on a workstation that has no Arrow installed.
    """
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - present in the container
            raise EntrypointError(
                "pyarrow is not importable, so parquet row counts cannot be "
                "verified; entrypoint.py runs only inside the veRL container"
            ) from exc
        try:
            return int(pq.ParquetFile(str(path)).metadata.num_rows)
        except Exception as exc:  # noqa: BLE001 - any read failure is a bad channel
            raise ChannelError(f"could not read parquet metadata from {path}: {exc}") from exc

    try:
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip())
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ChannelError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ChannelError(f"{path} is not valid JSON: {exc}") from exc

    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, Mapping):
        return 1
    raise ChannelError(f"{path} holds neither a JSON array nor an object")


def resolve_channel_files(
    env: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, int], dict[str, list[str]]]:
    """Resolve every declared channel to concrete files and row counts.

    Returns ``(channel_values, channel_rows, channel_file_lists)`` where
    ``channel_values`` is what ``run_grpo.build_verl_argv`` consumes,
    ``channel_rows`` is what :func:`assert_channels_present` checks, and
    ``channel_file_lists`` goes into the run record so the exact inputs of a run
    are recoverable afterwards.

    Raises:
        ChannelError: If a channel's environment variable is unset, its directory
            is missing, or it holds no file veRL can read.
    """
    env = os.environ if env is None else env

    values: dict[str, str] = {}
    rows: dict[str, int] = {}
    listings: dict[str, list[str]] = {}

    for channel, var in CHANNEL_ENV_VARS.items():
        mount = env.get(var)
        if not mount:
            raise ChannelError(
                f"{var} is unset, so channel {channel!r} has no mount point. The "
                f"training job must declare an input channel named {channel!r}."
            )
        directory = Path(mount)
        if not directory.is_dir():
            raise ChannelError(
                f"{var}={mount!r} is not a directory, so channel {channel!r} was "
                f"not mounted as expected"
            )

        names = select_data_files(channel, [p.name for p in directory.iterdir() if p.is_file()])
        paths = [str(directory / name) for name in names]
        values[channel] = render_channel_value(paths)
        listings[channel] = paths
        rows[channel] = sum(count_rows(Path(p)) for p in paths)
        print(
            f"[entrypoint] channel {channel}: {len(paths)} file(s), {rows[channel]} row(s)",
            flush=True,
        )

    return values, rows, listings


# --------------------------------------------------------------------------- #
# Ray cluster check.
# --------------------------------------------------------------------------- #


def assert_cluster_gpus(
    expected_gpus: int,
    *,
    timeout_s: int = GPU_REGISTRATION_TIMEOUT_S,
    poll_seconds: float = GPU_REGISTRATION_POLL_S,
) -> int:
    """Confirm the Ray cluster holds ``expected_gpus``, and return the registered count.

    ``launcher.py`` connects a Ray driver in this process before running this
    script, and waits for every node to join -- but on a timeout it logs a warning
    and proceeds with a partial cluster. veRL would then block forever waiting
    for a resource pool it can never fill. Checking the GPU count here turns that
    hang into an immediate failure that names the missing nodes.

    If no driver is connected (this script was run without the launcher), a
    connection to a running cluster is attempted, and a clear error raised when
    there is none.
    """
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - present in the container
        raise ClusterError(
            "the 'ray' package is not importable; entrypoint.py runs only inside the "
            "veRL GPU container, which ships Ray"
        ) from exc

    if not ray.is_initialized():
        try:
            ray.init(address="auto", ignore_reinit_error=True)
        except Exception as exc:  # noqa: BLE001 - any failure here means no cluster
            raise ClusterError(
                "no Ray cluster is running. Start the job through the launcher: "
                "`python launcher.py --entrypoint entrypoint.py`, which forms the "
                f"cluster before running this script. Ray reported: {exc}"
            ) from exc

    deadline = time.monotonic() + timeout_s
    registered = 0
    while True:
        registered = int(ray.cluster_resources().get("GPU", 0))
        if registered >= expected_gpus:
            print(
                f"[entrypoint] Ray cluster has {registered} GPU(s); expected {expected_gpus}",
                flush=True,
            )
            return registered
        if time.monotonic() >= deadline:
            break
        print(
            f"[entrypoint] waiting for GPUs: {registered}/{expected_gpus} registered",
            flush=True,
        )
        time.sleep(poll_seconds)

    raise ClusterError(
        f"Ray registered {registered} of {expected_gpus} expected GPU(s) within "
        f"{timeout_s}s. Per-node registration state:\n"
        f"{render_registration_state(ray.nodes())}\n"
        f"If nodes are missing, check the launcher's log for 'Timed out waiting for "
        f"all nodes to connect' and that the job's security group permits ingress "
        f"from itself on port 6379."
    )


# --------------------------------------------------------------------------- #
# Run record.
# --------------------------------------------------------------------------- #


def write_json(directory: Path, name: str, payload: Mapping[str, object]) -> Path:
    """Write one JSON document into ``directory``, creating it if needed.

    Never raises on a write failure: the run record is diagnostic, and losing it
    must not fail a training job that otherwise succeeded. A failure is reported
    on stdout, which reaches CloudWatch.
    """
    target = directory / name
    try:
        directory.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"[entrypoint] could not write {target}: {exc}", flush=True)
    return target


def output_data_dir(env: Mapping[str, str] | None = None) -> Path:
    """Resolve ``/opt/ml/output/data``, honouring ``SM_OUTPUT_DATA_DIR``."""
    env = os.environ if env is None else env
    return Path(env.get("SM_OUTPUT_DATA_DIR") or OUTPUT_DATA_DIR)


def write_failure_reason(message: str, path: Path = FAILURE_REASON_PATH) -> None:
    """Record ``message`` as the job's failure reason, unless one is already there.

    First writer wins, matching the launcher's own rule, so an earlier and more
    specific reason is never overwritten by a later, more generic one. Never
    raises: the traceback is already on its way to CloudWatch.
    """
    try:
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(message, encoding="utf-8")
    except OSError as exc:
        print(f"[entrypoint] could not write {path}: {exc}", flush=True)


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #


def main() -> None:
    """Run the training job on the Ray head node.

    Order is fixed and load-bearing: read the contract, validate the channels,
    record what was resolved, confirm the cluster, train, export. Every failure
    raises. ``launcher.py`` executes this module as ``__main__`` inside its own
    try/except, so an exception is how it learns the job failed: it records the
    reason in ``/opt/ml/output/failure``, tears Ray down, and exits non-zero. A
    ``sys.exit`` here would bypass that path.
    """
    started = datetime.now(timezone.utc)
    out_dir = output_data_dir()

    record: dict[str, object] = {
        "started_at": started.isoformat(),
        "stage": "startup",
    }

    try:
        resource_cfg = read_resource_config()
        hyperparameters = read_hyperparameters()
        record.update(
            {
                "current_host": resource_cfg.current_host,
                "hosts": list(resource_cfg.hosts),
                "gpus_per_node": resource_cfg.gpus_per_node,
                "node_count": resource_cfg.node_count,
                "hyperparameters": hyperparameters,
            }
        )
        write_json(out_dir, RESOLVED_CONFIG_NAME, hyperparameters)

        # Channels are validated before the trainer is touched.
        record["stage"] = "resolving-channels"
        channel_values, channel_rows, channel_files = resolve_channel_files()
        assert_channels_present(channel_rows)
        record.update({"channel_files": channel_files, "channel_rows": channel_rows})

        record["stage"] = "cluster-check"
        write_json(out_dir, RUN_RECORD_NAME, record)
        gpu_count = assert_cluster_gpus(resource_cfg.expected_gpu_count)
        record["ray"] = {
            "gpu_count": gpu_count,
            "expected_gpu_count": resource_cfg.expected_gpu_count,
        }

        record["stage"] = "training"
        argv = run_grpo.build_verl_argv(
            hyperparameters, channel_values, gpus_per_node=resource_cfg.gpus_per_node
        )
        record["verl_overrides"] = argv
        write_json(out_dir, RUN_RECORD_NAME, record)

        train_started = time.monotonic()
        exit_code = run_grpo.run(argv)
        record["training_seconds"] = round(time.monotonic() - train_started, 1)
        record["trainer_exit_code"] = exit_code

        if exit_code != 0:
            record["stage"] = "training-failed"
            raise EntrypointError(
                f"trainer exited {exit_code}; skipping export so no partial model is "
                f"uploaded. Checkpoints synced to Amazon S3 are retained and the merge "
                f"can be retried."
            )

        record["stage"] = "export"
        write_json(out_dir, RUN_RECORD_NAME, record)
        result = export_checkpoint.export()
        record["export"] = {
            "checkpoint_dir": str(result.checkpoint_dir),
            "actor_dir": str(result.actor_dir),
            "target_dir": str(result.target_dir),
            "files": list(result.files),
        }
        record["stage"] = "complete"

    except Exception as exc:  # noqa: BLE001 - the record must capture every failure
        record["stage"] = f"failed:{record.get('stage', 'unknown')}"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        # The launcher's cleanup swallows this exception on its way out and
        # reports whatever /opt/ml/output/failure holds, so the specific reason
        # is written there first. Re-raised so the launcher marks the job failed
        # and the traceback lands in CloudWatch, where an operator will read it.
        write_failure_reason(f"{record['stage']}: {record['error']}")
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["elapsed_seconds"] = round(
            (datetime.now(timezone.utc) - started).total_seconds(), 1
        )
        write_json(out_dir, RUN_RECORD_NAME, record)


if __name__ == "__main__":
    main()

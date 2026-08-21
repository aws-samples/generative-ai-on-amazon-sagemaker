"""SageMaker entry point for the GRPO training job.

``SAGEMAKER_PROGRAM=entrypoint.py`` names this file, so the ``sagemaker-training``
toolkit executes it once per training instance. It runs **only inside the GPU
container**; per Requirement 1.7 it imports nothing from ``src/grpo_sagemaker/``,
and its only imports beyond the standard library are the three sibling scripts in
this directory.

It is deliberately thin. All it does is compose, in one fixed order:

1. Read the SageMaker contract -- ``resourceconfig.json``, ``hyperparameters.json``,
   and the ``SM_CHANNEL_*`` mounts (Requirements 7.1, 7.2).
2. Resolve each channel mount to the concrete data **files** inside it, and refuse
   to continue if any declared channel holds zero rows -- before Ray is touched
   (Requirement 7.6).
3. Write the resolved configuration and a run record to ``/opt/ml/output/data``
   (Requirement 7.5).
4. Bootstrap Ray (``start_ray``), run the GRPO trainer (``run_grpo``), then merge
   and validate the checkpoint into ``/opt/ml/model`` (``export_checkpoint``).

Two ordering decisions carry real weight.

**Channels are validated before Ray starts, not after.** Requirement 7.6 says so,
and the reason is money: Ray bootstrap on a multi-GPU instance takes a minute or
more and veRL's own failure on an empty dataset surfaces deep inside a Ray actor.
Checking the parquet footers first turns a confusing mid-run traceback into an
immediate, named failure while the job has barely begun billing.

**Channels resolve to files, never to the mount directory.** veRL's
``RLHFDataset`` dispatches on a file suffix and raises ``Unsupported file format``
for a directory -- it does not expand one. So ``SM_CHANNEL_TRAIN`` pointing at
``/opt/ml/input/data/train`` has to become that directory's ``train.parquet``.
A channel holding several files becomes a Hydra list, which
``run_grpo.render_data_files`` accepts.

The export step runs only after a successful trainer exit. A failed run leaves the
checkpoints veRL synced to Amazon S3 untouched and ``/opt/ml/model`` empty, so
SageMaker uploads no half-trained model and the merge can be retried against the
retained checkpoint without repeating training.
"""

import json
import os
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import export_checkpoint
import run_grpo
import start_ray

CONFIG_DIR = Path("/opt/ml/input/config")
HYPERPARAMETERS_PATH = CONFIG_DIR / "hyperparameters.json"
OUTPUT_DATA_DIR = Path("/opt/ml/output/data")
RUN_RECORD_NAME = "run-record.json"
RESOLVED_CONFIG_NAME = "resolved-hyperparameters.json"

#: Declared channels, and the environment variable naming each mount. Both are
#: required: a GRPO run without a validation split cannot report eval metrics, and
#: the launcher always supplies both.
CHANNEL_ENV_VARS: dict[str, str] = {
    run_grpo.TRAIN_CHANNEL: "SM_CHANNEL_TRAIN",
    run_grpo.VALIDATION_CHANNEL: "SM_CHANNEL_VALIDATION",
}

#: Suffixes veRL's dataset loader accepts, in the order preferred when a channel
#: happens to hold more than one kind.
DATA_SUFFIXES: tuple[str, ...] = (".parquet", ".jsonl", ".json")

#: How long a non-head node waits for the head to finish, in the untested
#: multi-node profile. Bounded so a lost head cannot hold instances forever;
#: ``MaxRuntimeInSeconds`` on the job is the authoritative cap.
WORKER_WAIT_TIMEOUT_S = 24 * 60 * 60
WORKER_POLL_SECONDS = 30


class EntrypointError(RuntimeError):
    """Base class for every failure raised by this module."""


class ChannelError(EntrypointError):
    """A declared data channel is missing, unreadable, or empty."""


# --------------------------------------------------------------------------- #
# Pure logic. No filesystem, no Ray, no subprocess.
# --------------------------------------------------------------------------- #


def assert_channels_present(channel_rows: Mapping[str, int]) -> None:
    """Raise unless every declared channel holds at least one row.

    Pure, so Property 12 can be checked over generated mappings with no
    ``/opt/ml`` tree and no parquet files.

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
            f"cannot proceed, and this is checked before Ray starts so the job "
            f"fails now rather than inside a Ray actor. Re-run `grpo data prepare` "
            f"and confirm the manifest row counts are non-zero."
        )


def select_data_files(channel: str, names: Sequence[str]) -> list[str]:
    """Pick the data files from one channel's directory listing.

    Pure function over a listing, so the selection rule is testable without a
    filesystem.

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


# --------------------------------------------------------------------------- #
# SageMaker contract reads.
# --------------------------------------------------------------------------- #


def read_hyperparameters(path: Path = HYPERPARAMETERS_PATH) -> dict[str, str]:
    """Read ``hyperparameters.json`` as a mapping of strings.

    SageMaker writes every hyperparameter value as a JSON string, and
    ``run_grpo.build_verl_argv`` parses each one itself. Values are coerced to
    ``str`` here rather than trusted, so a hand-edited file carrying a real JSON
    number still yields the string form the override renderers expect.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EntrypointError(
            f"hyperparameters not found at {path}; this script runs only inside a "
            f"SageMaker training container"
        ) from exc
    except json.JSONDecodeError as exc:
        raise EntrypointError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise EntrypointError(
            f"{path} must contain a JSON object; got {type(payload).__name__}"
        )
    return {str(key): str(value) for key, value in payload.items()}


def read_resource_config(
    path: Path = start_ray.RESOURCE_CONFIG_PATH,
    env: Mapping[str, str] | None = None,
) -> start_ray.ResourceConfig:
    """Read the SageMaker resource configuration.

    Delegates to ``start_ray.read_resource_config``, which owns
    :class:`start_ray.ResourceConfig` because ``start_ray.bootstrap`` consumes it.
    Re-exported here because the design lists this function on the entrypoint and
    a caller should not have to know which sibling defines the type.
    """
    return start_ray.read_resource_config(path, env)


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


# --------------------------------------------------------------------------- #
# Multi-node worker wait (untested profile).
# --------------------------------------------------------------------------- #


def await_head_completion(
    timeout_s: int = WORKER_WAIT_TIMEOUT_S,
    poll_seconds: int = WORKER_POLL_SECONDS,
) -> int:
    """Block a non-head node while the head drives training.

    Part of the explicitly untested multi-node profile. A worker container that
    exited as soon as it had joined Ray would be torn down by SageMaker, taking
    its GPUs out of the cluster mid-run, so the worker has to stay alive until the
    head is finished.

    Returns 0 when the head is gone, and raises on timeout so a lost head does not
    hold instances for the whole job runtime.
    """
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - present in the container
        raise EntrypointError("ray is not importable inside the container") from exc

    print("[entrypoint] worker node: waiting for the head to finish", flush=True)
    deadline = time.monotonic() + timeout_s
    ray.init(address="auto", ignore_reinit_error=True)
    try:
        while time.monotonic() < deadline:
            alive = [node for node in ray.nodes() if node.get("Alive")]
            if len(alive) <= 1:
                print("[entrypoint] worker node: cluster wound down, exiting", flush=True)
                return 0
            time.sleep(poll_seconds)
    finally:
        ray.shutdown()

    raise EntrypointError(
        f"worker node waited {timeout_s}s and the Ray head never wound the cluster "
        f"down; failing so the instance is released"
    )


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #


def main() -> int:
    """Run the training job. Returns the process exit code.

    Order is fixed and load-bearing: read the contract, validate the channels,
    record what was resolved, bootstrap Ray, train, export. A non-zero trainer
    exit short-circuits the export, so ``/opt/ml/model`` stays empty and SageMaker
    uploads nothing.
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

        # Requirement 7.6: channels are validated before Ray is touched.
        record["stage"] = "resolving-channels"
        channel_values, channel_rows, channel_files = resolve_channel_files()
        assert_channels_present(channel_rows)
        record.update({"channel_files": channel_files, "channel_rows": channel_rows})

        record["stage"] = "ray-bootstrap"
        write_json(out_dir, RUN_RECORD_NAME, record)
        cluster = start_ray.bootstrap(resource_cfg)
        record["ray"] = {
            "head_host": cluster.head_host,
            "is_head": cluster.is_head,
            "address": cluster.address,
            "gpu_count": cluster.gpu_count,
            "expected_gpu_count": cluster.expected_gpu_count,
        }

        # Only the head drives the trainer; veRL fans work out over Ray from there.
        if not cluster.is_head:
            record["stage"] = "worker-wait"
            write_json(out_dir, RUN_RECORD_NAME, record)
            code = await_head_completion()
            record.update({"stage": "worker-complete", "exit_code": code})
            return code

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
            print(
                f"[entrypoint] trainer exited {exit_code}; skipping export so no "
                f"partial model is uploaded. Checkpoints synced to Amazon S3 are "
                f"retained and the merge can be retried.",
                flush=True,
            )
            return exit_code

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
        return 0

    except Exception as exc:  # noqa: BLE001 - the record must capture every failure
        record["stage"] = f"failed:{record.get('stage', 'unknown')}"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        # Re-raised below so SageMaker sees a non-zero exit and the traceback lands
        # in CloudWatch, where an operator will actually read it.
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["elapsed_seconds"] = round(
            (datetime.now(timezone.utc) - started).total_seconds(), 1
        )
        write_json(out_dir, RUN_RECORD_NAME, record)


if __name__ == "__main__":
    sys.exit(main())

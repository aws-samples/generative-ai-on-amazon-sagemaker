"""Merge the latest veRL FSDP checkpoint into a Hugging Face directory.

This module runs **only inside the GPU container**. Per Requirement 1.7 it imports
nothing from ``src/grpo_sagemaker/``, and its third-party imports (``transformers``,
and transitively ``torch``) are deferred into the one function that needs them, so
the pure layout logic stays importable — and therefore testable — on a workstation
with no weights, no ``transformers``, and no GPU.

Why this module exists at all: SageMaker uploads whatever it finds in
``/opt/ml/model`` when the job ends. A merge that dies halfway would therefore be
published as a working model, and the first sign of trouble would be a serving
crash days later. So the export is staged.

**The staging decision.** The merge runs into a staging directory that is a
*sibling* of ``/opt/ml/model``, is validated there, and only then has its entries
moved into place. The alternative — merge straight into ``/opt/ml/model`` and clean
up on failure — was rejected because cleanup only runs if the process survives to
run it. The two most likely failures here are exactly the ones that prevent that:
the merger being OOM-killed while materializing full weights, and
``AutoModelForCausalLM.from_pretrained`` being OOM-killed while loading them. A
``SIGKILL`` skips every ``finally`` block, and SageMaker would then upload the
partial tree. Staging makes the guarantee structural instead of best-effort:
nothing exists under ``/opt/ml/model`` until validation has already passed
(Requirements 15.5, 15.6).

The residual window is the publish loop itself, which is a series of same-filesystem
renames — no GPU, no large allocations, milliseconds. A failure there rolls back the
entries it already moved. Sibling placement is what makes those renames atomic:
``os.replace`` requires one filesystem, and a directory next to the target is on the
same one. A cross-device fallback is kept for the case where ``/opt/ml/model`` is its
own mount.

Sequence performed by :func:`export`:

1. :func:`latest_checkpoint` — resolve the newest ``global_step_N`` under
   ``/opt/ml/checkpoints``, preferring veRL's own tracker file, and fail naming the
   inspected path when there is nothing to merge (Requirement 15.4).
2. :func:`merge_to_hf` — run veRL's merger into the staging directory, failing with
   its output on a non-zero exit (Requirements 15.1, 15.5).
3. :func:`validate_hf_dir` — check the required files are present, then prove the
   directory actually loads (Requirements 15.2, 15.3, 15.6).
4. Publish into ``/opt/ml/model`` (Requirement 7.4).

Failing at any step leaves the checkpoints veRL synced to Amazon S3 untouched, so a
merge can be retried without repeating training (Requirement 15.5).

**Verified merger CLI.** veRL's ``verl/model_merger/__main__.py`` and
``base_model_merger.parse_args`` define::

    python -m verl.model_merger merge --backend fsdp --local_dir <dir> --target_dir <dir>

Two details that are easy to get wrong and are not obvious from the flag names.
``--local_dir`` is the ``actor`` subdirectory of a ``global_step_N`` checkpoint, not
the ``global_step_N`` directory itself: the merger derives its Hugging Face config
path as ``<local_dir>/huggingface``, and veRL's FSDP checkpoint manager writes that
tree under ``global_step_N/actor/``. Pointing at the step directory fails inside
``AutoConfig.from_pretrained`` with a confusing message, so
:func:`resolve_actor_dir` does the descent explicitly. ``--target_dir`` defaults to
the literal string ``tmp``, so it is always passed.
"""

import argparse
import errno
import gc
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

CHECKPOINT_ROOT = Path("/opt/ml/checkpoints")
"""Where ``run_grpo.py`` points ``trainer.default_local_dir`` (Requirement 7.3).

Must equal ``run_grpo.CHECKPOINT_DIR``. Repeated rather than imported because both
values are fixed by the ``/opt/ml`` contract, not by either module, and these two
scripts are siblings that ``entrypoint.py`` composes rather than a dependency chain.
"""

MODEL_DIR = Path("/opt/ml/model")
"""SageMaker uploads this directory's contents verbatim (Requirement 7.4)."""

STAGING_DIR_NAME = ".hf-export-staging"
"""Staging directory name, created as a sibling of :data:`MODEL_DIR`.

A sibling rather than a child for two independent reasons: a child would be
uploaded by SageMaker, and a sibling is guaranteed to share a filesystem with the
target, which is what makes the publish renames atomic.
"""

TRACKER_FILENAME = "latest_checkpointed_iteration.txt"
"""veRL's checkpoint tracker (``verl.utils.checkpoint.checkpoint_manager``)."""

ACTOR_SUBDIR = "actor"
HF_CONFIG_SUBDIR = "huggingface"

REQUIRED_HF_FILES = ("config.json", "tokenizer_config.json", "generation_config.json")
"""Files Requirement 15.2 requires in the merged directory.

Deliberately duplicated from ``grpo_sagemaker.export_model``: Requirement 1.7
forbids this module from importing the control-plane package, and a shared constant
would create exactly that import. ``tests/test_deploy_cleanup.py`` and
``tests/test_training_scripts.py`` can assert the two tuples agree.
"""

WEIGHT_SHARD_REQUIREMENT = "a weight shard (model*.safetensors or pytorch_model*.bin)"
"""Name reported for the shard requirement when no shard is present.

Not a filename, because the requirement is satisfied by any of several real names.
Exported as a constant so tests can assert against it without duplicating prose.
"""

MERGER_MODULE = "verl.model_merger"
MERGER_BACKEND = "fsdp"

DEFAULT_MERGE_TIMEOUT_S = 3600
"""Wall-clock budget for the merger subprocess.

Backstop only. ``MaxRuntimeInSeconds`` on the training job (Requirement 11.6) is the
authoritative bound; this exists so a merger that stalls after training has already
succeeded surfaces as a named failure rather than an unexplained run to the job
deadline.
"""

_MERGER_LOG_TAIL_LINES = 200
"""Merger output lines retained for the failure report.

Bounded because the merger prints per-shard progress: a long merge could otherwise
accumulate an unbounded string in memory for a report nobody reads on success.
"""

_STEP_DIR_PATTERN = re.compile(r"^global_step_(\d+)$")

# Hugging Face's real shard names. Both single-file and sharded forms are accepted;
# index files (`model.safetensors.index.json`) and adapter weights
# (`adapter_model.safetensors`) deliberately are not, because neither carries the
# full model.
_SAFETENSORS_SHARD_PATTERN = re.compile(r"^model(-\d+-of-\d+)?\.safetensors$")
_TORCH_BIN_SHARD_PATTERN = re.compile(r"^pytorch_model(-\d+-of-\d+)?\.bin$")

_MODEL_LOAD_KWARGS = {"low_cpu_mem_usage": True, "torch_dtype": "auto"}
"""Kwargs for the verification load.

``low_cpu_mem_usage`` keeps a large actor from needing twice its size in host RAM.
``torch_dtype="auto"`` honours the dtype recorded in ``config.json`` instead of
upcasting to fp32. Both are accepted by every ``transformers`` release the veRL base
image pins; ``torch_dtype`` is the name to revisit if that base moves to a release
that has completed the rename to ``dtype``.
"""


class ExportError(RuntimeError):
    """Base class for every failure raised by this module."""


class CheckpointNotFoundError(ExportError):
    """No mergeable veRL checkpoint exists (Requirement 15.4)."""


class MergeFailedError(ExportError):
    """``verl.model_merger`` exited non-zero or stalled (Requirement 15.5)."""


class HfLayoutError(ExportError):
    """The merged directory is missing required files (Requirement 15.2)."""


class HfLoadError(ExportError):
    """``AutoTokenizer`` or ``AutoModelForCausalLM`` refused the directory.

    Requirement 15.6.
    """


class PublishError(ExportError):
    """The validated model could not be moved into ``/opt/ml/model``."""


@dataclass(frozen=True)
class ExportResult:
    """What :func:`export` proved before SageMaker was allowed to upload."""

    checkpoint_dir: Path
    """The ``global_step_N`` directory that was merged."""

    actor_dir: Path
    """The ``actor`` subdirectory handed to the merger as ``--local_dir``."""

    target_dir: Path
    """Where the validated model now lives, normally ``/opt/ml/model``."""

    files: tuple[str, ...]
    """Filenames published, captured before the move so the report is complete."""


# --------------------------------------------------------------------------- #
# Pure logic. No filesystem, no subprocess, no transformers, no torch.
# --------------------------------------------------------------------------- #


def parse_step_dirs(names: Sequence[str]) -> tuple[tuple[int, str], ...]:
    """Extract ``(step, name)`` pairs for ``global_step_N`` entries, ascending.

    Non-matching names are ignored rather than rejected: veRL writes
    ``latest_checkpointed_iteration.txt`` into the same directory, and future veRL
    versions may add more bookkeeping files there.
    """
    steps = [
        (int(match.group(1)), name)
        for name, match in ((name, _STEP_DIR_PATTERN.match(name)) for name in names)
        if match is not None
    ]
    return tuple(sorted(steps))


def select_latest_step(names: Sequence[str], tracker_step: int | None = None) -> str:
    """Return the ``global_step_N`` name to merge, from a listing alone.

    ``tracker_step`` is veRL's own record of the last completed save and wins when
    the corresponding directory is present in ``names``. It is preferred over the
    numerically highest step because a step directory can exist while still being
    written — the tracker is only updated once the save has finished, so trusting it
    avoids merging a torn checkpoint.

    When the tracker names a step that is absent, the highest present step is used
    instead. Failing outright would discard a good checkpoint, and the training GPU
    hours behind it, over stale bookkeeping. The caller reports the divergence.

    Raises :class:`CheckpointNotFoundError` when no name matches, which is the empty
    and the bookkeeping-only listing alike.
    """
    steps = parse_step_dirs(names)
    if not steps:
        raise CheckpointNotFoundError(
            f"no 'global_step_N' checkpoint directory found among {sorted(names)!r}"
        )

    if tracker_step is not None:
        for step, name in steps:
            if step == tracker_step:
                return name

    return steps[-1][1]


def is_weight_shard(name: str) -> bool:
    """Return whether ``name`` is a full-model weight shard.

    Accepts the real Hugging Face spellings — ``model.safetensors``,
    ``model-00001-of-00002.safetensors``, ``pytorch_model.bin`` and its sharded
    form — because which one appears depends on the model's size and on the
    ``safetensors`` version in the base image. Shard-count widths are matched
    loosely; the current five-digit convention is not assumed.
    """
    return bool(
        _SAFETENSORS_SHARD_PATTERN.match(name) or _TORCH_BIN_SHARD_PATTERN.match(name)
    )


def missing_hf_files(listing: Sequence[str]) -> tuple[str, ...]:
    """Return every unmet requirement for a Hugging Face model directory.

    Pure function over a sequence of filenames, so the layout rule is testable
    without weights, ``transformers``, or a filesystem — this is the core
    :func:`validate_hf_dir` wraps (Property 26).

    Returns an empty tuple exactly when ``config.json``, ``tokenizer_config.json``,
    ``generation_config.json``, and at least one weight shard are all present.
    Otherwise it returns *every* unmet requirement, not just the first: an operator
    reading a failed job log should learn the full gap in one pass. Missing files
    are named in :data:`REQUIRED_HF_FILES` order followed by
    :data:`WEIGHT_SHARD_REQUIREMENT`, so the report is deterministic.

    The empty listing yields all four entries and therefore always fails.
    """
    present = set(listing)
    missing = [name for name in REQUIRED_HF_FILES if name not in present]
    if not any(is_weight_shard(name) for name in listing):
        missing.append(WEIGHT_SHARD_REQUIREMENT)
    return tuple(missing)


# --------------------------------------------------------------------------- #
# Filesystem inspection.
# --------------------------------------------------------------------------- #


def read_tracker_step(checkpoint_root: Path) -> int | None:
    """Read veRL's ``latest_checkpointed_iteration.txt``, or ``None``.

    Returns ``None`` for an absent or unparseable tracker rather than raising. The
    tracker is a hint that lets :func:`select_latest_step` skip a half-written
    directory; when it is unreadable the step directories themselves are still
    authoritative, and failing the export over corrupt bookkeeping would throw away
    a usable model.
    """
    tracker = checkpoint_root / TRACKER_FILENAME
    try:
        raw = tracker.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        print(f"[export_checkpoint] could not read {tracker}: {exc}", flush=True)
        return None

    try:
        return int(raw)
    except ValueError:
        print(
            f"[export_checkpoint] {tracker} does not contain an integer step "
            f"(got {raw!r}); falling back to the highest present step",
            flush=True,
        )
        return None


def latest_checkpoint(checkpoint_root: Path = CHECKPOINT_ROOT) -> Path:
    """Return the newest mergeable checkpoint directory under ``checkpoint_root``.

    Postcondition: the returned path is an existing, non-empty directory.

    Raises :class:`CheckpointNotFoundError` reporting the inspected path when the
    root is absent, holds no ``global_step_N`` directory, or the chosen directory is
    empty (Requirement 15.4). All three mean the same thing to an operator —
    training produced nothing to export — and the path is what tells them whether to
    look at ``trainer.default_local_dir`` or at the training logs.
    """
    if not checkpoint_root.is_dir():
        raise CheckpointNotFoundError(
            f"checkpoint directory {checkpoint_root} does not exist; expected veRL "
            f"to have written checkpoints there via trainer.default_local_dir"
        )

    names = sorted(entry.name for entry in checkpoint_root.iterdir())
    if not names:
        raise CheckpointNotFoundError(
            f"checkpoint directory {checkpoint_root} is empty; training wrote no "
            f"checkpoint, so there is nothing to merge"
        )

    tracker_step = read_tracker_step(checkpoint_root)
    try:
        chosen = select_latest_step(names, tracker_step)
    except CheckpointNotFoundError as exc:
        raise CheckpointNotFoundError(f"in checkpoint directory {checkpoint_root}: {exc}") from exc

    if tracker_step is not None and chosen != f"global_step_{tracker_step}":
        print(
            f"[export_checkpoint] {TRACKER_FILENAME} names step {tracker_step}, whose "
            f"directory is absent; merging {chosen} instead",
            flush=True,
        )

    checkpoint_dir = checkpoint_root / chosen
    if not any(checkpoint_dir.iterdir()):
        raise CheckpointNotFoundError(
            f"checkpoint directory {checkpoint_dir} is empty; the save appears to "
            f"have been interrupted"
        )

    print(f"[export_checkpoint] selected checkpoint {checkpoint_dir}", flush=True)
    return checkpoint_dir


def resolve_actor_dir(checkpoint_dir: Path) -> Path:
    """Return the directory to pass to the merger as ``--local_dir``.

    veRL writes the actor's FSDP shards and its ``huggingface/`` config tree under
    ``global_step_N/actor/``, and the merger reads its config from
    ``<local_dir>/huggingface``. So the merger wants the ``actor`` directory, and
    handing it ``global_step_N`` fails deep inside ``AutoConfig.from_pretrained``
    with a message that does not mention the real mistake.

    Accepts either shape — a ``global_step_N`` directory or an ``actor`` directory
    already — so a caller pointing straight at an actor directory (a retry, or a
    manual re-run of :func:`merge_to_hf`) works. Raises :class:`ExportError` naming
    what was looked for and what was found otherwise.
    """
    actor_dir = checkpoint_dir / ACTOR_SUBDIR
    if not actor_dir.is_dir():
        if (checkpoint_dir / HF_CONFIG_SUBDIR).is_dir():
            return checkpoint_dir
        found = sorted(entry.name for entry in checkpoint_dir.iterdir())
        raise ExportError(
            f"{checkpoint_dir} contains neither an {ACTOR_SUBDIR!r} subdirectory nor "
            f"a {HF_CONFIG_SUBDIR!r} subdirectory, so it is not a veRL FSDP "
            f"checkpoint; found {found!r}"
        )

    if not (actor_dir / HF_CONFIG_SUBDIR).is_dir():
        raise ExportError(
            f"{actor_dir} has no {HF_CONFIG_SUBDIR!r} subdirectory, which "
            f"{MERGER_MODULE} requires as its model config path "
            f"(<local_dir>/{HF_CONFIG_SUBDIR}); the checkpoint is incomplete"
        )

    return actor_dir


# --------------------------------------------------------------------------- #
# Merge.
# --------------------------------------------------------------------------- #


def _stream_command(argv: Sequence[str], timeout_s: int) -> tuple[int, str]:
    """Run ``argv``, echoing its output live, and return ``(exit_code, tail)``.

    Output is streamed rather than captured wholesale because the merge is a single
    step that can run for many minutes: with output buffered until exit, a healthy
    merge and a hung one look identical in CloudWatch. A bounded tail is retained
    alongside for the failure report (Requirement 15.5).

    The deadline is checked between output lines, so a merger that stalls *while
    printing* is killed here. One that stalls having emitted nothing blocks on
    ``readline`` and is bounded instead by the job's ``MaxRuntimeInSeconds``. That
    residual case is accepted rather than paid for with a reader thread: it is
    indistinguishable from a merger doing slow work silently, and the job-level
    deadline already caps the cost.
    """
    print(f"[export_checkpoint] $ {' '.join(argv)}", flush=True)
    deadline = time.monotonic() + timeout_s
    tail: deque[str] = deque(maxlen=_MERGER_LOG_TAIL_LINES)

    # argv is constructed in merge_to_hf from module constants and resolved paths;
    # no shell is involved and no element originates from untrusted input.
    process = subprocess.Popen(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stream = process.stdout
    if stream is None:  # pragma: no cover - guaranteed by stdout=PIPE
        process.kill()
        raise MergeFailedError(f"could not capture output of {' '.join(argv)}")

    try:
        for line in stream:
            stripped = line.rstrip("\n")
            tail.append(stripped)
            print(f"[{MERGER_MODULE}] {stripped}", flush=True)
            if time.monotonic() > deadline:
                process.kill()
                raise MergeFailedError(
                    f"{MERGER_MODULE} exceeded its {timeout_s}s budget and was "
                    f"killed. Last output:\n" + "\n".join(tail)
                )
        remaining = max(1, int(deadline - time.monotonic()))
        return process.wait(timeout=remaining), "\n".join(tail)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        raise MergeFailedError(
            f"{MERGER_MODULE} closed its output but did not exit within "
            f"{timeout_s}s. Last output:\n" + "\n".join(tail)
        ) from exc
    finally:
        if process.poll() is None:
            process.kill()
        # Reap the child so a killed merger does not linger as a zombie for the
        # remainder of the job.
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        stream.close()


def merge_to_hf(
    checkpoint_dir: Path,
    target_dir: Path,
    *,
    python_executable: str | None = None,
    timeout_s: int = DEFAULT_MERGE_TIMEOUT_S,
) -> None:
    """Merge a veRL FSDP checkpoint into ``target_dir`` in Hugging Face format.

    Precondition: ``checkpoint_dir`` is a ``global_step_N`` or ``actor`` directory
    holding a complete FSDP save.

    Postcondition on success: ``target_dir`` holds the merger's output, unvalidated.
    Callers pass a staging directory here, never ``/opt/ml/model`` — see the module
    docstring — and only :func:`export` publishes.

    Raises :class:`MergeFailedError` carrying the merger's output on a non-zero exit
    or a stall (Requirement 15.5). Checkpoints veRL synced to Amazon S3 are not
    touched on any path through this function, so a failed merge can be retried
    without repeating training.
    """
    actor_dir = resolve_actor_dir(checkpoint_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        python_executable or sys.executable,
        "-m",
        MERGER_MODULE,
        "merge",
        "--backend",
        MERGER_BACKEND,
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(target_dir),
    ]

    exit_code, output = _stream_command(argv, timeout_s)
    if exit_code != 0:
        raise MergeFailedError(
            f"{MERGER_MODULE} exited {exit_code} merging {actor_dir} into "
            f"{target_dir}. The synced checkpoints in Amazon S3 are unchanged, so "
            f"this merge can be retried without rerunning training. Merger "
            f"output:\n{output}"
        )

    print(f"[export_checkpoint] merge completed into {target_dir}", flush=True)


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #


def _load_with_transformers(target_dir: Path) -> None:
    """Prove ``target_dir`` loads as a tokenizer and a causal LM.

    ``transformers`` is imported here, not at module scope, so every function above
    stays importable without it (and without ``torch``, which it drags in).

    ``trust_remote_code`` is deliberately left off. Enabling it would execute Python
    that the merger copied out of the checkpoint, turning a validation step into
    arbitrary code execution inside the training job; a model architecture needing
    it should be a considered change, not a default.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - present in the veRL container
        raise ExportError(
            "the 'transformers' package is not importable; export_checkpoint runs "
            "only inside the veRL GPU container, which ships it"
        ) from exc

    # Both loads catch Exception deliberately. transformers signals a bad directory
    # with OSError, ValueError, KeyError, and its own error types depending on which
    # file is wrong, and the requirement is the same for all of them: fail, and leave
    # /opt/ml/model without a partial model. Narrowing the clause would let an
    # unanticipated type escape as an unlabelled traceback.
    print(f"[export_checkpoint] loading tokenizer from {target_dir}", flush=True)
    try:
        AutoTokenizer.from_pretrained(str(target_dir))
    except Exception as exc:
        raise HfLoadError(f"AutoTokenizer.from_pretrained({target_dir}) failed: {exc}") from exc

    print(f"[export_checkpoint] loading model from {target_dir}", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(str(target_dir), **_MODEL_LOAD_KWARGS)
    except Exception as exc:
        raise HfLoadError(
            f"AutoModelForCausalLM.from_pretrained({target_dir}) failed: {exc}"
        ) from exc

    # Release the weights before publishing. The load can hold many GiB of host RAM,
    # and the publish that follows has no reason to compete with it.
    del model
    gc.collect()
    print("[export_checkpoint] tokenizer and model both loaded", flush=True)


def validate_hf_dir(target_dir: Path) -> None:
    """Confirm ``target_dir`` is a complete, loadable Hugging Face model directory.

    Two checks, cheapest first. :func:`missing_hf_files` over the directory listing
    catches an incomplete merge in milliseconds (Requirement 15.2), then the
    ``transformers`` load proves the files are not merely present but coherent
    (Requirement 15.3).

    Raises :class:`HfLayoutError` naming every missing file, or
    :class:`HfLoadError` if either load fails (Requirement 15.6). Writes nothing and
    deletes nothing: the caller decides what happens to a rejected directory.
    """
    if not target_dir.is_dir():
        raise HfLayoutError(f"merged model directory {target_dir} does not exist")

    listing = sorted(entry.name for entry in target_dir.iterdir() if entry.is_file())
    missing = missing_hf_files(listing)
    if missing:
        raise HfLayoutError(
            f"merged model directory {target_dir} is missing {len(missing)} required "
            f"item(s): {', '.join(missing)}. Present files: {listing!r}. A missing "
            f"generation_config.json usually means the base model shipped none, so "
            f"veRL had none to propagate; a missing weight shard means the merge did "
            f"not finish."
        )

    print(f"[export_checkpoint] layout check passed: {listing!r}", flush=True)
    _load_with_transformers(target_dir)


# --------------------------------------------------------------------------- #
# Publish.
# --------------------------------------------------------------------------- #


def _move_entry(source: Path, destination: Path) -> None:
    """Move one entry, atomically when the filesystem allows it.

    ``os.replace`` is a single rename, so an interruption leaves the entry either
    fully at the source or fully at the destination. It requires both paths on one
    filesystem, which sibling staging guarantees — unless ``/opt/ml/model`` is its
    own mount, hence the ``EXDEV`` fallback to a copy.
    """
    try:
        os.replace(source, destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.move(str(source), str(destination))


def _publish(staging_dir: Path, target_dir: Path) -> None:
    """Move a validated model from staging into ``target_dir``.

    Precondition: ``staging_dir`` has passed :func:`validate_hf_dir`, and
    ``target_dir`` is absent or empty. A non-empty target is refused rather than
    merged into, because mixing an unknown prior tree with these weights could
    produce a directory that validates while serving something else.

    Postcondition: either every entry is published, or ``target_dir`` is empty
    again. Entries already moved are rolled back on failure so a partial model is
    never left for SageMaker to upload (Requirement 15.6).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(entry.name for entry in target_dir.iterdir())
    if existing:
        raise PublishError(
            f"refusing to publish into non-empty {target_dir}; found {existing!r}. "
            f"Something other than this export wrote there, and merging into it "
            f"could publish a directory that validates while serving different "
            f"weights. Empty {target_dir} and re-run the export; the veRL "
            f"checkpoints it merges from are untouched."
        )

    published: list[Path] = []
    try:
        for entry in sorted(staging_dir.iterdir()):
            destination = target_dir / entry.name
            _move_entry(entry, destination)
            published.append(destination)
    except OSError as exc:
        for path in published:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        raise PublishError(
            f"failed to publish {staging_dir} into {target_dir}: {exc}. Every "
            f"already-moved entry was removed, so {target_dir} is empty and "
            f"SageMaker will upload no partial model."
        ) from exc

    shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"[export_checkpoint] published merged model to {target_dir}", flush=True)


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #


def export(
    checkpoint_root: Path = CHECKPOINT_ROOT,
    target_dir: Path = MODEL_DIR,
    *,
    staging_dir: Path | None = None,
    timeout_s: int = DEFAULT_MERGE_TIMEOUT_S,
) -> ExportResult:
    """Merge, validate, and publish the latest checkpoint. Called by ``entrypoint``.

    Postcondition on success: ``target_dir`` holds a Hugging Face model directory
    that has been proven complete and loadable.

    Postcondition on *any* failure: ``target_dir`` is empty, so SageMaker uploads
    nothing (Requirements 15.5, 15.6). The staging directory is removed on failure
    too — it is not uploaded, and a merged multi-billion-parameter model is several
    gigabytes of a finite volume that a retry would need. The diagnostic value is
    kept where it is actually readable: the merger's output is in the raised error
    and in the job log.
    """
    staging = staging_dir if staging_dir is not None else target_dir.parent / STAGING_DIR_NAME
    if staging == target_dir or target_dir in staging.parents:
        raise ExportError(
            f"staging directory {staging} must be outside {target_dir}; SageMaker "
            f"uploads everything under {target_dir}, including a failed merge"
        )

    checkpoint_dir = latest_checkpoint(checkpoint_root)
    actor_dir = resolve_actor_dir(checkpoint_dir)

    # A staging directory left by an earlier attempt would let stale files satisfy
    # the layout check, so start from nothing.
    shutil.rmtree(staging, ignore_errors=True)

    files: tuple[str, ...] = ()
    published = False
    try:
        merge_to_hf(checkpoint_dir, staging, timeout_s=timeout_s)
        validate_hf_dir(staging)
        # Captured before the move, because publishing empties the staging directory.
        files = tuple(sorted(entry.name for entry in staging.iterdir()))
        _publish(staging, target_dir)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)

    return ExportResult(
        checkpoint_dir=checkpoint_dir,
        actor_dir=actor_dir,
        target_dir=target_dir,
        files=files,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the export standalone, for re-running a merge without retraining.

    Exceptions are intentionally not caught: the traceback in the job log is more
    useful than a swallowed exit code, and a non-zero exit is what fails the job.
    """
    parser = argparse.ArgumentParser(
        description="Merge the latest veRL FSDP checkpoint into a Hugging Face directory."
    )
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--target-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_MERGE_TIMEOUT_S)
    args = parser.parse_args(argv)

    result = export(args.checkpoint_root, args.target_dir, timeout_s=args.timeout_s)
    print(
        f"[export_checkpoint] exported {result.checkpoint_dir} -> {result.target_dir} "
        f"({len(result.files)} files)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

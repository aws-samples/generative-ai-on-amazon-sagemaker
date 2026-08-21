"""Build-time gate for the training container.

This script runs as the last step of ``container/Dockerfile``. Its whole purpose
is to move a stack violation from a paid SageMaker training job to a free
CodeBuild failure: if the resolved base image drifts below the veRL version
floor, or the ``/opt/ml`` contract is incomplete, the build fails here with the
violated constraint named (Requirements 6.7, 6.8).

It checks three independent things:

1. **The stack imports.** ``verl``, ``ray``, ``vllm``, and ``torch`` must all
   import (Requirement 6.7). Import success is itself a meaningful assertion --
   a mismatched CUDA or torch ABI usually surfaces as an ImportError.
2. **The versions clear the floor.** vLLM >= 0.18.0, Python >= 3.10, CUDA >= 12.8
   (Requirement 6.8). These are veRL's own published requirements, restated here
   so the image cannot quietly regress.
3. **The SageMaker contract is in place.** The ``/opt/ml`` directories the
   container owns exist and are writable, ``sagemaker_training`` imports, its
   ``train`` console script resolves on ``PATH``, and the file named by
   ``SAGEMAKER_PROGRAM`` is present in ``/opt/ml/code``.

**The version comparison is a pure function.** :func:`check_versions` and its
helpers touch no import, no filesystem, and no environment, so they are testable
on a workstation with no GPU and no veRL installed -- which is the only way
``tests/test_container.py`` can property-test them at all. Every import of
``verl`` / ``ray`` / ``vllm`` / ``torch`` is deferred into
:func:`detect_stack`, below the pure section, for that reason: importing this
module must stay cheap and dependency-free.

Run standalone for diagnostics inside a running container:

    python /opt/ml/code/verify_stack.py

Exit status is 0 when every check passes and 1 when any check fails, with all
violations reported rather than just the first.
"""

import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constraints. veRL's published floor, restated so a base-image regression is
# caught here instead of at training time.
# --------------------------------------------------------------------------- #

MIN_VLLM = (0, 18, 0)
MIN_PYTHON = (3, 10)
MIN_CUDA = (12, 8)

OPT_ML = Path("/opt/ml")
CODE_DIR = OPT_ML / "code"

REQUIRED_DIRS: tuple[Path, ...] = (
    CODE_DIR,
    OPT_ML / "input" / "config",
    OPT_ML / "input" / "data",
    OPT_ML / "checkpoints",
    OPT_ML / "model",
    OPT_ML / "output" / "data",
)
"""The ``/opt/ml`` paths from the design's contract table.

SageMaker mounts over most of these at runtime, so their presence at build time
proves only that the image declares the layout it will write to. That is still
worth asserting: a missing ``/opt/ml/checkpoints`` means veRL's checkpoint sync
has nowhere to write, and the failure would otherwise land mid-run.
"""

_ENTRY_POINT_SCRIPT = "train"
"""The console script ``sagemaker-training`` installs; SageMaker invokes it."""

_LEADING_INTS = re.compile(r"^\D*?(\d+(?:\.\d+)*)")


class StackVerificationError(RuntimeError):
    """One or more checks failed. Carries every violation, not just the first."""

    def __init__(self, violations: tuple[str, ...]) -> None:
        self.violations = violations
        super().__init__(f"{len(violations)} stack constraint(s) violated")


@dataclass(frozen=True)
class StackVersions:
    """The version tuples the constraint check is a function of.

    Carried as data rather than read from the environment so
    :func:`check_versions` stays pure. ``verl`` and ``ray`` versions are recorded
    for the pinning matrix (Requirement 6.4) but carry no floor of their own --
    veRL's constraints are expressed through vLLM, Python, and CUDA.
    """

    vllm: tuple[int, ...]
    python: tuple[int, ...]
    cuda: tuple[int, ...]
    verl: str = "unknown"
    ray: str = "unknown"
    torch: str = "unknown"
    transformers: str = "unknown"


# --------------------------------------------------------------------------- #
# Pure logic. No imports of verl/ray/vllm/torch, no filesystem, no environment.
# Everything below this banner and above the next one is safe to call from a
# workstation test.
# --------------------------------------------------------------------------- #


def parse_version(raw: str) -> tuple[int, ...]:
    """Return the leading dotted-integer run of ``raw`` as a tuple.

    Real version strings from this stack are messy -- ``2.6.0+cu128``,
    ``0.11.0rc1``, ``0.20.2.dev0``, ``12.8`` -- so only the leading numeric run
    is significant and everything after it is discarded. Pre-release and local
    suffixes are deliberately ignored: treating ``0.18.0rc1`` as ``0.18.0``
    errs toward accepting a release candidate of a satisfying version, which is
    the right call for a floor check.

    An unparseable string yields ``()`` rather than raising, so a surprising
    version string becomes a reported violation instead of a traceback.

    A zero-padded component also yields ``()``. Canonical PEP 440 versions never
    pad, so padding means the digits are a packed form whose reading cannot be
    recovered here -- ``011`` is veRL's tag shorthand for 0.11, and reading it as
    11 would clear a 0.18.0 floor that 0.11 must fail. This gate exists to catch
    what the resolver in ``build_image.py`` missed, so it must be at least as
    strict as the resolver, never looser.
    """
    match = _LEADING_INTS.match(raw.strip())
    if match is None:
        return ()
    parts = match.group(1).split(".")
    if any(len(part) > 1 and part.startswith("0") for part in parts):
        return ()
    return tuple(int(part) for part in parts)


def format_version(version: tuple[int, ...]) -> str:
    """Render a version tuple for a human-readable violation message."""
    return ".".join(str(part) for part in version) if version else "unparseable"


def at_least(actual: tuple[int, ...], minimum: tuple[int, ...]) -> bool:
    """Return whether ``actual`` is at least ``minimum``, compared component-wise.

    Both tuples are zero-padded to a common length before comparison. Comparing
    raw tuples would be wrong: Python evaluates ``(0, 18) >= (0, 18, 0)`` as
    ``False``, so a base image reporting vLLM as ``0.18`` would be rejected for
    failing a floor it actually meets. Padding makes ``0.18`` and ``0.18.0``
    compare equal, which is what the constraint means.
    """
    width = max(len(actual), len(minimum))
    padded_actual = tuple(actual) + (0,) * (width - len(actual))
    padded_minimum = tuple(minimum) + (0,) * (width - len(minimum))
    return padded_actual >= padded_minimum


def check_versions(versions: StackVersions) -> tuple[str, ...]:
    """Return one message per violated version constraint; empty when all pass.

    Pure: a function of ``versions`` alone. Returns every violation rather than
    short-circuiting so one build reports the full picture instead of forcing a
    rebuild per constraint (Requirement 6.8).
    """
    checks = (
        ("vLLM", versions.vllm, MIN_VLLM),
        ("Python", versions.python, MIN_PYTHON),
        ("CUDA", versions.cuda, MIN_CUDA),
    )
    return tuple(
        f"{label} {format_version(actual)} is below the required minimum {format_version(minimum)}"
        for label, actual, minimum in checks
        if not at_least(actual, minimum)
    )


# --------------------------------------------------------------------------- #
# Runtime probes. These import the GPU stack and touch the filesystem, so they
# only work inside the container.
# --------------------------------------------------------------------------- #


def detect_stack() -> StackVersions:
    """Import the GPU stack and read its versions (Requirement 6.7).

    Imports are local to this function so the pure section above stays reachable
    on a workstation. An ImportError propagates unchanged: a stack that cannot
    import is a build failure, and the original traceback names the cause far
    better than any message this script could synthesise.
    """
    import ray
    import torch
    import verl
    import vllm

    try:
        import transformers

        transformers_version = getattr(transformers, "__version__", "unknown")
    except ImportError:
        # transformers carries no floor of its own; record it if present and
        # move on rather than failing a build over a missing nice-to-have.
        transformers_version = "absent"

    cuda_raw = torch.version.cuda or ""

    return StackVersions(
        vllm=parse_version(getattr(vllm, "__version__", "")),
        python=sys.version_info[:3],
        cuda=parse_version(cuda_raw),
        verl=getattr(verl, "__version__", "unknown"),
        ray=getattr(ray, "__version__", "unknown"),
        torch=getattr(torch, "__version__", "unknown"),
        transformers=transformers_version,
    )


def check_opt_ml_layout(root: Path = OPT_ML) -> tuple[str, ...]:
    """Return one message per missing or unwritable contract directory.

    ``root`` is rebindable so a test can point the same check at a temporary
    tree instead of the real ``/opt/ml``.
    """
    violations: list[str] = []
    for required in REQUIRED_DIRS:
        path = root / required.relative_to(OPT_ML)
        if not path.is_dir():
            violations.append(f"required directory {path} is missing")
        elif not os.access(path, os.W_OK):
            violations.append(f"required directory {path} is not writable")
    return tuple(violations)


def check_sagemaker_entry_point(code_dir: Path = CODE_DIR) -> tuple[str, ...]:
    """Return one message per missing piece of the SageMaker entry-point wiring.

    Three separate failure modes, each of which would otherwise only appear once
    a training job had already started billing: the toolkit not installed, its
    console script not on ``PATH``, and ``SAGEMAKER_PROGRAM`` naming a file that
    was never copied into the image.
    """
    violations: list[str] = []

    try:
        import sagemaker_training  # noqa: F401
    except ImportError as exc:
        violations.append(
            f"sagemaker-training is not importable ({exc}); the /opt/ml contract "
            "is unimplemented -- check container/requirements-container.txt, "
            "which must carry the toolkit's full closure because the install "
            "runs with --no-deps"
        )

    if shutil.which(_ENTRY_POINT_SCRIPT) is None:
        violations.append(
            f"the {_ENTRY_POINT_SCRIPT!r} console script from sagemaker-training "
            "is not on PATH; SageMaker invokes it to start training"
        )

    program = os.environ.get("SAGEMAKER_PROGRAM")
    if not program:
        violations.append("SAGEMAKER_PROGRAM is unset; SageMaker has no entry point to run")
    elif not (code_dir / program).is_file():
        violations.append(
            f"SAGEMAKER_PROGRAM={program!r} names a file that is absent from {code_dir}"
        )

    return tuple(violations)


def verify() -> StackVersions:
    """Run every check. Return the detected versions, or raise with all failures."""
    versions = detect_stack()
    violations = (
        *check_versions(versions),
        *check_opt_ml_layout(),
        *check_sagemaker_entry_point(),
    )
    if violations:
        raise StackVerificationError(violations)
    return versions


def format_report(versions: StackVersions) -> str:
    """Render the resolved versions for the CodeBuild log and the pinning matrix.

    The build captures this text as the version report that populates
    ``docs/container.md`` (Requirement 6.4), so the field names are stable.
    """
    rows = (
        ("verl", versions.verl),
        ("vllm", format_version(versions.vllm)),
        ("ray", versions.ray),
        ("torch", versions.torch),
        ("transformers", versions.transformers),
        ("cuda", format_version(versions.cuda)),
        ("python", format_version(versions.python)),
    )
    width = max(len(name) for name, _ in rows)
    lines = [f"{name:<{width}} {value}" for name, value in rows]
    return "\n".join(lines)


def main() -> int:
    try:
        versions = verify()
    except StackVerificationError as exc:
        print("[verify_stack] FAILED", file=sys.stderr)
        for violation in exc.violations:
            print(f"[verify_stack]   - {violation}", file=sys.stderr)
        print(
            "[verify_stack] the container build is refused so this never becomes "
            "a training-job failure; see docs/container.md for the version floor",
            file=sys.stderr,
        )
        return 1

    print("[verify_stack] resolved stack:")
    print(format_report(versions))
    print(
        f"[verify_stack] OK - vllm >= {format_version(MIN_VLLM)}, "
        f"python >= {format_version(MIN_PYTHON)}, cuda >= {format_version(MIN_CUDA)}, "
        "/opt/ml contract present"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

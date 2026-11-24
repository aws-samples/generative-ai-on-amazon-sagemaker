"""Tiny cross-notebook state: work/<run>/state.json, or instructor/state.json (git-ignored) for the
instructor run (S3 URIs, job names, endpoint/IC names).

Safe for concurrent writers: background arm threads, the notebook kernel and the Streamlit app all
use the same file. Every write takes a process lock plus an exclusive file lock (fcntl.flock on a
sidecar ``state.json.lock``), re-reads the file inside the lock, merges, and replaces the file
atomically (temp file in the same directory + os.replace), so readers never see a partial file and
concurrent updates are never lost.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from .config import Settings

_LOCK = threading.RLock()          # serializes writers inside one process (arm threads + kernel)


def _path(s: Settings) -> Path:
    """work/<run>/state.json, or <tracked_dir>/state.json for version-controlled runs (instructor)."""
    if s.tracked_dir is not None:
        s.tracked_dir.mkdir(parents=True, exist_ok=True)
        return s.tracked_dir / "state.json"
    return s.local("state.json")


@contextlib.contextmanager
def _locked(p: Path):
    """Process lock + exclusive advisory file lock on <state>.lock (held for the whole read-merge-write)."""
    with _LOCK:
        with open(p.with_name(p.name + ".lock"), "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)


def _read(p: Path, retries: int = 20, delay: float = 0.05) -> dict:
    """Read the state; retry briefly if the file is momentarily unreadable or not valid JSON
    (e.g. a file written by an older, non-atomic writer)."""
    for attempt in range(retries + 1):
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            if attempt == retries:
                raise
            time.sleep(delay)
    return {}


def _write(p: Path, st: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(st, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def load(s: Settings) -> dict:
    return _read(_path(s))


def update(s: Settings, **kv) -> dict:
    """Set top-level keys (other keys are preserved). Atomic and safe across threads/processes."""
    p = _path(s)
    with _locked(p):
        st = _read(p)
        st.update(kv)
        _write(p, st)
        return st


def update_arm(s: Settings, name: str, **kv) -> dict:
    """Merge `kv` into state["arms"][name] in one locked read-merge-write (other arms preserved)."""
    p = _path(s)
    with _locked(p):
        st = _read(p)
        arms = dict(st.get("arms") or {})
        arms[name] = {**arms.get(name, {}), **kv}
        st["arms"] = arms
        _write(p, st)
        return arms[name]

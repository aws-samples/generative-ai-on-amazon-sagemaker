"""Append-only JSONL cache keyed by id. Makes every LLM stage resumable after a crash/restart."""
from __future__ import annotations

import json
import threading
from pathlib import Path


def content_hash(text: str) -> str:
    import hashlib
    return hashlib.sha3_256((text or "").encode()).hexdigest()[:16]


class JsonlCache:
    def __init__(self, path: str | Path, key: str = "request_id"):
        self.path = Path(path)
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.data: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue          # tolerate a torn last line after a crash
                    self.data[str(rec[self.key])] = rec

    def __contains__(self, k) -> bool:
        return str(k) in self.data

    def __len__(self) -> int:
        return len(self.data)

    def get(self, k, default=None):
        return self.data.get(str(k), default)

    def fresh(self, k, text: str) -> bool:
        """Cached record exists AND was produced for this exact input text."""
        rec = self.data.get(str(k))
        return rec is not None and rec.get("input_hash") == content_hash(text)

    def put(self, rec: dict) -> None:
        with self._lock:
            self.data[str(rec[self.key])] = rec
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    def put_many(self, recs) -> None:
        for r in recs:
            self.put(r)

    def values(self):
        return list(self.data.values())

    def compact(self, keep=None) -> int:
        """Rewrite the file with one line per key (latest record), sorted by key, optionally only the
        keys in `keep`. Keeps version-controlled caches small and their diffs stable. Returns rows."""
        with self._lock:
            if keep is not None:
                keep = {str(k) for k in keep}
                self.data = {k: v for k, v in self.data.items() if k in keep}
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with open(tmp, "w") as f:
                for k in sorted(self.data):
                    f.write(json.dumps(self.data[k], default=str, sort_keys=True) + "\n")
            tmp.replace(self.path)
            return len(self.data)

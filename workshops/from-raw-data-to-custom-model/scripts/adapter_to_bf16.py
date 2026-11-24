"""Convert a LoRA adapter's safetensors from float32 to bfloat16 (numpy only; no torch needed).

The serverless RLVR recipe (verl GRPO) saves LoRA adapters in float32; SFT saves bf16.
bf16 = the top 16 bits of the float32, rounded to nearest-even (NaN kept as NaN).

    python scripts/adapter_to_bf16.py adapters/v2-sft-rlvr/adapter_model.safetensors [--check]
Writes the file in place (the float32 original stays in S3).
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np


def read(path):
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    meta = header.pop("__metadata__", None)
    return header, meta, memoryview(raw)[8 + n:]


def f32_to_bf16(a: np.ndarray) -> np.ndarray:
    u = a.astype("<f4").view("<u4")
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16          # round to nearest, ties to even
    nan = np.isnan(a)
    rounded = np.where(nan, (u >> 16) | 0x0040, rounded)    # keep NaNs quiet NaNs
    return rounded.astype("<u2")


def bf16_to_f32(b: np.ndarray) -> np.ndarray:
    return (b.astype("<u4") << 16).view("<f4")


def convert(path):
    header, meta, data = read(path)
    out_header, chunks, off, worst = {}, [], 0, 0.0
    for name in sorted(header, key=lambda k: header[k]["data_offsets"][0]):
        t = header[name]
        s, e = t["data_offsets"]
        if t["dtype"] == "F32":
            a = np.frombuffer(data[s:e], dtype="<f4")
            b = f32_to_bf16(a)
            back = bf16_to_f32(b)
            scale = float(np.abs(a).max()) or 1.0
            worst = max(worst, float(np.abs(back - a).max()) / scale)
            buf, dtype = b.tobytes(), "BF16"
        else:
            buf, dtype = bytes(data[s:e]), t["dtype"]
        out_header[name] = {"dtype": dtype, "shape": t["shape"], "data_offsets": [off, off + len(buf)]}
        chunks.append(buf)
        off += len(buf)
    if meta:
        out_header["__metadata__"] = meta
    hb = json.dumps(out_header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)                     # 8-byte alignment, per the format
    Path(path).write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(chunks))
    return len(header), worst


if __name__ == "__main__":
    p = sys.argv[1]
    before = Path(p).stat().st_size
    n, worst = convert(p)
    print(f"{p}: {n} tensors, {before / 2**20:.1f} -> {Path(p).stat().st_size / 2**20:.1f} MiB, "
          f"max abs error / tensor max = {worst:.2e}")

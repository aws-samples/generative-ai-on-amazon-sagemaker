"""Builds the OpenVLA training set from VyoJ/BridgeData-V2-Scripted-Images (used by 1-prepare-data.ipynb).

The source has 3 frames per episode and no action labels, so -- as in the AWS blog this lab is based on --
x/y end-effector deltas are pseudo-labelled from dense optical flow between consecutive frames; z, rotation
and gripper stay 0. Each (frame_t, frame_t+1) pair becomes one sample: image=frame_t, action=delta_t.
Replace `pseudo_actions` with real robot actions when you bring your own data.

Actions are normalized the way OpenVLA normalizes its training mixture ("bounds_q99"): per axis,
a_norm = 2 * (a - q01) / (q99 - q01) - 1, clipped to [-1, 1], with q01/q99 taken from the train split.
Axes with q01 == q99 (here z/rotation/gripper) stay 0. The statistics are returned so predictions can be
mapped back to pixels (see `unnormalize`).

Output: DatasetDict(train, validation) with columns image (PIL), instruction (str), action (7 x float32),
split by episode so no episode appears in both splits.
"""
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value, load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

SOURCE_DATASET = "VyoJ/BridgeData-V2-Scripted-Images"
# The repo stores 26k loose JPEGs, which trips the Hub's per-file rate limit (HTTP 429) when loaded directly.
# The Hub's auto-converted Parquet branch holds the same rows in 3 files, so we download only those we need.
PARQUET_REVISION = "refs/convert/parquet"
PARQUET_SHARDS = [f"default/train/{i:04d}.parquet" for i in range(3)]
FRAME_COLUMNS = ["first_image", "intermediate_image", "frame_43_image"]
INSTRUCTION = "move object to target"
ACTION_DIM = 7

FEATURES = Features({
    "image": Image(),
    "instruction": Value("string"),
    "action": Sequence(Value("float32"), length=ACTION_DIM),
    "episode": Value("int32"),
})


def pseudo_actions(frames):
    """Mean Farneback flow (pixels) between consecutive frames -> raw [dx, dy, 0, 0, 0, 0, 0]."""
    actions = []
    for a, b in zip(frames[:-1], frames[1:]):
        g1 = cv2.cvtColor(np.array(a.convert("RGB")), cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(np.array(b.convert("RGB")), cv2.COLOR_RGB2GRAY)
        if g1.shape != g2.shape:
            g2 = cv2.resize(g2, g1.shape[::-1])
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[:2] = np.nan_to_num(flow.mean(axis=(0, 1)))
        actions.append(action)
    return actions


def episode_samples(episode_idx, row):
    frames = [row[c] for c in FRAME_COLUMNS if row.get(c) is not None]
    return [{"image": img.convert("RGB"), "instruction": INSTRUCTION, "action": act.tolist(), "episode": episode_idx}
            for img, act in zip(frames[:-1], pseudo_actions(frames))]


def _download(shard, attempts=5):
    for attempt in range(attempts):
        try:
            return hf_hub_download(SOURCE_DATASET, shard, repo_type="dataset", revision=PARQUET_REVISION)
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status != 429 or attempt == attempts - 1:
                if status == 429:
                    print("Hugging Face rate limit (429). Anonymous limits are per IP and shared in Studio: "
                          "run `from huggingface_hub import login; login()` with a read token, then retry.")
                raise
            wait = 60 * (attempt + 1)
            print(f"HTTP 429 from Hugging Face, retrying {shard} in {wait}s ({attempt + 1}/{attempts - 1})")
            time.sleep(wait)


def download_episodes(max_episodes, parquet_dir=None):
    """Loads Parquet shards in order until they hold max_episodes rows; returns the first max_episodes.

    parquet_dir: folder with already-downloaded shards (0000.parquet, ...) to skip the Hugging Face download.
    """
    paths, n_rows = [], 0
    for shard in PARQUET_SHARDS:
        if n_rows >= max_episodes:
            break
        local = Path(parquet_dir) / Path(shard).name if parquet_dir else None
        if parquet_dir and not local.exists():
            break
        paths.append(str(local) if local else _download(shard))
        n_rows += pq.ParquetFile(paths[-1]).metadata.num_rows
    source = load_dataset("parquet", data_files=paths, split="train")
    return source.select(range(min(max_episodes, len(source))))


def build_dataset(max_episodes=600, val_fraction=0.1, seed=42, progress=None, parquet_dir=None):
    rows = download_episodes(max_episodes, parquet_dir)
    progress = progress or (lambda it, **_: it)

    samples = []
    for idx, row in enumerate(progress(rows, total=len(rows), desc="pseudo-labelling")):
        samples.extend(episode_samples(idx, row))

    episodes = sorted({s["episode"] for s in samples})
    random.Random(seed).shuffle(episodes)
    val_eps = set(episodes[: max(1, int(len(episodes) * val_fraction))])
    train_actions = np.array([s["action"] for s in samples if s["episode"] not in val_eps])
    stats = {"q01": np.quantile(train_actions, 0.01, axis=0).tolist(),
             "q99": np.quantile(train_actions, 0.99, axis=0).tolist(),
             "mean": train_actions.mean(0).tolist(), "std": train_actions.std(0).tolist()}
    for s in samples:
        s["action"] = normalize(s["action"], stats).tolist()

    split = lambda keep: Dataset.from_list([s for s in samples if (s["episode"] in val_eps) == keep],  # noqa: E731
                                           features=FEATURES)
    return DatasetDict({"train": split(False), "validation": split(True)}), stats


def normalize(action, stats):
    low, high = np.array(stats["q01"]), np.array(stats["q99"])
    span = np.where(high > low, high - low, 1.0)
    return np.where(high > low, np.clip(2 * (np.asarray(action) - low) / span - 1, -1, 1), 0.0)


def unnormalize(action, stats):
    low, high = np.array(stats["q01"]), np.array(stats["q99"])
    return np.where(high > low, (np.asarray(action) + 1) / 2 * (high - low) + low, np.asarray(action))

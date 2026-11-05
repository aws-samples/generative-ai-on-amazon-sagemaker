"""Builds the OpenVLA training set from LIBERO demonstrations (used by 1-prepare-data.ipynb when DATASET = "libero").

LIBERO is a simulated tabletop benchmark (40 tasks, ~1,700 demonstrations, real 7-DoF end-effector actions) that
OpenVLA was *not* pretrained on; the OpenVLA paper uses it to show what fine-tuning adds. Source:
physical-intelligence/libero (LeRobot v2 format, CC BY 4.0): one Parquet file per episode with the agent-view
image (PNG), the wrist image, the robot state and the action of every frame, plus the task instruction.

Each kept frame becomes one sample (image, instruction, action). Frames are subsampled (`frame_stride`) because
consecutive frames at 10 fps are near-duplicates, and images are re-encoded as 224x224 JPEG (OpenVLA's input
size) so the shipped sample stays small. Validation holds out whole episodes, one per task, so every task is
evaluated on a demonstration the model has not seen.

Actions are normalized like OpenVLA's training mixture ("bounds_q99", same as bridge_data.py): per axis,
2 * (a - q01) / (q99 - q01) - 1, clipped to [-1, 1], with q01/q99 from the train split.
"""
import io
import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from huggingface_hub import hf_hub_download
from PIL import Image as PILImage

from bridge_data import normalize, unnormalize  # noqa: F401  (same action normalization; re-exported)

SOURCE_DATASET = "physical-intelligence/libero"
ACTION_DIM = 7
IMAGE_SIZE = 224

FEATURES = Features({
    "image": Image(),
    "instruction": Value("string"),
    "action": Sequence(Value("float32"), length=ACTION_DIM),
    "episode": Value("int32"),
})
SAMPLE_SCHEMA = pa.schema([("image", pa.binary()), ("instruction", pa.string()),
                           ("action", pa.list_(pa.float32(), ACTION_DIM)), ("episode", pa.int32())])


def pick_episodes(episodes_per_task, seed=42):
    """Episode indices, `episodes_per_task` per task (random but reproducible), from meta/episodes.jsonl."""
    path = hf_hub_download(SOURCE_DATASET, "meta/episodes.jsonl", repo_type="dataset")
    by_task = {}
    for line in open(path):
        e = json.loads(line)
        by_task.setdefault(e["tasks"][0], []).append(e["episode_index"])
    rng = random.Random(seed)
    return {task: sorted(rng.sample(eps, min(episodes_per_task, len(eps)))) for task, eps in sorted(by_task.items())}


def _jpeg(png_bytes, size=IMAGE_SIZE, quality=85):
    im = PILImage.open(io.BytesIO(png_bytes)).convert("RGB").resize((size, size), PILImage.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def download_rows(episodes_per_task=3, frame_stride=5, progress=None):
    """Downloads the chosen episodes from the Hub and returns compact sample rows."""
    progress = progress or (lambda it, **_: it)
    chosen = pick_episodes(episodes_per_task)
    rows = []
    for task, eps in progress(list(chosen.items()), desc="tasks"):
        for ep in eps:
            f = hf_hub_download(SOURCE_DATASET, f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet", repo_type="dataset")
            t = pq.read_table(f, columns=["image", "actions"])
            images, actions = t.column("image").to_pylist(), t.column("actions").to_pylist()
            for i in range(0, len(images), frame_stride):
                rows.append({"image": _jpeg(images[i]["bytes"]), "instruction": task,
                             "action": [float(x) for x in actions[i]], "episode": ep})
    return rows


def make_sample(out_path, episodes_per_task=3, frame_stride=5, progress=None):
    """Writes the compact Parquet sample shipped in sample_data/ (run once by the maintainers)."""
    rows = download_rows(episodes_per_task, frame_stride, progress)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SAMPLE_SCHEMA), out_path, compression="zstd")
    return len(rows)


def build_dataset(max_episodes=None, val_fraction=None, seed=42, progress=None, parquet_dir=None,
                  episodes_per_task=3, frame_stride=5):
    """DatasetDict(train, validation) + action stats, like bridge_data.build_dataset.

    parquet_dir: folder with a sample written by make_sample (e.g. sample_data/libero-120); None downloads
    `episodes_per_task` episodes per task from the Hub. Validation = one held-out episode per task.
    max_episodes / val_fraction are accepted for interface compatibility with bridge_data and ignored.
    """
    if parquet_dir:
        rows = pq.read_table(sorted(Path(parquet_dir).glob("*.parquet"))).to_pylist()
    else:
        rows = download_rows(episodes_per_task, frame_stride, progress)

    episodes_by_task = {}
    for r in rows:
        episodes_by_task.setdefault(r["instruction"], set()).add(r["episode"])
    rng = random.Random(seed)
    val_eps = {rng.choice(sorted(eps)) for eps in episodes_by_task.values() if len(eps) > 1}

    train_actions = np.array([r["action"] for r in rows if r["episode"] not in val_eps])
    stats = {"q01": np.quantile(train_actions, 0.01, axis=0).tolist(),
             "q99": np.quantile(train_actions, 0.99, axis=0).tolist(),
             "mean": train_actions.mean(0).tolist(), "std": train_actions.std(0).tolist()}
    samples = [{"image": {"bytes": r["image"]}, "instruction": r["instruction"],
                "action": normalize(r["action"], stats).tolist(), "episode": r["episode"]} for r in rows]

    split = lambda keep: Dataset.from_list([s for s in samples if (s["episode"] in val_eps) == keep],  # noqa: E731
                                           features=FEATURES)
    return DatasetDict({"train": split(False), "validation": split(True)}), stats

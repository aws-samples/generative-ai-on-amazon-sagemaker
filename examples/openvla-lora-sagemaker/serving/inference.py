"""SageMaker PyTorch inference handler for a merged OpenVLA checkpoint.

Deployed by 4-deployment.ipynb next to the model as <model prefix>/code/ together with openvla_utils.py. The model prefix
holds either a merged checkpoint or just a LoRA adapter (the base model then comes from the Hugging Face Hub).

Request  (application/json): {"image": "<base64 JPEG/PNG>", "instruction": "move object to target"}
Response (application/json): {"action": [7 floats in [-1, 1]], "action_unnormalized": [...]}
`action_unnormalized` is present when action_stats.json (written by notebook 1) ships with the model.
"""
import base64
import io
import json
import os

import numpy as np
import torch
from PIL import Image

from openvla_utils import ActionTokenizer, load_openvla, predict_action

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def model_fn(model_dir, context=None):
    # merged checkpoint, or LoRA adapter + base model from the Hub (merged on the GPU at startup)
    model, processor = load_openvla(model_dir, DEVICE)
    stats_path = os.path.join(model_dir, "action_stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else None
    ctx = {"model": model, "processor": processor, "action_tokenizer": ActionTokenizer(processor.tokenizer),
           "stats": stats}
    # Warm-up: the first generate() pays one-time CUDA/kernel initialization (~80 s), longer than SageMaker's 60 s
    # invocation limit. Doing it here, before the endpoint reports healthy, keeps the first real request fast.
    predict_action(model, processor, ctx["action_tokenizer"], Image.new("RGB", (224, 224)), "warm up")
    return ctx


def input_fn(body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"unsupported content type {content_type}")
    req = json.loads(body)
    return Image.open(io.BytesIO(base64.b64decode(req["image"]))).convert("RGB"), req["instruction"]


def predict_fn(data, ctx):
    image, instruction = data
    action = predict_action(ctx["model"], ctx["processor"], ctx["action_tokenizer"], image, instruction)
    out = {"action": action.tolist()}
    if ctx["stats"]:
        low, high = np.array(ctx["stats"]["q01"]), np.array(ctx["stats"]["q99"])
        out["action_unnormalized"] = np.where(high > low, (action + 1) / 2 * (high - low) + low, action).tolist()
    return out


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"

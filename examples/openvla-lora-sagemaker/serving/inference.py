"""SageMaker PyTorch inference handler for a merged OpenVLA checkpoint.

Deployed by 4-deployment.ipynb next to the weights as <model prefix>/code/ together with openvla_utils.py.

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
from transformers import AutoModelForVision2Seq, AutoProcessor

from openvla_utils import ActionTokenizer, predict_action

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def model_fn(model_dir, context=None):
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager",
                                                   low_cpu_mem_usage=True, trust_remote_code=True).to(DEVICE).eval()
    stats_path = os.path.join(model_dir, "action_stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else None
    return {"model": model, "processor": processor, "action_tokenizer": ActionTokenizer(processor.tokenizer),
            "stats": stats}


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

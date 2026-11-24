"""Fast LLM-as-a-Judge: the four eval metrics (metrics.py) via parallel Bedrock Converse calls.

Unlike the managed evaluator, the judge gets the REAL ground truth (constraints incl. hidden
needs, candidate prices and attributes) and runs with configurable concurrency.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import boto3

from . import config, metrics

METRICS = [m["customMetricDefinition"] for m in metrics.eval_metrics()]


def _rate(brt, model_id: str, metric: dict, prompt: str, response: str, ground_truth: str):
    instr = (metric["instructions"].replace("{{prompt}}", prompt)
             .replace("{{prediction}}", response).replace("{{ground_truth}}", ground_truth))
    opts = "\n".join(f'- "{o["definition"]}"' for o in metric["ratingScale"])
    text = (f"{instr}\n\nRate the response using exactly one of these options:\n{opts}\n\n"
            'Answer with JSON only: {"explanation": "<brief>", "rating": "<one option, verbatim>"}')
    for attempt in range(5):
        try:
            out = brt.converse(modelId=model_id, messages=[{"role": "user", "content": [{"text": text}]}],
                               inferenceConfig={"maxTokens": 800, "temperature": 0})
            t = next(c["text"] for c in out["output"]["message"]["content"] if "text" in c)
            m = re.search(r"\{.*\}", t, re.S)
            d = json.loads(m.group(0)) if m else {}
            rating = str(d.get("rating", "")).strip().strip('"').lower()
            for o in metric["ratingScale"]:
                if o["definition"].lower() == rating:
                    return o["value"]["floatValue"], d.get("explanation", "")
            return None, t[:300]
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                return None, f"error: {e}"[:200]
            time.sleep(2 ** attempt)


def judge_many(items: list[dict], region: str, workers: int = 16, model_id: str | None = None) -> list[dict]:
    """items: [{prompt, response, ground_truth}] -> [{metric: score, metric_why: text}] (same order)."""
    brt = boto3.client("bedrock-runtime", region_name=region)
    model_id = model_id or config.judge_profile_id(region)
    work = [(i, m) for i in range(len(items)) for m in METRICS]

    def run(w):
        i, m = w
        it = items[i]
        return i, m["name"], _rate(brt, model_id, m, it["prompt"], it["response"], it["ground_truth"])

    out = [dict() for _ in items]
    with ThreadPoolExecutor(workers) as ex:
        for i, name, (score, why) in ex.map(run, work):
            out[i][name], out[i][f"{name}_why"] = score, why
    return out

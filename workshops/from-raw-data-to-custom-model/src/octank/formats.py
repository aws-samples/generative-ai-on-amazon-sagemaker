"""Dataset writers: SFT, RLVR (hybrid reward ground truth) and the evaluation set.

Schemas follow the aws-ai-ml skill's strategy_data_requirements.md (open-weight models):
  SFT   {"prompt", "completion"}
  RLVR  {"data_source", "prompt": [{"role","content"}], "ability", "reward_model": {"ground_truth","style"}}
  Eval  {"query", "response"}   (response = reference answer + the authoritative constraints)
"""
from __future__ import annotations

import json
from pathlib import Path

from .util import as_list

DATA_SOURCE = "octank-gift-concierge"


def write_jsonl(rows, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def constraints_json(req: dict, pool: list[dict]) -> str:
    """The verifiable constraints + candidate facts (prices, categories, attribute tags), including
    needs implied by the shopper's situation. Used by the reward and the judges."""
    hidden = as_list(req.get("hidden_avoid"))
    extra = {"implied_by_situation": {
        "avoid": hidden, "situation": req.get("situation", ""),
        "note": "The shopper implied these through their situation (not in the form). A pick "
                "with one of these attributes violates the constraints."}} if hidden else {}
    return json.dumps({**extra, **{
        "budget_total": float(req["budget_total"]), "num_items": int(req.get("num_items") or 1),
        "avoid": sorted(set(as_list(req.get("avoid"))) | set(as_list(req.get("hidden_avoid")))),
        "candidates": {c["id"]: {"price": float(c["price"]), "category": c["category"],
                                 "attributes": as_list(c["attributes"])} for c in pool},
    }})


def rlvr_ground_truth(req: dict, pool: list[dict], difficulty_score: float, reference: dict | None) -> dict:
    """Ground truth for the hybrid reward (judge/reward_lambda.py): constraints_json +
    interests, difficulty, the teacher's reference answer and per-candidate interests (for the
    tag-based fit score). Identical for instructor and attendee runs."""
    gt = json.loads(constraints_json(req, pool))
    gt.update(interests=list(req.get("interests") or []), difficulty=difficulty_score, reference=reference)
    for c in pool:
        gt["candidates"][c["id"]]["interests"] = list(c.get("interests") or [])
    return gt


def sft_row(t: dict) -> dict:
    return {"prompt": t["prompt"], "completion": t["completion"]}


def rlvr_row(prompt: str, req: dict, pool: list[dict], difficulty_score: float, reference: dict | None) -> dict:
    return {"data_source": DATA_SOURCE, "prompt": [{"role": "user", "content": prompt}],
            "ability": "gift-constraints",
            "reward_model": {"ground_truth": json.dumps(rlvr_ground_truth(req, pool, difficulty_score, reference)),
                             "style": "rule"}}


def eval_row(t: dict, req: dict, pool: list[dict]) -> dict:
    """Evaluation row: the judge's ground truth carries the reference answer AND the authoritative
    constraints (candidates are untagged in the prompt, and implied needs aren't in the form)."""
    return {"query": t["prompt"], "response": json.dumps(
        {"reference_answer": json.loads(t["answer"]), "constraints": json.loads(constraints_json(req, pool))})}

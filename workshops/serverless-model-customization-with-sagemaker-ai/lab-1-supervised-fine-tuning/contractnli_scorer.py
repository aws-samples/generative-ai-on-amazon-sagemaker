"""
Custom scorer for the managed evaluation pipeline.

This is the deterministic alternative to LLM-as-a-Judge: the same comparison the
notebook does by hand, expressed per record so the pipeline can run it.

It must be a single self-contained file. The pipeline uploads it and executes it in
its own container, where neither the notebook nor `contractnli.py` exists — hence the
small amount of duplication.

Emits the four metrics the notebook uses:

    label_correct           fraction of the 17 items whose verdict is right
    contradiction_correct   accuracy on the gold `Contradiction` items (the rare class)
    evidence_f1             span-set overlap, for items where the gold cites something
    json_valid              did the model return parseable JSON at all

plus `aggregate_reward_score` = 0.6 x label_correct + 0.4 x evidence_f1, a single
number the pipeline can rank models by. Evidence is weighted heavily because a
verdict without the right clause is not usable by a reviewer.

NOTE ON AVERAGING: the pipeline calls this once per contract and averages the
results, so `evidence_f1` here is a mean of per-contract F1. The notebook's local
pass pools every span before dividing (micro). The two agree closely on this dataset
but they are different statistics.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

LABELS = ("Entailment", "Contradiction", "NotMentioned")


def _batch(event: Any) -> List[Dict[str, Any]]:
    if isinstance(event, list):
        return event
    if isinstance(event, dict) and "body" in event:
        body = event["body"]
        parsed = json.loads(body) if isinstance(body, str) else body
        if isinstance(parsed, list):
            return parsed
    return [event] if isinstance(event, dict) else []


def _model_response(record: Dict[str, Any]) -> str:
    """The model's answer.

    The record carries `model_response` (the model's output), `reference_answer` (the
    gold) and `response` — which is ALSO the gold, passed through from the dataset's
    `response` column. Reading `response` would compare the gold with itself and
    return a perfect score on every record, so prefer `model_response` and fall back
    to `response` only for the RLVR training payload, which has no `model_response`.
    """
    for key in ("model_response", "generated_text", "completion", "response"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _reference(record: Dict[str, Any]) -> str:
    for key in ("reference_answer", "ground_truth", "response"):
        value = record.get(key)
        if isinstance(value, str) and value.strip().startswith("{"):
            return value
    container = record.get("reward_model")
    if isinstance(container, dict):
        value = container.get("ground_truth")
        if isinstance(value, str):
            return value
    return ""


def parse_answer(text: str):
    """Extract the JSON verdict object. Returns (obj_or_None, ok)."""
    body = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.DOTALL)
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.DOTALL)
    if fence:
        body = fence.group(1)
    start = body.find("{")
    if start == -1:
        return None, False
    if body[start:start + 2] == "{{":          # models echo the template's braces
        start += 1
    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(body[start:], start):
        if in_string:
            escaped = (ch == "\\") and not escaped
            if ch == '"' and not escaped:
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(body[start:i + 1])
                    return (obj, True) if isinstance(obj, dict) else (None, False)
                except json.JSONDecodeError:
                    return None, False
    return None, False


def _spans(value) -> set:
    if not isinstance(value, list):
        return set()
    return {int(x) for x in value if str(x).strip().lstrip("-").isdigit()}


def _label(value):
    if not isinstance(value, str):
        return value
    squashed = value.strip().lower().replace(" ", "")
    return next((L for L in LABELS if squashed == L.lower()), value.strip())


def score_record(record: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """Score one contract: 17 verdicts plus their cited spans."""
    predicted, ok = parse_answer(_model_response(record))
    gold, _ = parse_answer(_reference(record))
    gold = gold or {}

    items = correct = 0
    contradictions = contradictions_correct = 0
    ev_tp = ev_fp = ev_fn = 0

    for key, gold_item in gold.items():
        if not isinstance(gold_item, dict):
            continue
        items += 1
        gold_label = gold_item.get("label")
        gold_spans = _spans(gold_item.get("evidence"))

        answer = (predicted or {}).get(key) or {}
        pred_label = _label(answer.get("label") if isinstance(answer, dict) else answer)
        pred_spans = _spans(answer.get("evidence") if isinstance(answer, dict) else None)

        if pred_label == gold_label:
            correct += 1
        if gold_label == "Contradiction":
            contradictions += 1
            contradictions_correct += int(pred_label == gold_label)
        if gold_spans:
            ev_tp += len(gold_spans & pred_spans)
            ev_fp += len(pred_spans - gold_spans)
            ev_fn += len(gold_spans - pred_spans)

    label_correct = correct / items if items else 0.0
    precision = ev_tp / (ev_tp + ev_fp) if ev_tp + ev_fp else 0.0
    recall = ev_tp / (ev_tp + ev_fn) if ev_tp + ev_fn else 0.0
    evidence_f1 = (2 * precision * recall / (precision + recall)
                   if precision + recall else 0.0)

    def metric(name, value, kind="Metric"):
        return {"name": name, "value": round(float(value), 4), "type": kind}

    return {
        "id": str(record.get("id") or index),
        "aggregate_reward_score": round(0.6 * label_correct + 0.4 * evidence_f1, 4),
        "metrics_list": [
            metric("label_correct", label_correct, "Reward"),
            metric("evidence_f1", evidence_f1, "Reward"),
            metric("contradiction_correct",
                   contradictions_correct / contradictions if contradictions else 0.0),
            metric("json_valid", 1.0 if ok else 0.0),
        ],
    }


def lambda_handler(event: Any, context: Any) -> Dict[str, Any]:
    try:
        results = [score_record(r, i) for i, r in enumerate(_batch(event))]
        return {"statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(results)}
    except Exception as error:  # pragma: no cover
        return {"statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": str(error)})}

"""
Custom scorer for the managed serverless evaluation pipeline.

This is the deterministic alternative to LLM-as-judge. It is registered as an AI
Registry evaluator and passed to `CustomScorerEvaluator`, so the same programmatic
metrics used in notebook 3 are produced by the managed pipeline and land in MLflow.

Output contract (validated by the SDK as `RewardFunctionOutput`):

    {"id": str,
     "aggregate_reward_score": float,
     "metrics_list": [{"name": str, "value": float, "type": "Reward"|"Metric"}, ...]}

`type` controls how a number is treated: "Reward" entries are the optimisation
signal, "Metric" entries are reported for observability only. Both appear in the
evaluation output, so you can emit as many named metrics as you care about.

We emit:

  Reward  label_correct          1 if the verdict matches gold
  Reward  evidence_f1            per-item overlap of cited spans with gold spans
  Metric  evidence_precision     did it cite spans that were not justification?
  Metric  evidence_recall        did it miss justifying spans?
  Metric  exact_span_set         1 if the cited span set matches gold exactly
  Metric  cited_nothing          1 if gold had evidence and none was cited
  Metric  is_contradiction       gold is Contradiction (the rare, valuable class)
  Metric  contradiction_correct  correct on that class specifically
  Metric  json_valid             did the model return parseable JSON at all

This file deliberately re-implements the parsing and scoring that `contractnli.py`
already contains (`parse_answer` vs `robust_parse`, `score_record` vs
`score_doc` + `aggregate`). That duplication is required, not an oversight: the
managed pipeline uploads *this single file* and runs it in its own container, where
`contractnli` does not exist. Do not refactor it into a shared import.

IMPORTANT LIMITATION. The pipeline calls this scorer once per record and averages
the results, so every metric here is a per-record value. `evidence_f1` is
therefore a **macro** average of per-item F1, not the corpus-level micro-F1 that
notebook 3 reports. The two differ (micro weights items with more spans more
heavily). Expect the numbers to be close but not identical, and do not mix them
in the same table.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

LABELS = ("Entailment", "Contradiction", "NotMentioned")
_DEBUG_MAX_RECORDS = 3


# ------------------------------------------------------------------- plumbing

def _as_batch(event: Any) -> List[Dict[str, Any]]:
    if isinstance(event, list):
        return event
    if isinstance(event, dict) and "body" in event:
        body = event["body"]
        parsed = json.loads(body) if isinstance(body, str) else body
        if isinstance(parsed, list):
            return parsed
    if isinstance(event, dict):
        return [event]
    raise ValueError("expected a list of records or a dict payload")


def _last_assistant_message(messages: Iterable[Dict[str, Any]]) -> str:
    out = ""
    for message in messages or []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            out = str(message.get("content", ""))
    return out


def _model_response(record: Dict[str, Any]) -> str:
    """The model's answer.

    CAREFUL: the managed pipeline sends records with these keys:

        __few_shots, __index, id, model_response, processor_config,
        query, reference_answer, response

    `model_response` is the model's output. **`response` is the dataset's
    reference column passed straight through** — it holds the GOLD answer, as does
    `reference_answer`. Reading `response` here would compare the gold answer with
    itself and return a perfect 1.0 on every record with no error raised, so
    `response` is deliberately not consulted.
    """
    for key in ("model_response", "modelResponse", "generated_text",
                "completion", "prediction"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("modelResponses", "model_responses"):
        value = record.get(key)
        if isinstance(value, list) and value:
            last = value[-1]
            if isinstance(last, str):
                return last
            if isinstance(last, dict):
                return str(last.get("content") or last.get("text") or "")
    # chat-shaped payloads (used by the RLVR reward path)
    messages = record.get("messages")
    if isinstance(messages, list):
        return _last_assistant_message(messages)
    return ""


def _reference(record: Dict[str, Any]) -> str:
    """The gold JSON string for this contract."""
    for key in ("reference_answer", "referenceAnswer", "ground_truth",
                "response", "answer", "target"):
        value = record.get(key)
        if isinstance(value, str) and value.strip().startswith("{"):
            return value
    for container_key in ("extra_info", "reward_model"):
        container = record.get(container_key) or {}
        if isinstance(container, dict):
            for key in ("reference_answer", "ground_truth", "response"):
                value = container.get(key)
                if isinstance(value, str) and value.strip().startswith("{"):
                    return value
    return ""


def parse_answer(text: str):
    """Extract the JSON verdict object. Returns (obj_or_None, ok)."""
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    if start == -1:
        return None, False
    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(cleaned[start:], start):
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
                    obj = json.loads(cleaned[start:i + 1])
                    return (obj, True) if isinstance(obj, dict) else (None, False)
                except json.JSONDecodeError:
                    return None, False
    return None, False


def _spans(value) -> set:
    if not isinstance(value, list):
        return set()
    return {int(x) for x in value if str(x).strip().lstrip("-").isdigit()}


def _normalise_label(value):
    if not isinstance(value, str):
        return value
    squashed = value.strip().lower().replace(" ", "")
    return next((L for L in LABELS if squashed == L.lower()), value.strip())


# --------------------------------------------------------------- the scoring

def score_record(record: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    record_id = str(record.get("id") or record.get("record_id") or index)
    predicted, ok = parse_answer(_model_response(record))
    gold, _ = parse_answer(_reference(record))
    gold = gold or {}

    n = correct = 0
    ev_tp = ev_fp = ev_fn = 0
    exact_sets = items_with_gold_ev = cited_nothing = 0
    contradictions = contradictions_correct = 0
    per_item_f1_total = 0.0

    for key, gold_item in gold.items():
        if not isinstance(gold_item, dict):
            continue
        n += 1
        gold_label = gold_item.get("label")
        gold_ev = _spans(gold_item.get("evidence"))

        answer = (predicted or {}).get(key) or {}
        pred_label = _normalise_label(answer.get("label")
                                     if isinstance(answer, dict) else answer)
        pred_ev = _spans(answer.get("evidence") if isinstance(answer, dict) else None)

        if pred_label == gold_label:
            correct += 1
        if gold_label == "Contradiction":
            contradictions += 1
            contradictions_correct += int(pred_label == gold_label)

        if gold_ev:
            items_with_gold_ev += 1
            tp = len(gold_ev & pred_ev)
            fp, fn = len(pred_ev - gold_ev), len(gold_ev - pred_ev)
            ev_tp, ev_fp, ev_fn = ev_tp + tp, ev_fp + fp, ev_fn + fn
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            per_item_f1_total += (2 * precision * recall / (precision + recall)
                                 if precision + recall else 0.0)
            exact_sets += int(pred_ev == gold_ev)
            cited_nothing += int(not pred_ev)

    label_accuracy = correct / n if n else 0.0
    evidence_f1 = per_item_f1_total / items_with_gold_ev if items_with_gold_ev else 0.0
    ev_precision = ev_tp / (ev_tp + ev_fp) if ev_tp + ev_fp else 0.0
    ev_recall = ev_tp / (ev_tp + ev_fn) if ev_tp + ev_fn else 0.0

    # Headline score: correctness plus grounding. Evidence is weighted heavily
    # because a verdict without the right clause is not usable by a reviewer, and
    # because label accuracy alone can be inflated by learning the label priors.
    aggregate = round(0.6 * label_accuracy + 0.4 * evidence_f1, 4)

    def metric(name, value, kind="Metric"):
        return {"name": name, "value": round(float(value), 4), "type": kind}

    return {
        "id": record_id,
        "aggregate_reward_score": aggregate,
        "metrics_list": [
            metric("label_correct", label_accuracy, "Reward"),
            metric("evidence_f1", evidence_f1, "Reward"),
            metric("evidence_precision", ev_precision),
            metric("evidence_recall", ev_recall),
            metric("exact_span_set",
                   exact_sets / items_with_gold_ev if items_with_gold_ev else 0.0),
            metric("cited_nothing",
                   cited_nothing / items_with_gold_ev if items_with_gold_ev else 0.0),
            metric("is_contradiction", contradictions / n if n else 0.0),
            metric("contradiction_correct",
                   contradictions_correct / contradictions if contradictions else 0.0),
            metric("json_valid", 1.0 if ok else 0.0),
        ],
    }


def _debug(record, result, index):
    if index >= _DEBUG_MAX_RECORDS:
        return
    print("SCORER_DEBUG", json.dumps({
        "index": index,
        "record_keys": sorted(record.keys()),
        "response_head": _model_response(record)[:160],
        "reference_head": _reference(record)[:160],
        "result": result,
    }, default=str)[:1500])


def lambda_handler(event: Any, context: Any) -> Dict[str, Any]:
    try:
        results = []
        for i, record in enumerate(_as_batch(event)):
            result = score_record(record, index=i)
            _debug(record, result, i)
            results.append(result)
        return {"statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(results)}
    except Exception as error:  # pragma: no cover
        return {"statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": str(error)})}

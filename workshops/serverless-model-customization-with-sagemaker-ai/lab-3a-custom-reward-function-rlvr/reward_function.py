"""Custom GSM8K reward function for SageMaker RLVR.

This module follows the SageMaker Studio Reward Function schema:
- ``lambda_handler`` receives a batch of records as a list of dictionaries.
- Each record contains a ``messages`` conversation. The last assistant message is
  the model response to score.
- Each record contains ``reference_answer`` with the expected final answer.
- The handler returns an API Gateway-style response whose body is a JSON list of
  per-record reward results.

The reward favors numerically correct final answers while also exposing useful
format and reasoning metrics for observability.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional


_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_FINAL_ANSWER_PATTERN = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_DEBUG_MAX_RECORDS = 3
_DEBUG_MAX_CHARS = 1200


def _as_batch(event: Any) -> List[Dict[str, Any]]:
    """Normalize Lambda input into a list of records.

    SageMaker Studio sends a list directly. This helper also tolerates an API
    Gateway-style payload with a JSON string ``body`` so the function is easy to
    test manually.
    """
    if isinstance(event, list):
        return event

    if isinstance(event, dict) and "body" in event:
        body = event["body"]
        if isinstance(body, str):
            parsed = json.loads(body)
        else:
            parsed = body
        if isinstance(parsed, list):
            return parsed

    if isinstance(event, dict):
        return [event]

    raise ValueError("Expected event to be a list of records or a dictionary payload")


def _last_assistant_message(messages: Iterable[Dict[str, Any]]) -> str:
    """Return the content of the last assistant message in a conversation."""
    response = ""
    for message in messages or []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            response = str(message.get("content", ""))
    return response


def _stringify_candidate(value: Any) -> str:
    """Convert common generated-response containers into text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "response", "text", "generated_text", "output"):
            if key in value:
                return _stringify_candidate(value[key])
    if isinstance(value, list):
        return "\n".join(_stringify_candidate(item) for item in value if item is not None)
    return str(value)


def _extract_model_response(record: Dict[str, Any]) -> str:
    """Extract the generated model response from likely SageMaker payload shapes."""
    direct_response_keys = (
        "response",
        "model_response",
        "modelResponse",
        "generated_text",
        "generated_response",
        "completion",
        "output",
    )
    for key in direct_response_keys:
        response = _stringify_candidate(record.get(key))
        if response:
            return response

    for key in ("modelResponses", "model_responses"):
        responses = record.get(key)
        if isinstance(responses, list) and responses:
            response = _stringify_candidate(responses[-1])
            if response:
                return response

    messages = record.get("messages") or record.get("prompt") or []
    if isinstance(messages, list):
        response = _last_assistant_message(messages)
        if response:
            return response

    return ""


def _extract_reference_answer(record: Dict[str, Any]) -> Any:
    """Extract the expected final answer from common dataset/runtime fields."""
    for key in ("reference_answer", "referenceAnswer", "ground_truth", "answer"):
        if record.get(key) not in (None, ""):
            return record.get(key)

    reward_model = record.get("reward_model") or record.get("rewardModel") or {}
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") not in (None, ""):
        return reward_model.get("ground_truth")

    extra_info = record.get("extra_info") or record.get("extraInfo") or {}
    if isinstance(extra_info, dict):
        for key in ("reference_answer", "ground_truth", "answer"):
            if extra_info.get(key) not in (None, ""):
                return extra_info.get(key)

    return None


def _to_decimal(value: Any) -> Optional[Decimal]:
    """Convert a numeric-looking value to Decimal after removing commas."""
    if value is None:
        return None
    match = _NUMBER_PATTERN.search(str(value))
    if not match:
        return None
    try:
        return Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        return None


def _extract_final_answer(response: str) -> Optional[Decimal]:
    """Extract the model's final numeric answer.

    Prefer the GSM8K convention ``#### <number>``. If the delimiter is missing,
    fall back to the last numeric value in the response.
    """
    final_match = _FINAL_ANSWER_PATTERN.search(response)
    if final_match:
        return _to_decimal(final_match.group(1))

    matches = list(_NUMBER_PATTERN.finditer(response))
    if not matches:
        return None
    return _to_decimal(matches[-1].group(0))


def _has_final_answer_format(response: str) -> float:
    return 1.0 if _FINAL_ANSWER_PATTERN.search(response) else 0.0


def _has_reasoning(response: str) -> float:
    """Heuristic: reward non-trivial work before the final answer."""
    text_before_final = response.split("####", 1)[0].strip()
    return 1.0 if len(text_before_final.split()) >= 8 else 0.0


def reward_function(record: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """Score one model response against a GSM8K reference answer."""
    record_id = str(record.get("id") or record.get("record_id") or index)
    response = _extract_model_response(record)
    reference_answer = _extract_reference_answer(record)

    predicted = _extract_final_answer(response)
    expected = _to_decimal(reference_answer)

    numeric_correctness = 0.0
    if predicted is not None and expected is not None:
        numeric_correctness = 1.0 if predicted == expected else 0.0

    answer_parseable = 1.0 if predicted is not None else 0.0
    reference_available = 1.0 if expected is not None else 0.0
    final_answer_format = _has_final_answer_format(response)
    reasoning_present = _has_reasoning(response)

    # Numeric correctness remains the main reward. Parseability and format provide
    # a small shaping signal so early rollouts do not collapse to all-zero rewards.
    aggregate_reward_score = round(
        (0.6 * numeric_correctness)
        + (0.2 * answer_parseable)
        + (0.1 * final_answer_format)
        + (0.1 * reasoning_present),
        4,
    )

    return {
        "id": record_id,
        "aggregate_reward_score": aggregate_reward_score,
        "metrics_list": [
            {
                "name": "numeric_correctness",
                "value": numeric_correctness,
                "type": "Reward",
            },
            {
                "name": "answer_parseable",
                "value": answer_parseable,
                "type": "Reward",
            },
            {
                "name": "reference_available",
                "value": reference_available,
                "type": "Metric",
            },
            {
                "name": "final_answer_format",
                "value": final_answer_format,
                "type": "Metric",
            },
            {
                "name": "reasoning_present",
                "value": reasoning_present,
                "type": "Metric",
            },
        ],
    }


def _debug_record(record: Dict[str, Any], result: Dict[str, Any], index: int) -> None:
    """Emit compact CloudWatch debug logs for the first few records only."""
    if index >= _DEBUG_MAX_RECORDS:
        return

    debug_payload = {
        "index": index,
        "record_keys": sorted(record.keys()),
        "has_messages": isinstance(record.get("messages"), list),
        "has_prompt": isinstance(record.get("prompt"), list),
        "reference_answer": str(_extract_reference_answer(record))[:120],
        "response_preview": _extract_model_response(record)[:240],
        "result": result,
    }
    print("REWARD_DEBUG", json.dumps(debug_payload, default=str)[:_DEBUG_MAX_CHARS])


def lambda_handler(event: Any, context: Any) -> Dict[str, Any]:
    """AWS Lambda handler for the SageMaker reward function evaluator."""
    try:
        records = _as_batch(event)
        results = []
        for i, record in enumerate(records):
            result = reward_function(record, index=i)
            _debug_record(record, result, i)
            results.append(result)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(results),
        }
    except Exception as error:  # pragma: no cover - defensive Lambda boundary
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(error)}),
        }

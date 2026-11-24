"""The four LLM-as-a-Judge evaluation metrics (Bedrock custom-metric format).

Used by the fast judge (`fastjudge.py`), by the managed evaluator's generation pipelines
(`fasteval.launch`) and by the demo app's live judge, so every score is on the same definitions:
ConstraintAdherence (binary), RecipientFit, ReasoningQuality, PickVariety (5-point, 0-1).
"""
from __future__ import annotations

import json

CONTEXT = ("You are grading a gift recommendation from the Octank Gift Concierge. The concierge "
           "received a shopper request and a list of CANDIDATES and had to pick items from that "
           "list. The response may include a <think> reasoning section followed by a JSON answer "
           "with picks, total and gift_message.")

DEFS = {
    "ConstraintAdherence": (
        "Hard rules, checked against the authoritative constraints: exactly the requested number "
        "of picks; every id exists in CANDIDATES; no duplicates; the sum of the picked prices is "
        "within the budget; no pick has an attribute the shopper asked to avoid, INCLUDING "
        "attributes implied by their situation (listed under implied_by_situation); and the "
        "response ends with the JSON answer (a response cut off mid-reasoning has no answer). "
        "Any violation is a failure."),
    "RecipientFit": (
        "How well the picks suit THIS recipient: relationship, age, occasion, stated interests, "
        "and what the message implies (e.g. 'has everything' -> experiences or consumables, "
        "'tiny studio' -> nothing bulky). A pick that only matches a keyword, or would suit almost "
        "anyone, is weak. Excellent means the picks are clearly the best available candidates."),
    "ReasoningQuality": (
        "Whether the reasoning efficiently works through the budget, the avoid list, the "
        "shopper's situation and specific details, and whether each 'why' cites concrete request "
        "details. Concise, correct reasoning is best: do not reward length, repetition, "
        "re-listing candidates or filler."),
    "PickVariety": (
        "Whether the picks are varied (different categories or uses) rather than near-identical "
        "items. For a single requested item, Excellent if it is a strong, specific choice."),
}


def _five(score_name: str) -> list[dict]:
    labels = ["Very poor", "Poor", "Acceptable", "Good", "Excellent"]
    return [{"definition": l, "value": {"floatValue": v}}
            for l, v in zip(labels, [0.0, 0.25, 0.5, 0.75, 1.0])]


def eval_metrics() -> list[dict]:
    vars_ = ("\n\nRequest and candidates: {{prompt}}\nResponse: {{prediction}}\n"
             "Ground truth (reference answer + authoritative constraints, prices and attributes): "
             "{{ground_truth}}")
    out = []
    for name, text in DEFS.items():
        scale = ([{"definition": "Violates at least one rule", "value": {"floatValue": 0.0}},
                  {"definition": "All rules satisfied", "value": {"floatValue": 1.0}}]
                 if name == "ConstraintAdherence" else _five(name))
        out.append({"customMetricDefinition": {"name": name,
                                               "instructions": f"{CONTEXT}\n\n{text}{vars_}",
                                               "ratingScale": scale}})
    return out


def eval_metrics_json() -> str:
    return json.dumps(eval_metrics())

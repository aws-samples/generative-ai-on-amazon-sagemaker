"""Deterministic constraint checks. Used to filter teacher outputs (rejection sampling) and to
score evaluation and app responses. The RLVR reward (judge/reward_lambda.py) implements the
same rules standalone. Pure Python, no dependencies."""
from __future__ import annotations

import json
import re

from .util import as_list

_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def split_thinking(text: str) -> tuple[str, str]:
    """Return (thinking, answer_text). Handles '<think>..</think>' and an unclosed '</think>'."""
    text = text or ""
    m = _THINK.search(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    if "</think>" in text:                    # template already opened <think>
        a, b = text.split("</think>", 1)
        return a.replace("<think>", "").strip(), b.strip()
    if '"picks"' not in text:                 # cut off mid-reasoning: the server never saw </think>,
        return text.replace("<think>", "").strip(), ""   # so it's all (unfinished) reasoning
    return "", text.strip()


def parse_answer(text: str) -> dict | None:
    _, ans = split_thinking(text)
    m = _FENCE.search(ans)
    cand = m.group(1) if m else ans
    s, e = cand.find("{"), cand.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        obj = json.loads(cand[s:e + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def check(answer: dict | None, req: dict, candidates: list[dict]) -> dict:
    """Return per-constraint results plus `hard_pass` (all hard constraints satisfied)."""
    by_id = {c["id"]: c for c in candidates}
    res = {"json_valid": answer is not None, "ids_in_candidates": False, "item_count_ok": False,
           "unique_ids": False, "within_budget": False, "no_avoid": False, "total_correct": False,
           "why_present": False, "category_variety": 0.0, "price_total": None}
    if not answer or not isinstance(answer.get("picks"), list):
        res["json_valid"] = False
        res["hard_pass"] = False
        return res
    picks = [p for p in answer["picks"] if isinstance(p, dict)]
    ids = [str(p.get("id", "")).strip() for p in picks]
    n = int(req.get("num_items") or 1)
    avoid = set(as_list(req.get("avoid"))) | set(as_list(req.get("hidden_avoid")))   # incl. implied needs
    res["ids_in_candidates"] = bool(ids) and all(i in by_id for i in ids)
    res["item_count_ok"] = len(ids) == n
    res["unique_ids"] = len(set(ids)) == len(ids)
    if res["ids_in_candidates"]:
        total = round(sum(float(by_id[i]["price"]) for i in ids), 2)
        res["price_total"] = total
        res["within_budget"] = total <= float(req["budget_total"]) + 1e-6
        res["no_avoid"] = not any(avoid & set(as_list(by_id[i]["attributes"])) for i in ids)
        try:
            res["total_correct"] = abs(float(answer.get("total", -1)) - total) <= 0.01 + 0.005 * total
        except (TypeError, ValueError):
            res["total_correct"] = False
        cats = {by_id[i]["category"] for i in ids}
        res["category_variety"] = round(len(cats) / max(len(ids), 1), 3)
    res["why_present"] = bool(picks) and all(len(str(p.get("why", "")).split()) >= 5 for p in picks)
    res["hard_pass"] = all(res[k] for k in ("json_valid", "ids_in_candidates", "item_count_ok",
                                            "unique_ids", "within_budget", "no_avoid"))
    return res


def check_text(text: str, req: dict, candidates: list[dict]) -> dict:
    thinking, _ = split_thinking(text)
    out = check(parse_answer(text), req, candidates)
    out["thinking_chars"] = len(thinking)
    return out

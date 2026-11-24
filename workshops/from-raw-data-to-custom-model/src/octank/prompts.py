"""The concierge prompt: a single user turn, identical for the teacher, SFT, RLVR, evaluation and
serving. Candidates are listed like a real catalog - name, category, price, description - with NO
interest or attribute tags, so the model has to infer them."""
from __future__ import annotations

import json

from .util import as_list, val

INSTRUCTIONS = """You are the Octank Gift Concierge. Recommend gifts from the CANDIDATES for the shopper below.

Rules:
1. Pick exactly {num_items} item(s), only from CANDIDATES, by id. No duplicates.
2. The total price of all picks must be at most the budget (${budget:.2f}).
3. Never pick an item that has anything the shopper wants to avoid ({avoid}). Candidates are not
   tagged - judge from each item's name and description. The shopper's situation may also rule
   items out even if they are not in the avoid list.
4. Choose what best fits the recipient, the occasion, their interests and what the message implies.
   Prefer variety across categories over near-identical items.
5. Think through the constraints and the promising candidates before you answer. Be concise.

Answer with ONLY this JSON (no markdown fences):
{{"picks": [{{"id": "OCT-....", "why": "<one sentence tied to specific request details>"}}], "total": <sum of pick prices>, "gift_message": "<short gift-card note>"}}"""


def _fmt_item(c: dict) -> str:
    desc = (c.get("description") or "")[:240]
    return f"[{c['id']}] {c['name']} | {c['category']} | ${float(c['price']):.2f} | {desc}"


def build_prompt(req: dict, candidates: list[dict]) -> str:
    avoid = ", ".join(as_list(req.get("avoid"))) or "nothing specified"
    head = INSTRUCTIONS.format(num_items=int(req.get("num_items") or 1),
                               budget=float(req["budget_total"]), avoid=avoid)
    form = {
        "relationship": val(req.get("relationship")), "recipient_age": val(req.get("age_band")),
        "occasion": val(req.get("occasion")), "budget_total": float(req["budget_total"]),
        "num_items": int(req.get("num_items") or 1), "interests": as_list(req.get("interests")),
        "avoid": as_list(req.get("avoid")), "notes": val(req.get("notes")) or "",
    }
    cands = "\n".join(_fmt_item(c) for c in candidates)
    return (f"{head}\n\nSHOPPER MESSAGE\n{req['message']}\n\nFORM FIELDS\n{json.dumps(form)}"
            f"\n\nCANDIDATES\n{cands}")


def format_completion(reasoning: str, answer_json: str) -> str:
    """SFT target in Qwen thinking format: <think>reasoning</think> then the JSON answer."""
    return f"<think>\n{reasoning.strip()}\n</think>\n\n{answer_json.strip()}"

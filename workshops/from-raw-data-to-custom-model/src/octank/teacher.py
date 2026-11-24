"""Teacher: gpt-oss-120b produces reasoning + answer for each request; deterministic checks keep
only verified-correct outputs (rejection sampling). These become SFT targets / eval references."""
from __future__ import annotations

import hashlib
import json

from . import checks, llm
from .config import GENERATOR_MODEL
from .prompts import build_prompt, format_completion

from .config import TEACHER_MAX_THINKING_CHARS as MAX_THINKING_CHARS  # noqa: E402  (~1,100 tokens)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha3_256(prompt.encode()).hexdigest()[:16]


def teach_one(req: dict, pool: list[dict], attempts: int = 2, region: str | None = None,
              effort: str = "medium", max_thinking_chars: int = MAX_THINKING_CHARS) -> dict:
    prompt = build_prompt(req, pool)
    ph = prompt_hash(prompt)
    last = None
    for a in range(attempts):
        res = llm.converse(GENERATOR_MODEL, prompt, max_tokens=6000,
                           temperature=0.6 if a == 0 else 0.9, reasoning_effort=effort,
                           region=region)
        ans = checks.parse_answer(res.text)
        chk = checks.check(ans, req, pool)
        last = {"request_id": req["request_id"], "prompt_hash": ph, "ok": False,
                "attempts": a + 1,
                "reasoning_chars": len(res.reasoning), "out_tokens": res.output_tokens,
                "checks": chk}
        if not chk["hard_pass"] or not res.reasoning or len(res.reasoning) > max_thinking_chars:
            continue
        ans["total"] = chk["price_total"]                    # fix arithmetic slips
        answer_json = json.dumps({"picks": [{"id": p["id"], "why": p.get("why", "")}
                                            for p in ans["picks"]],
                                  "total": ans["total"],
                                  "gift_message": ans.get("gift_message", "")})
        last.update(ok=True, prompt=prompt, reasoning=res.reasoning, answer=answer_json,
                    completion=format_completion(res.reasoning, answer_json))
        return last
    return last


def teach_many(items: list[tuple[dict, list[dict]]], cache=None, workers: int = 48,
               region: str | None = None, max_thinking_chars: int = MAX_THINKING_CHARS) -> list[dict]:
    def fresh(it):   # cached record is only valid if it was produced for the same prompt
        c = cache.get(it[0]["request_id"]) if cache is not None else None
        return c is not None and c.get("prompt_hash") == prompt_hash(build_prompt(*it))

    todo = [it for it in items if not fresh(it)]

    def run(it):
        return teach_one(it[0], it[1], region=region, max_thinking_chars=max_thinking_chars)

    def persist(_it, res):
        if cache is not None and isinstance(res, dict):
            cache.put(slim(res))

    results = llm.pmap(run, todo, workers=workers, desc="teacher", on_result=persist)
    out = {r["request_id"]: r for _, r, e in results if r}
    if cache is not None:
        out = {**{it[0]["request_id"]: cache.get(it[0]["request_id"]) for it in items
                  if fresh(it)}, **out}
    by_id = {it[0]["request_id"]: it for it in items}
    return [inflate(out[i], *by_id[i]) for i in by_id if i in out]


# The prompt and the SFT completion are rebuilt from the request + pool and the stored reasoning +
# answer, so cache records stay small (~85% of a full record was the prompt).
_DERIVED = ("prompt", "completion")


def slim(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in _DERIVED}


def inflate(rec: dict, req: dict, pool: list[dict]) -> dict:
    rec = dict(rec)
    if rec.get("ok"):
        rec["prompt"] = build_prompt(req, pool)
        rec["completion"] = format_completion(rec["reasoning"], rec["answer"])
    return rec

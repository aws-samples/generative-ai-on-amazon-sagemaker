"""Hybrid RLVR reward for the Octank Gift Concierge: exact rules in code + LLM taste.

Ground truth per prompt (reward_model.ground_truth, from formats.constraints_json + extras):
  budget_total, num_items, avoid (explicit + hidden), implied_by_situation.avoid (hidden only),
  interests, difficulty (0-100), reference (teacher answer or null),
  candidates {id: {price, category, attributes, interests}}

Reward pyramid (score in [-1, 1]):
  L1 structure : no parsable JSON answer (cut off / malformed)            -> -1.0
                 bad ids / wrong count / duplicates                        -> -0.6
  L2 rules     : over budget -0.1 | explicit avoided attribute -0.3 | HIDDEN need violated -0.4
                 (several broken -> the lowest applies, minus 0.1)
  L3 quality   : all rules pass -> 0.30
                 + 0.30 * fit      (share of picks matching a stated interest, scaled by what the
                                    pool allows, + interest coverage) - exact, from the tags
                 + 0.10 * variety  (distinct categories / picks)
                 + 0.30 * taste    (LLM: fit, reasoning, message vs the reference; 0-1)
                 - length penalty  (only above a difficulty-scaled target; up to 0.25)
  anti-gaming  : echo of the prompt -> -1; repetitive thinking -> -0.4; empty "why" -> -0.25

Taste: one Bedrock Converse call per sample (env TASTE_MODEL, default gpt-oss-20b), computed for
the whole batch IN PARALLEL before scoring. If a call fails or the batch budget runs out, the
sample's taste falls back to its deterministic fit (metric taste_ok=0), so a Bedrock hiccup never
produces garbage rewards. TASTE_WEIGHT=0 makes the reward fully deterministic.

lambda_handler is the aws-ai-ml skill template's handler; the ONE added line (marked) prefetches
taste scores for the batch in parallel. Without it, calls would run one sample at a time.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, List, Optional

# Only open-weight models available in BOTH us-east-1 and us-west-2 (workshop constraint; no Claude
# for anything that shapes training). Anything else falls back to gpt-oss-20b.
BOTH_REGION_MODELS = {"openai.gpt-oss-20b-1:0", "openai.gpt-oss-120b-1:0", "qwen.qwen3-32b-v1:0"}
TASTE_MODEL = os.environ.get("TASTE_MODEL", "openai.gpt-oss-20b-1:0")
if TASTE_MODEL not in BOTH_REGION_MODELS:
    TASTE_MODEL = "openai.gpt-oss-20b-1:0"
TASTE_WEIGHT = float(os.environ.get("TASTE_WEIGHT", "0.30"))
TASTE_WORKERS = int(os.environ.get("TASTE_WORKERS", "32"))
TASTE_BUDGET_S = float(os.environ.get("TASTE_BUDGET_S", "120"))   # wall-clock cap per batch
_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_TASTE: Dict[int, Optional[float]] = {}                          # batch index -> taste (0-1)


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------
def _response(sample: Dict[str, Any]) -> str:
    """The assistant turn (skill template's extraction)."""
    response = ""
    for msg in sample.get("messages", sample.get("prompt", [])):
        if msg.get("role") == "assistant":
            response = msg.get("content", "")
    return response or ""


def _prompt(sample: Dict[str, Any]) -> str:
    for msg in sample.get("messages", sample.get("prompt", [])):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _split(text: str):
    """(thinking, answer). A response cut off before </think> is all thinking."""
    m = _THINK.search(text)
    if m:
        return m.group(1), text[m.end():]
    if "</think>" in text:
        a, b = text.split("</think>", 1)
        return a, b
    if '"picks"' not in text:
        return text, ""
    return "", text


def _answer(answer_text: str) -> Optional[dict]:
    s, e = answer_text.find("{"), answer_text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        obj = json.loads(answer_text[s:e + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and isinstance(obj.get("picks"), list) else None


def _gt(sample: Dict[str, Any]) -> dict:
    gt = sample.get("reward_model", {}).get("ground_truth", "") or "{}"
    return json.loads(gt) if isinstance(gt, str) else gt


# ---------------------------------------------------------------------------------------------
# Deterministic parts
# ---------------------------------------------------------------------------------------------
def _fit(ids: List[str], gt: dict) -> float:
    """Share of picks that match a stated interest, relative to what the pool makes possible,
    plus coverage of the shopper's distinct interests. 0-1."""
    cands, interests = gt.get("candidates", {}), set(gt.get("interests", []))
    if not ids or not interests:
        return 0.5
    budget, n, avoid = float(gt["budget_total"]), int(gt["num_items"]), set(gt.get("avoid", []))
    possible = [c for c in cands.values() if interests & set(c.get("interests", []))
                and not avoid & set(c.get("attributes", [])) and float(c["price"]) <= budget]
    best_share = min(1.0, len(possible) / max(n, 1)) or 1.0
    matched = [i for i in ids if interests & set(cands[i].get("interests", []))]
    share = (len(matched) / len(ids)) / best_share
    covered = set().union(*[set(cands[i].get("interests", [])) for i in matched]) & interests if matched else set()
    coverage = len(covered) / min(len(interests), len(ids))
    return max(0.0, min(1.0, 0.7 * share + 0.3 * coverage))


def _length_penalty(thinking: str, answer_text: str, difficulty: float) -> float:
    """~tokens = chars/4. Free up to a target that grows with difficulty (450 -> 900 tokens);
    beyond it, linear up to -0.25 at 2x the target."""
    approx_tokens = (len(thinking) + len(answer_text)) / 4.0
    target = 450 + 4.5 * max(0.0, min(100.0, difficulty))
    over = max(0.0, approx_tokens - target) / target
    return 0.25 * min(1.0, over)


def _repetitive(thinking: str) -> bool:
    """Padding / loops: many repeated sentences in the reasoning."""
    sents = [s.strip().lower() for s in re.split(r"[.\n]", thinking) if len(s.strip()) > 20]
    return len(sents) >= 12 and len(set(sents)) / len(sents) < 0.6


# ---------------------------------------------------------------------------------------------
# LLM taste (parallel per batch)
# ---------------------------------------------------------------------------------------------
TASTE_PROMPT = """You are grading a gift recommendation from the Octank Gift Concierge.
The hard rules (budget, item count, avoid list, needs implied by the shopper's situation) were
already checked by code and PASSED. Grade only quality, from 0 to 10 each:
- fit: how well the picks suit THIS recipient (relationship, age, occasion, interests, what the
  message implies). Keyword-only or generic picks are weak; the best available candidates are 10.
- reasoning: does the reasoning efficiently work through the request and give each pick a "why"
  tied to specific details? Do NOT reward length or repetition.
- message: is the gift message warm, specific to the relationship and occasion?
A REFERENCE answer from a strong model is shown for calibration: equal quality = 7; better = 8-10.

REQUEST AND CANDIDATES:
{prompt}

REFERENCE ANSWER:
{reference}

RESPONSE TO GRADE (reasoning, then answer):
{response}

Return ONLY JSON: {{"fit": <0-10>, "reasoning": <0-10>, "message": <0-10>}}"""


def _taste_one(sample: Dict[str, Any]) -> Optional[float]:
    import boto3
    from botocore.config import Config
    client = boto3.client("bedrock-runtime", config=Config(read_timeout=40, connect_timeout=5,
                                                           retries={"max_attempts": 3, "mode": "adaptive"}))
    gt = _gt(sample)
    response = _response(sample)
    text = TASTE_PROMPT.format(prompt=_prompt(sample)[:9000], reference=json.dumps(gt.get("reference")),
                               response=response[-6000:])
    args = {"modelId": TASTE_MODEL, "messages": [{"role": "user", "content": [{"text": text}]}],
            "inferenceConfig": {"maxTokens": 800, "temperature": 0}}
    if "gpt-oss" in TASTE_MODEL:
        args["additionalModelRequestFields"] = {"reasoning_effort": "low"}
    out = client.converse(**args)["output"]["message"]["content"]
    t = "".join(c.get("text", "") for c in out)
    m = re.search(r"\{[^{}]*\}", t, re.S)
    d = json.loads(m.group(0)) if m else {}
    vals = [float(d[k]) for k in ("fit", "reasoning", "message") if k in d]
    if len(vals) < 3:
        return None
    return max(0.0, min(1.0, (0.45 * vals[0] + 0.40 * vals[1] + 0.15 * vals[2]) / 10.0))


def _eligible(sample: Dict[str, Any]) -> bool:
    """Taste only matters for answers that pass every rule: don't spend LLM calls on the rest."""
    try:
        gt = _gt(sample)
        thinking, answer_text = _split(_response(sample))
        ans = _answer(answer_text)
        return ans is not None and _rules(ans, gt)[0] == 0.0
    except Exception:  # noqa: BLE001
        return False


def _prefetch_taste(batch: List[Dict[str, Any]]) -> None:
    _TASTE.clear()
    if TASTE_WEIGHT <= 0:
        return
    idx = [i for i, s in enumerate(batch) if _eligible(s)]
    if not idx:
        return
    t0 = time.time()
    # No `with` block: its exit would wait for running calls past the budget. TASTE_BUDGET_S is a hard
    # cap - stragglers are abandoned (their samples fall back to the deterministic fit).
    ex = ThreadPoolExecutor(min(TASTE_WORKERS, len(idx)))
    try:
        futs = {ex.submit(_taste_one, batch[i]): i for i in idx}
        done, _ = wait(futs, timeout=TASTE_BUDGET_S)
        for f in done:
            try:
                _TASTE[futs[f]] = f.result()
            except Exception:  # noqa: BLE001
                _TASTE[futs[f]] = None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    _TASTE["_elapsed_s"] = round(time.time() - t0, 1)  # type: ignore[index]


# ---------------------------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------------------------
def _rules(ans: dict, gt: dict):
    """(penalty, flags). penalty 0.0 = all rules pass; otherwise the (negative) score to use."""
    cands = gt.get("candidates", {})
    picks = [p for p in ans.get("picks", []) if isinstance(p, dict)]
    ids = [str(p.get("id", "")) for p in picks]
    f = {"ids_valid": bool(ids) and all(i in cands for i in ids),
         "count_ok": len(ids) == int(gt["num_items"]), "unique": len(set(ids)) == len(ids)}
    if not all(f.values()):
        return -0.6, {**f, "within_budget": False, "no_avoid": False, "no_hidden": False}
    hidden = set((gt.get("implied_by_situation") or {}).get("avoid", []))
    explicit = set(gt.get("avoid", [])) - hidden
    attrs = set().union(*[set(cands[i].get("attributes", [])) for i in ids])
    f["within_budget"] = sum(float(cands[i]["price"]) for i in ids) <= float(gt["budget_total"]) + 1e-6
    f["no_avoid"] = not explicit & attrs
    f["no_hidden"] = not hidden & attrs
    broken = [p for k, p in (("within_budget", -0.1), ("no_avoid", -0.3), ("no_hidden", -0.4)) if not f[k]]
    if not broken:
        return 0.0, f
    return min(broken) - 0.1 * (len(broken) - 1), f


def reward_function(sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    gt = _gt(sample)
    response = _response(sample)
    thinking, answer_text = _split(response)
    ans = _answer(answer_text)
    m = {"answer_present": 0.0, "rules_pass": 0.0, "within_budget": 0.0, "no_avoid": 0.0,
         "no_hidden": 0.0, "fit": 0.0, "variety": 0.0, "taste": 0.0, "taste_ok": 0.0,
         "length_penalty": 0.0, "approx_tokens": (len(thinking) + len(answer_text)) / 4.0}

    if ans is None:
        score = -1.0                                            # L1: no answer (cut off)
    else:
        m["answer_present"] = 1.0
        penalty, flags = _rules(ans, gt)
        for k in ("within_budget", "no_avoid", "no_hidden"):
            m[k] = float(flags.get(k, False))
        if penalty < 0:
            score = penalty                                     # L1/L2 failures
        else:
            m["rules_pass"] = 1.0
            cands = gt["candidates"]
            ids = [str(p.get("id")) for p in ans["picks"]]
            m["fit"] = _fit(ids, gt)
            m["variety"] = len({cands[i]["category"] for i in ids}) / len(ids)
            taste = _TASTE.get(index)
            m["taste_ok"] = float(taste is not None)
            m["taste"] = taste if taste is not None else m["fit"]
            m["length_penalty"] = _length_penalty(thinking, answer_text, float(gt.get("difficulty", 50)))
            score = (0.30 + 0.30 * m["fit"] + 0.10 * m["variety"] + TASTE_WEIGHT * m["taste"]
                     - m["length_penalty"])
            if any(len(str(p.get("why", "")).split()) < 4 for p in ans["picks"]):
                score -= 0.25                                   # anti-gaming: empty justifications
        if _repetitive(thinking):
            score -= 0.4                                        # anti-gaming: padding / loops
        if answer_text.strip() and answer_text.strip()[:200] in _prompt(sample):
            score = -1.0                                        # anti-gaming: echo

    metrics = [{"name": "hybrid_reward", "value": float(min(1.0, max(-1.0, score))), "type": "Reward"}] + \
        [{"name": k, "value": float(v), "type": "Metric"} for k, v in m.items()]
    return {"id": str(sample.get("id", f"sample-{index:03d}")),
            "aggregate_reward_score": float(min(1.0, max(-1.0, score))),
            "metrics_list": metrics}


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda Handler for reward function
    """
    try:
        # Extract batch from event
        batch = event.get("input", event) if isinstance(event, dict) else event
        if "batch" in event:
            batch = event.get("batch", [])
        elif "body" in event:
            body = json.loads(event.get("body", "{}"))
            batch = body.get("batch", [])

        if not batch:
            return {"error": "Missing or empty batch"}

        _prefetch_taste(batch)  # ADDED (only change vs the skill template): parallel LLM taste

        # Process each sample
        results = []
        for i, sample in enumerate(batch):
            try:
                result = reward_function(sample, i)
                results.append(result)
            except Exception as e:
                return {"error": str(e)}

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(results),
        }
    except Exception as e:
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}

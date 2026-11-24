"""Synthetic shopper requests: structured "form" fields + a free-text message, with planted flaws.

Technique (identical for instructor and attendees - only `n` and the flaw preset differ):
  1. Sample a structured spec per request from configurable distributions (with a planted
     occasion skew and thin coverage for rare occasions / small budgets). ~45% of requests carry a
     *situation* mentioned only in the message ("recently got sober") that implies attributes to
     avoid (`hidden_avoid`); most of those also get an interest that conflicts with it.
  2. gpt-oss-120b writes the shopper's free-text message for a batch of specs, in varied voices.
  3. Plant data-quality flaws (missing fields, contradictions, exact + near duplicates, toxic and
     ambiguous requests) so every curation step has something real to find. The hidden `_flaw`
     column is the answer key used to score the curation steps.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

import pandas as pd

from . import llm
from .cache import content_hash
from .config import GENERATOR_MODEL
from .vocab import (BUDGET_BAND_WEIGHTS, BUDGET_BANDS, IMPLICIT_CUES, INTERESTS, OCCASION_WEIGHTS,
                    RELATIONSHIPS)

NOTES = [
    "says they have everything", "just moved to a new city", "we live far apart now",
    "recently retired", "picked up a new hobby this year", "minimalist, hates clutter",
    "loves surprises", "very practical person", "going through a tough time lately",
    "eco-conscious", "works long hours", "just started college", "new parent, exhausted",
    "loves hosting friends", "into anything handmade", "", "", "",
]
STYLES = ["terse, lowercase, no punctuation", "chatty and warm, a bit rambling",
          "polite and formal", "excited, uses a couple of emoji", "typo-prone and rushed",
          "matter-of-fact", "anxious about getting it right", "funny and self-deprecating"]

CHILD_RELS = {"son", "daughter", "niece", "nephew"}
ELDER_RELS = {"grandma", "grandpa"}


@dataclass
class FlawRates:
    missing_field: float
    contradiction: float
    exact_dup: float
    near_dup: float
    toxic: float
    ambiguous: float


INSTRUCTOR_FLAWS = FlawRates(0.08, 0.03, 0.04, 0.06, 0.02, 0.03)
ATTENDEE_FLAWS = FlawRates(0.05, 0.02, 0.02, 0.03, 0.015, 0.02)   # enough to illustrate


def _age_band(rel: str, rng: random.Random) -> str:
    if rel in CHILD_RELS:
        return rng.choice(["0-4", "5-12", "13-17", "18-29"])
    if rel in ELDER_RELS:
        return "70+"
    if rel in {"mom", "dad", "in_law"}:
        return rng.choice(["30-49", "50-69", "70+"])
    return rng.choice(["18-29", "30-49", "50-69"])


CUE_RATE = 0.45
CHILD_ONLY_CUES = {"has a toddler who puts everything in their mouth"}
# Conflict pairing: a cued request also gets an interest that the hidden need conflicts with, so
# the strongest keyword matches are exactly the items to avoid ("loves cocktails" + "sober").
CONFLICT_INTERESTS = {
    "alcohol": ["cocktails", "wine"], "caffeine": ["coffee", "tea"],
    "fragrance": ["candles", "skincare"], "nuts": ["chocolate", "baking"],
    "leather": ["fashion", "jewelry"], "bulky": ["home decor", "woodworking"],
    "clutter": ["home decor", "crafts"], "gadget": ["tech", "smart home", "video games"],
    "screen": ["video games", "film"], "small_parts": ["puzzles", "crafts"],
    "noisy": ["guitar", "podcasts"],
}
CONFLICT_RATE = 0.8          # share of cued requests that also get a conflicting interest

# Explicit "avoid" options limited to attributes a reader can infer from a name/description
# (prompts show no tags). Excluded: logo, plastic, eco, luxury, handmade, fragile, novelty, clutter.
INFERABLE_AVOID = ["gadget", "alcohol", "fragrance", "food", "caffeine", "nuts", "screen", "noisy",
                   "subscription", "experience", "consumable", "personalized", "leather", "bulky",
                   "small_parts"]


def sample_spec(rng: random.Random, i: int, occasion: str | None = None,
                budget_band_idx: int | None = None, age_band: str | None = None) -> dict:
    rel = rng.choice(RELATIONSHIPS)
    occ = occasion or rng.choices(list(OCCASION_WEIGHTS), weights=OCCASION_WEIGHTS.values())[0]
    bi = budget_band_idx if budget_band_idx is not None else \
        rng.choices(range(len(BUDGET_BANDS)), weights=BUDGET_BAND_WEIGHTS)[0]
    lo, hi = BUDGET_BANDS[bi]
    budget = max(15, int(round(rng.uniform(max(lo, 15), hi) / 5) * 5))
    num_items = rng.choices([1, 2, 3], weights=[0.25, 0.45, 0.30])[0]     # more multi-item (sum traps)
    if budget < 40:
        num_items = min(num_items, 2 if budget >= 25 else 1)
    age = age_band or _age_band(rel, rng)
    cue, hidden = "", []
    if rng.random() < CUE_RATE:
        options = [c for c in IMPLICIT_CUES if c not in CHILD_ONLY_CUES or age in ("0-4", "30-49")]
        cue = rng.choice(options)
        hidden = list(IMPLICIT_CUES[cue])
    avoid_pool = [a for a in INFERABLE_AVOID if a not in hidden]
    interests = rng.sample(INTERESTS, rng.choices([1, 2, 3], weights=[0.25, 0.5, 0.25])[0])
    if hidden and rng.random() < CONFLICT_RATE:
        opts = [i for h in hidden for i in CONFLICT_INTERESTS.get(h, []) if i not in interests]
        if opts:
            interests = [rng.choice(opts)] + interests[:2]
    return {
        "request_id": f"r{i:06d}", "relationship": rel, "age_band": age, "occasion": occ,
        "budget_total": float(budget), "num_items": num_items,
        "interests": interests,
        "avoid": rng.sample(avoid_pool, rng.choices([0, 1, 2], weights=[0.55, 0.33, 0.12])[0]),
        "notes": rng.choice(NOTES), "style": rng.choice(STYLES),
        "situation": cue, "hidden_avoid": hidden,
    }


MSG_PROMPT = """You write realistic messages that online shoppers type into a gift-concierge chat
box at Octank, a gift shop. For EACH spec below, write one message in the requested voice.

Each message must naturally convey: who the gift is for (relationship + rough age), the
occasion, the total budget, how many gifts they want, the recipient's interests, anything to
avoid, and the note (if any).

If the spec has a "situation", mention it in passing as a fact about the recipient's life. NEVER
say what to avoid because of it, and never name a product type to avoid for it - the reader must
infer that themselves. (Example: situation "recently got sober" -> "she's been sober six months
and is doing great", NOT "so no alcohol".)

Paraphrase - do not copy field names, and do not list fields. Vary length (1-4 sentences) and
phrasing. Do not invent extra constraints.

SPECS:
{specs}

Return ONLY a JSON list of objects: {{"request_id": "...", "message": "..."}} in the same order."""


def spec_view(s: dict) -> dict:
    keys = ["request_id", "relationship", "age_band", "occasion", "budget_total", "num_items",
            "interests", "avoid", "notes", "style", "situation"]
    return {k: s[k] for k in keys}


def write_messages(specs: list[dict], batch_size: int = 10, workers: int = 32,
                   region: str | None = None, cache=None, prompt_tpl: str | None = None,
                   view=None) -> dict[str, str]:
    """gpt-oss-120b writes free-text messages for specs (batched). Returns {request_id: msg}.
    If `cache` (JsonlCache keyed by request_id) is given, completed batches are persisted
    immediately and already-cached specs are skipped (resumable)."""
    prompt_tpl = prompt_tpl or MSG_PROMPT
    view = view or spec_view
    out: dict[str, str] = {}
    if cache is not None:          # reuse only if cached for the identical spec
        hit = [s for s in specs if cache.fresh(s["request_id"], json.dumps(view(s), sort_keys=True))]
        out.update({s["request_id"]: cache.get(s["request_id"])["message"] for s in hit})
        hit_ids = {s["request_id"] for s in hit}
        specs = [s for s in specs if s["request_id"] not in hit_ids]
    batches = [specs[i:i + batch_size] for i in range(0, len(specs), batch_size)]

    def run(batch):
        prompt = prompt_tpl.format(specs=json.dumps([view(s) for s in batch], indent=1))
        data, _ = llm.converse_json(GENERATOR_MODEL, prompt, max_tokens=4000, temperature=0.9,
                                    reasoning_effort="low", region=region)
        ids = {s["request_id"] for s in batch}
        return {d["request_id"]: str(d["message"]).strip() for d in data
                if isinstance(d, dict) and d.get("request_id") in ids and d.get("message")}

    def persist(batch, res):
        if cache is not None and isinstance(res, dict):
            spec_by_id = {s["request_id"]: s for s in batch}
            cache.put_many({"request_id": k, "message": v,
                            "input_hash": content_hash(json.dumps(view(spec_by_id[k]),
                                                                  sort_keys=True))}
                           for k, v in res.items())

    for _, res, err in llm.pmap(run, batches, workers=workers, desc="messages",
                                on_result=persist):
        if res:
            out.update(res)
    return out


# --- Planted flaws ------------------------------------------------------------------------------
# Flawed requests are composed from slots (opener + intent + detail + budget + closer) so every
# planted toxic/ambiguous message is distinct: exact dedup must not silently remove them before
# the safety filter and human review get a chance to catch them.
TOXIC_INTENTS = [
    "a gag gift for my {who} that tells {pr_o} {pr_c} fat and useless",
    "something to scare my {who} into leaving town",
    "a gift that humiliates my {who} because of {pr_pos} accent",
    "a present that makes my {who} feel worthless",
    "something that says nobody wants {pr_o} here for my {who}",
    "a gift that insults my {who}'s religion",
    "something creepy to leave on my {who}'s car so {pr} {pr_k} i'm watching",
    "a gift that will get my {who} kicked out, maybe something illegal",
    "a 'funny' gift mocking my {who}'s disability",
    "something threatening for my {who}, nothing traceable",
    "lingerie for my {who} even though {pr} told me to leave {pr_o} alone",
    "a gift making fun of my {who}'s ethnicity",
    "something telling my {who} to drop dead",
    "a present to shame my {who} about {pr_pos} weight",
    "a joke gift telling my {who}'s immigrant family to go home",
]
AMBIGUOUS_INTENTS = [
    "something for my ex... not sure where we stand",
    "a romantic gift for my married {who}, {pr} doesn't know how I feel yet",
    "a gift for someone, not sure what they like at all",
    "3 gifts that feel luxurious and expensive but under $20 total",
    "something for my {who}'s kid, no idea how old they are",
    "a present for my {who} that shows I'm interested in more than work",
    "a birthday or maybe anniversary gift for my partner, can't remember which is next",
    "a gift for my {who} but {pr} hates gifts and wants nothing",
    "something my teenage neighbor would like, just between us",
    "a gift that's alcohol but not alcohol for my sober {who}",
    "a gift for my {who} or maybe their spouse, whoever opens it",
    "something for my {who} to apologize, but not too much of an apology",
]
_WHO = ["coworker", "boss", "neighbor", "roommate", "sister", "brother", "ex", "landlord",
        "teacher", "brother-in-law", "mother-in-law", "classmate", "friend", "cousin", "manager",
        "tenant", "stepdad", "aunt", "new hire", "teammate"]
_PRON = [("she", "she's", "her", "her", "knows"), ("he", "he's", "him", "his", "knows"),
         ("they", "they're", "them", "their", "know")]
_OPEN = ["", "hey, ", "ok so ", "quick q - ", "need help: ", "honestly ", "lol ", "pls help, ",
         "looking for ", "can you find ", "i want ", "help me pick "]
_DETAIL = ["", " for christmas", " for their birthday", " this weekend", " before friday",
           " for the office party", " to drop off tomorrow", " for the holidays", " asap"]
_BUDGET = ["", " under ${b}", " around ${b}", " ${b} max", " budget is ${b}", " cheap",
           " no more than ${b}", " like ${b}"]
_WHY = ["", " because {pr} annoyed me", " since {pr_c} been awful lately", " after what {pr} did",
        " because i'm done being nice", " cause {pr} deserves it", " because everyone agrees",
        " since we had a fight", " to get back at {pr_o}", " because {pr} won't stop bragging",
        " after the last party", " because it would be hilarious"]
_CTX = ["", " - we haven't talked in a while", ", it's complicated", ", long story",
        ", they're hard to shop for", ", I barely know them honestly", ", trying not to overthink it",
        ", my sister says I'm overthinking", ", it's for the {who} I mentioned", ", kind of awkward",
        ", first time doing this", ", want it to land right", ", it's been a weird year",
        ", don't want to get it wrong", ", no pressure but it matters"]
_CLOSE = ["", ".", "!", " thx", " lol", " - ideas?", " please", " any ideas?", " 🙃"]


def _compose(intents: list[str], rng: random.Random, budget: float | None) -> str:
    pr = rng.choice(_PRON)
    who = rng.choice(_WHO)
    kw = dict(who=who, pr=pr[0], pr_c=pr[1], pr_o=pr[2], pr_pos=pr[3], pr_k=pr[4])
    intent = rng.choice(intents).format(**kw)
    why = rng.choice(_WHY if intents is TOXIC_INTENTS else _CTX).format(**kw)
    b = int(budget or rng.choice([20, 30, 40, 50, 75]))
    msg = (rng.choice(_OPEN) + intent + why + rng.choice(_DETAIL) +
           rng.choice(_BUDGET).replace("${b}", f"${b}") + rng.choice(_CLOSE))
    return msg[0].upper() + msg[1:] if rng.random() < 0.5 else msg


def _perturb(text: str, rng: random.Random) -> str:
    """Near-duplicate: small surface edits that keep most word shingles intact."""
    subs = [("gift", "present"), ("budget", "spend"), ("around", "about"),
            ("looking for", "searching for"), ("something", "smth"), ("really", "rly"),
            ("thanks", "thx"), ("please", "pls")]
    t = text
    for a, b in rng.sample(subs, 3):
        t = re.sub(rf"\b{a}\b", b, t, flags=re.I)
    ops = [lambda s: s.lower(), lambda s: re.sub(r"[!.,]", "", s),
           lambda s: "hi! " + s, lambda s: s + " thanks so much",
           lambda s: s.replace("  ", " ")]
    for op in rng.sample(ops, 2):
        t = op(t)
    return t


def plant_flaws(df: pd.DataFrame, rates: FlawRates, seed: int = 11) -> pd.DataFrame:
    rng = random.Random(seed)
    df = df.copy()
    df["_flaw"] = "none"
    n = len(df)
    idx = list(range(n))
    rng.shuffle(idx)
    cursor = 0

    def take(rate):
        nonlocal cursor
        k = int(round(n * rate))
        sel = idx[cursor:cursor + k]
        cursor += k
        return sel

    slot_rng = random.Random(seed + 1000)          # separate stream: other flaws stay identical
    for i in take(rates.toxic):
        df.at[i, "message"] = _compose(TOXIC_INTENTS, slot_rng, df.at[i, "budget_total"])
        df.at[i, "_flaw"] = "toxic"
    for i in take(rates.ambiguous):
        df.at[i, "message"] = _compose(AMBIGUOUS_INTENTS, slot_rng, df.at[i, "budget_total"])
        df.at[i, "_flaw"] = "ambiguous"
    df["_orig_occasion"] = df["occasion"]
    df["_orig_budget"] = df["budget_total"]
    for i in take(rates.missing_field):
        col = rng.choice(["budget_total", "occasion"])
        df.at[i, col] = None
        df.at[i, "_flaw"] = f"missing_{'budget' if col == 'budget_total' else 'occasion'}"
    for i in take(rates.contradiction):
        cur = df.at[i, "occasion"]
        df.at[i, "occasion"] = rng.choice([o for o in OCCASION_WEIGHTS if o != cur])
        df.at[i, "_flaw"] = "contradiction"

    clean_rows = df[df["_flaw"] == "none"]
    next_id = 0                      # planted duplicates get their own "x" id space (never collide)
    extra = []
    for _ in range(int(round(n * rates.exact_dup))):
        r = clean_rows.iloc[rng.randrange(len(clean_rows))].to_dict()
        r.update(request_id=f"x{next_id:06d}", _flaw="exact_dup", _dup_of=r["request_id"])
        next_id += 1
        extra.append(r)
    for _ in range(int(round(n * rates.near_dup))):
        r = clean_rows.iloc[rng.randrange(len(clean_rows))].to_dict()
        r.update(request_id=f"x{next_id:06d}", _flaw="near_dup", _dup_of=r["request_id"],
                 message=_perturb(r["message"], rng))
        next_id += 1
        extra.append(r)
    out = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def generate_requests(n: int, rates: FlawRates, seed: int = 42, workers: int = 32,
                      id_offset: int = 0, region: str | None = None, cache=None) -> pd.DataFrame:
    """End-to-end raw request generation (specs -> messages -> planted flaws).
    Deterministic given (n, seed): the same specs are sampled every run, so a resumed run
    reuses cached messages and produces the identical dataset."""
    rng = random.Random(seed)
    specs = [sample_spec(rng, id_offset + i) for i in range(n)]
    msgs = write_messages(specs, workers=workers, region=region, cache=cache)
    rows = [dict(s, message=msgs[s["request_id"]], source="generated")
            for s in specs if s["request_id"] in msgs]
    df = pd.DataFrame(rows)
    df["_dup_of"] = None
    return plant_flaws(df, rates, seed=seed + 1)


# ---- Extreme requests (demo scenarios and a share of the RL prompts) ---------------------------
EXTREME_MIN_SCORE = 70


def extreme_spec(rng: random.Random, i: int, catalog: pd.DataFrame) -> dict | None:
    """One Extreme request (or None): hidden need + 1-2 explicit avoids, 3 items, budget 1.00-1.12x
    the cheapest relevant set; kept only if graded >= 70 with BOTH 40- and 80-candidate pools.
    Used for the Extreme demo scenarios and, with a different seed, for RL training prompts."""
    import math
    from . import difficulty
    from .pools import build_pool, build_pool_80, cheapest_relevant_set
    s = sample_spec(rng, i)
    if not s["hidden_avoid"]:
        return None
    s["num_items"] = 3
    extra = [a for a in INFERABLE_AVOID if a not in s["hidden_avoid"] and a not in s["avoid"]]
    s["avoid"] = list(dict.fromkeys(s["avoid"] + rng.sample(extra, rng.choice([1, 2]))))[:3]
    floor = cheapest_relevant_set(s, catalog, 3)
    if floor is None:
        return None
    s["budget_total"] = float(max(20, math.ceil(floor * rng.uniform(1.0, 1.12) / 5) * 5))
    g40 = difficulty.grade(s, build_pool(s, catalog))
    g80 = difficulty.grade(s, build_pool_80(s, catalog))
    if g40["feasible"] and g80["feasible"] and min(g40["score"], g80["score"]) >= EXTREME_MIN_SCORE:
        return {**s, "_difficulty": {"40": g40["score"], "80": g80["score"]}}
    return None


def stats(df: pd.DataFrame) -> dict:
    return {"rows": len(df), "with_hidden_cue_pct": round(100 * df["situation"].astype(bool).mean(), 1),
            "multi_item_pct": round(100 * (df["num_items"] > 1).mean(), 1)}

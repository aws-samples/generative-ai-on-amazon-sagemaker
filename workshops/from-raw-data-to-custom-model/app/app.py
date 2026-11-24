"""Octank Gift Concierge - side-by-side demo of base vs fine-tuned adapters on one endpoint.

    streamlit run app/app.py --server.port 8501        (or scripts/restart_app.sh [attendee|demo])
    open <Studio URL>/jupyterlab/default/proxy/8501/

OCTANK_APP_RUN picks the state file with the endpoint + inference component names:
"attendee" (written by Labs 0 and 4) or "demo" (the shared instructor endpoint).
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from octank import catalog, checks, config, difficulty, formats, metrics, pools, serving, state  # noqa: E402
from octank.prompts import build_prompt  # noqa: E402
from octank.fastjudge import judge_many  # noqa: E402
from octank.vocab import ATTRIBUTES, INTERESTS, OCCASIONS, RELATIONSHIPS  # noqa: E402


def md(text) -> str:
    """Escape '$' for st.markdown/st.write/st.info/st.caption: Streamlit renders text between two
    '$' signs as a LaTeX formula (no spaces, no wrapping -> clipped in narrow columns). Judge
    explanations, pick reasons and totals are full of prices."""
    return str(text).replace("\\$", "$").replace("$", "\\$")

st.set_page_config(page_title="Octank Gift Concierge", page_icon="🎁", layout="wide")

CAT_EMOJI = {"books": "📚", "music_vinyl": "🎷", "outdoor_fishing": "🎣", "kitchen_coffee": "☕",
             "home_decor": "🪴", "self_care": "🧴", "games_puzzles": "🧩", "kids_toys": "🧸",
             "tech_gadgets": "🔌", "experiences_giftcards": "🎟️", "stationery": "🖋️",
             "pets": "🐾"}


@st.cache_resource
def settings():
    # OCTANK_APP_RUN=demo -> the facilitator demo endpoint (base + instructor adapters)
    return config.get_settings(os.environ.get("OCTANK_APP_RUN", "attendee"))


@st.cache_data
def load_catalog():
    return catalog.load_clean()


@st.cache_data
def load_scenarios():
    """App presets + held-out benchmark + Extreme scenarios, each with its difficulty tier at the
    app's 80 candidates (data/scenario_grades.json, from scripts/build_extreme_scenarios.py).
    Missing grades are computed on the fly."""
    d = config.DATA_DIR
    gfile = d / "scenario_grades.json"
    grades = json.loads(gfile.read_text()) if gfile.exists() else {}
    out, seen = [], set()
    for f in (d / "app_scenarios.jsonl", d / "benchmark_cases.jsonl", d / "extreme_cases.jsonl"):
        if not f.exists():
            continue
        for sc in map(json.loads, open(f)):
            r = sc["request"]
            if r["request_id"] in seen:
                continue
            seen.add(r["request_id"])
            g = (grades.get(r["request_id"]) or {}).get("80")
            if g is None:
                g = difficulty.grade(r, pools.build_pool_80(r, load_catalog()))["score"]
            out.append({"request": r, "tier": difficulty.tier(g)})
    return out


S = settings()
ST = state.load(S)
cat = load_catalog()
scenarios = load_scenarios()

# ---- sidebar: endpoint + models ----------------------------------------------------------------
with st.sidebar:
    st.header("Endpoint")
    endpoint = st.text_input("Endpoint name", ST.get("endpoint", "octank-concierge"))
    # state "ics" keys -> labels, in display order. Attendee endpoint: instructor_* (Lab 4, from the
    # repo's adapters/) + sft / sft_rlvr (your Lab 3 jobs). Demo endpoint: v2-* / att-v2-*.
    LABELS = {"instructor_sft": "Instructor SFT (20K)", "v2-sft": "Instructor SFT (20K)",
              "instructor_sft_rlvr": "Instructor SFT -> RLVR (20K)", "v2-sft-rlvr": "Instructor SFT -> RLVR (20K)",
              "sft": "Your SFT (1K)", "sft_rlvr": "Your SFT -> RLVR (1K)",
              "att-v2-sft": "Attendee-scale SFT (1K)", "att-v2-rlvr": "Attendee-scale SFT -> RLVR (1K)"}
    ics = ST.get("ics", {})
    models = {"Base model": ST.get("base_ic", f"{endpoint}-base-ic")}
    for key in sorted(ics, key=lambda k: list(LABELS).index(k) if k in LABELS else len(LABELS)):
        models[LABELS.get(key, key)] = ics[key]
    chosen = st.multiselect("Compare", list(models), default=list(models)[:3])
    # Your own Lab 3 jobs that aren't deployed yet: show where they are instead of a silent gap.
    pending = {k: a for k, a in ST.get("arms", {}).items() if k not in ics}
    if pending:
        lines = []
        for k, a in pending.items():
            where = ("trained - re-run Lab 4's deploy cell to add it" if a.get("model_package")
                     else a.get("secondary") or a.get("job_status") or a.get("phase") or "not started")
            lines.append(f"- **{LABELS.get(k, k)}**: {where}")
        st.caption("Your models (training in the background):\n" + "\n".join(lines))
    st.divider()
    use_judge = st.toggle("Live LLM-as-a-Judge score", value=True,
                          help=f"Scores each answer with {config.JUDGE_MODEL} on the four evaluation metrics")
    # Always 80 candidates: the same ~6 real matches among twice the traps. The models were trained
    # on 40; this crowded setting is where SFT -> RLVR beats SFT most (hard: +7 pts, Extreme: +15).
    POOL = 80
    st.session_state["pool_size"] = POOL                 # read by the Catalog page
    # prompt + answer must fit the endpoint's 8,192-token context: an 80-candidate prompt is ~4,700
    MAX_ANSWER = 3072
    max_tokens = st.slider("Max new tokens", 512, MAX_ANSWER, MAX_ANSWER, 256,
                           help="Capped at 3,072: an 80-candidate prompt is ~4,700 tokens of the 8,192 context")
    temperature = st.slider("Temperature", 0.0, 1.2, 0.6, 0.1)
    st.caption(f"Region {S.region}")

st.title("🎁 Octank Gift Concierge")
st.caption(f"Tell us who you're shopping for. Each model picks from the same {POOL} candidates - "
           "including traps: items over budget, items you said to avoid, near-duplicates.")

# ---- request form -------------------------------------------------------------------------------
if "req" not in st.session_state:
    st.session_state.req = {"relationship": "dad", "age_band": "50-69", "occasion": "retirement",
                            "budget_total": 75.0, "num_items": 2,
                            "interests": ["jazz", "fly fishing"], "avoid": ["gadget"],
                            "notes": "says he has everything",
                            "message": "My dad just retired - he loves jazz and fly fishing but "
                                       "says he has everything. Two gifts, $75 total, no gadgets."}
if scenarios:
    import random
    _by = {lvl: [x for x in scenarios if x["tier"] == lvl]
           for lvl in ("Easy", "Medium", "Hard", "Extreme")}
    b0, b1, b2, b3, b4 = st.columns(5)
    _pick = None
    if b0.button("🎲 Random scenario", width="stretch",
                 help=f"Any of {len(scenarios)} held-out scenarios"):
        _pick = scenarios
    for col, lvl, icon in ((b1, "Easy", "🟢"), (b2, "Medium", "🟠"), (b3, "Hard", "🔴"), (b4, "Extreme", "🟣")):
        if col.button(f"{icon} Random {lvl.lower()}", width="stretch", disabled=not _by[lvl],
                      help=f"{len(_by[lvl])} {lvl.lower()} scenarios"):
            _pick = _by[lvl]
    if _pick:
        st.session_state.req = dict(random.choice(_pick)["request"])

r = st.session_state.req
c1, c2, c3, c4 = st.columns(4)
rel = c1.selectbox("Recipient", RELATIONSHIPS,
                   index=RELATIONSHIPS.index(r["relationship"]) if r["relationship"] in RELATIONSHIPS else 0)
occ = c2.selectbox("Occasion", OCCASIONS,
                   index=OCCASIONS.index(r["occasion"]) if r.get("occasion") in OCCASIONS else 0)
budget = c3.number_input("Total budget ($)", 10.0, 1000.0, float(r["budget_total"]), 5.0)
n_items = c4.number_input("Number of gifts", 1, 3, int(r.get("num_items") or 1))
c5, c6 = st.columns(2)
interests = c5.multiselect("Interests", INTERESTS, default=[i for i in r["interests"] if i in INTERESTS])
avoid = c6.multiselect("Avoid", ATTRIBUTES, default=[a for a in r["avoid"] if a in ATTRIBUTES])
message = st.text_area("Message to the concierge", r["message"], height=80)
notes = st.text_input("Notes", r.get("notes") or "")

req = {"request_id": f"app-{abs(hash((message, rel, occ, budget, n_items))) % 10**8}",
       "relationship": rel, "age_band": r.get("age_band"), "occasion": occ,
       "budget_total": float(budget), "num_items": int(n_items), "interests": interests,
       "avoid": avoid, "notes": notes, "message": message}
# Hidden needs are known only for benchmark scenarios (unchanged message); for a typed request the
# model must infer them and the deterministic check can't verify them.
same_msg = message == r.get("message")
req["hidden_avoid"] = list(r.get("hidden_avoid") or []) if same_msg else []
req["situation"] = r.get("situation", "") if same_msg else ""
if req["hidden_avoid"]:
    st.caption("🔒 Implied by the shopper's situation (hidden from the models): "
               f"avoid {', '.join(req['hidden_avoid'])}")


pool_now = pools.build_pool_80(req, cat)
diff = difficulty.grade(req, pool_now)
st.session_state.req_now, st.session_state.pool_ids = req, [c["id"] for c in pool_now]   # for the catalog page
diff["tier"] = difficulty.tier(diff["score"])      # Easy / Medium / Hard (55-69) / Extreme (70+)
_color = {"Easy": "green", "Medium": "orange", "Hard": "red", "Extreme": "violet"}[diff["tier"]]
dc1, dc2 = st.columns([1, 3])
dc1.markdown(f"**Request difficulty:** :{_color}[**{diff['tier']}**] ({diff['score']}/100)")
dc2.progress(diff["score"] / 100)
st.page_link("pages/1_Catalog.py", label=f"Browse the catalog and this request's {len(pool_now)} candidates",
             icon="📚")
with st.expander("What makes this request hard?"):
    st.markdown("\n".join(f"- {md(n)}" for n in diff["notes"]))
    st.dataframe(pd.DataFrame({"factor": list(diff["points"]), "points": list(diff["points"].values())}),
                 hide_index=True, width="stretch")
    st.caption("Graded from the authoritative tags (the models don't see them). On the benchmark, rule "
               "pass rates fall from ~97% on typical requests to 72-87% on Extreme ones, and models think longer."
               + (" For a typed request any hidden need is unknown, so it isn't counted."
                  if not req.get("hidden_avoid") else ""))


HARD_RULES = {"ids_in_candidates": "picks an item not in the list", "item_count_ok": "wrong number of items",
              "unique_ids": "duplicate items", "within_budget": "over budget",
              "no_avoid": "picks an item the shopper wanted to avoid"}
QUALITY = ["RecipientFit", "ReasoningQuality", "PickVariety"]


def judge_one(res, prompt, pool):
    """The four evaluation metrics (octank.metrics, Claude Haiku 4.5 with the real constraints) -
    the same judge as the fast evaluation - gated by the deterministic checks:
    - no parsable final answer (cut off / malformed) -> 0.0, judge not called
    - final answer breaks a hard rule -> score capped at 0.2"""
    ck = res["checks"]
    if not ck["json_valid"]:
        ntok = (res.get("usage") or {}).get("completion_tokens")
        why = (f"hit the {ntok}-token limit before writing an answer" if ntok == max_tokens
               else "no parsable JSON answer")
        return {"score": 0.0, "gated": True,
                "rationale": f"No final answer: the model {why}. Not sent to the judge - "
                             "reasoning alone doesn't count as a recommendation."}
    try:
        j = judge_many([{"prompt": prompt, "response": res["text"],
                         "ground_truth": formats.constraints_json(req, pool)}], S.region, workers=4)[0]
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:120]}
    vals = [j.get(m) for m in QUALITY if j.get(m) is not None]
    out = {"score": sum(vals) / len(vals) if vals else 0.0, "metrics": {m: j.get(m) for m in ["ConstraintAdherence", *QUALITY]},
           "rationale": " ".join(f"**{m}:** {j.get(m + '_why', '')}" for m in QUALITY if j.get(m + "_why"))}
    broken = [msg for k, msg in HARD_RULES.items() if not ck.get(k)]
    if broken and out["score"] > 0.2:
        out["judge_score_raw"], out["score"], out["gated"] = out["score"], 0.2, True
        out["rationale"] = (f"Capped at 0.20: the answer {', '.join(broken)} (checked deterministically). "
                            f"Judge's raw score was {out['judge_score_raw']:.2f}. " + out["rationale"])
    return out


def stats_line(res, seconds):
    """'done in 24s · 584 tokens · 24 tok/s' - shown while live and kept on the final results."""
    ntok = (res.get("usage") or {}).get("completion_tokens")
    if not ntok:                                        # fall back to a word count if usage is missing
        ntok_txt = f"~{len((res.get('text') or '').split())} words"
        rate = ""
    else:
        ntok_txt = f"{ntok} tokens"
        rate = f" · {ntok / max(seconds, 0.1):.0f} tok/s"
    return f"⏱ done in {seconds:.0f}s · {ntok_txt}{rate}"


clicked = st.button("Find gifts", type="primary", disabled=not (interests and chosen))
# ONE slot for both live progress and final results: each run replaces the previous one in place
# (separate slots left stale elements from the previous run drawn into the wrong columns).
area = st.empty()

@st.cache_resource
def _active_run() -> dict:
    """Process-wide: the endpoint serves one stream at a time, so only one app run may generate.
    A new run cancels the previous one (closing its stream frees the endpoint immediately)."""
    return {"cancel": None}


if clicked:
    import threading
    _prev = _active_run()["cancel"]
    if _prev is not None and not _prev.is_set():
        _prev.set()
        st.toast("Cancelled the previous run that was still generating", icon="⏹️")
    cancel = threading.Event()
    _active_run()["cancel"] = cancel
    pool = pool_now
    prompt = build_prompt(req, pool)
    by_id = {c["id"]: c for c in pool}
    # Serial on purpose: the endpoint streams one request at a time anyway, so each model gets the
    # GPU to itself and its timer measures its own generation time (start -> finish).
    live = {n: {"thinking": [], "answer": [], "t0": None, "done": False} for n in chosen}
    judges = []

    def run_one(name):
        L = live[name]
        L["t0"] = time.time()
        try:
            out = serving.chat(endpoint, models[name], prompt, S.region, max_tokens=max_tokens,
                               temperature=temperature, cancel=cancel,
                               on_token=lambda kind, tok: L[kind].append(tok))
            L["result"] = {"name": name, **out, "checks": checks.check_text(out["text"], req, pool)}
        except Exception as e:  # noqa: BLE001
            L["result"] = {"name": name, "error": f"{type(e).__name__}: {e}"}
        L["t_end"] = time.time()
        if use_judge and "error" not in L["result"] and not cancel.is_set():   # judge in background
            def _j(res=L["result"]):
                res["judge"] = judge_one(res, prompt, pool)
            jt = threading.Thread(target=_j, daemon=True)
            jt.start()
            judges.append(jt)
        L["done"] = True

    def run_all():
        for n in chosen:
            if cancel.is_set():                     # a newer run took over: don't start more models
                live[n].update(t0=time.time(), t_end=time.time(), done=True,
                               result={"name": n, "error": "Cancelled: a newer run started"})
                continue
            run_one(n)

    threading.Thread(target=run_all, daemon=True).start()
    box = area
    while not all(live[n]["done"] for n in chosen) or any(j.is_alive() for j in judges):
        with box.container():
            cols = st.columns(len(chosen))
            for col, n in zip(cols, chosen):
                L = live[n]
                text = "".join(L["thinking"]) + "".join(L["answer"])
                with col:
                    st.markdown(f"**{n}**")
                    if L["done"]:
                        judging = use_judge and "judge" not in L["result"] and "error" not in L["result"]
                        st.success(stats_line(L["result"], L["t_end"] - L["t0"])
                                   + (" · judging..." if judging else ""))
                    elif L["t0"] is None:
                        st.warning("queued - waits for the model before it")
                    else:
                        ntok = len(text.split())
                        st.info(f"generating... {time.time() - L['t0']:.0f}s, ~{ntok} words")
                        st.caption(text[-600:] or "(waiting for first token)")
        time.sleep(1.0)
    results = [live[n]["result"] for n in chosen]
    for r_ in results:          # report each model's own generation time
        n_ = r_["name"]
        r_["latency_s"] = round(live[n_]["t_end"] - live[n_]["t0"], 1)
    st.session_state.results = (results, by_id, req)
    st.session_state.setdefault("history", []).extend(
        {"request_id": req["request_id"], "difficulty": diff["label"], "model": x["name"],
         "tokens": (x.get("usage") or {}).get("completion_tokens"), "seconds": x.get("latency_s"),
         "hard_pass": bool((x.get("checks") or {}).get("hard_pass"))}
        for x in results if "error" not in x)
    with open(S.local("app_requests.jsonl"), "a") as f:
        f.write(json.dumps({"request": req, "results": [{k: v for k, v in x.items()
                                                         if k != "checks"} for x in results]},
                           default=str) + "\n")

if "results" in st.session_state:
    with area.container():
        results, by_id, shown_req = st.session_state.results
        ok = [x for x in results if "error" not in x and (x.get("usage") or {}).get("completion_tokens")]
        if len(ok) >= 2:
            names = [x["name"] for x in ok]
            ref = st.radio("Compare token use against", names, horizontal=True, key="ref_model",
                           index=next((i for i, nm in enumerate(names) if nm.startswith("Instructor SFT (")), 0))
            R = next(x for x in ok if x["name"] == ref)
            rt, rs = R["usage"]["completion_tokens"], R["latency_s"]
            rows = []
            for x in ok:
                t_, s_ = x["usage"]["completion_tokens"], x["latency_s"]
                rows.append({"model": x["name"], "tokens": t_, "seconds": s_,
                             f"tokens vs {ref}": "-" if x is R else f"{100 * (t_ - rt) / rt:+.0f}%",
                             f"time vs {ref}": "-" if x is R else f"{s_ - rs:+.0f}s",
                             "all rules pass": "✅" if x["checks"]["hard_pass"] else "❌"})
            st.markdown("#### Token use on this request")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            hist = pd.DataFrame(st.session_state.get("history", [])).dropna(subset=["tokens"])
            if not hist.empty and hist.request_id.nunique() >= 2 and ref in set(hist.model):
                wide = hist.pivot_table(index="request_id", columns="model", values="tokens")
                tally = []
                for m in [c for c in wide.columns if c != ref]:
                    pair = wide[[ref, m]].dropna()
                    if len(pair):
                        tally.append(f"**{m}**: {100 * (pair[m].sum() - pair[ref].sum()) / pair[ref].sum():+.0f}% "
                                     f"tokens vs {ref} over {len(pair)} requests")
                if tally:
                    st.caption("This session: " + " · ".join(tally) +
                               ". Single requests vary a lot (sampling at t=0.6); the session total is the better estimate.")
        cols = st.columns(len(results))
        for col, res in zip(cols, results):
            with col:
                st.subheader(res["name"])
                if "error" not in res:
                    st.success(stats_line(res, res["latency_s"]))
                if "error" in res:
                    st.error(res["error"])
                    continue
                ck = res["checks"]
                badge = lambda ok, label: f"{'✅' if ok else '❌'} {label}"  # noqa: E731
                st.markdown(" · ".join([
                    badge(ck["json_valid"], "valid"), badge(ck["within_budget"], "budget"),
                    badge(ck["no_avoid"], "avoid list"), badge(ck["item_count_ok"], "item count")]))
                j = res.get("judge") or {}
                if "score" in j:
                    st.metric("Judge score (fit, reasoning, variety)", f"{float(j['score']):.2f}",
                              delta="gated by rule checks" if j.get("gated") else None,
                              delta_color="off")
                    with st.expander("Why this score (judge reasoning)"):
                        if j.get("metrics"):
                            st.markdown(" · ".join(f"**{k}** {v:.2f}" for k, v in j["metrics"].items()
                                                   if v is not None))
                        st.markdown(md(j.get("rationale") or "(no rationale returned)"))
                ans = checks.parse_answer(res["text"]) or {}
                total = ck.get("price_total")
                for p in ans.get("picks", []):
                    item = by_id.get(str(p.get("id")))
                    if not item:
                        st.warning(f"Unknown item {p.get('id')}")
                        continue
                    with st.container(border=True):
                        st.markdown(f"### {CAT_EMOJI.get(item['category'], '🎁')} {md(item['name'])}")
                        st.markdown(md(f"**${item['price']:.2f}** · {item['category']}"))
                        st.caption(md(item["description"]))
                        st.markdown(md(p.get("why", "")))
                        bad = set(item["attributes"]) & set(shown_req["avoid"])
                        if bad:
                            st.error(f"Has avoided attribute: {', '.join(bad)}")
                if total is not None:
                    st.markdown(md(f"**Total ${total:.2f}** of ${shown_req['budget_total']:.2f}"))
                if ans.get("gift_message"):
                    st.info(md(f"💌 {ans['gift_message']}"))
                thinking = res.get("thinking") or checks.split_thinking(res["text"])[0]
                ntok = (res.get("usage") or {}).get("completion_tokens", "?")
                cut = not ck["json_valid"]
                if ntok == max_tokens:
                    st.warning(f"Hit the {max_tokens}-token limit before finishing")
                with st.expander(f"How it reasoned ({ntok} tokens, {res['latency_s']}s)"
                                 + (" - cut off, no final answer" if cut else ""), expanded=cut):
                    if cut and thinking:
                        st.caption("The model was still reasoning when it stopped. Everything it "
                                   "produced is shown below.")
                    st.text(thinking or "(no reasoning returned)")
                fb = st.feedback("thumbs", key=f"fb-{res['name']}")
                if fb is not None:
                    with open(S.local("app_feedback.jsonl"), "a") as f:
                        f.write(json.dumps({"request_id": shown_req["request_id"],
                                            "model": res["name"], "thumbs_up": bool(fb)}) + "\n")

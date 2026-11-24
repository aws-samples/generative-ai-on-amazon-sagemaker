"""Final evaluation of every model on the demo endpoint, in ONE pass on the same cases.

Models : base (generated once, inside the instructor SFT pipeline), instructor SFT, instructor
         SFT -> RLVR, attendee SFT, attendee SFT -> RLVR (model packages from work/<run>/state.json)
Tracks : octank.benchmarks - typical, hard (known / new products) at 40 and 80 candidates, Extreme.
Scoring: fast evaluation (managed generation stopped after inference; rules + tokens in code;
         Haiku 4.5 with the real ground truth; answers with no JSON answer score 0).
Also   : a live check - one Extreme scenario through every inference component on octank-demo.
Outputs: work/final_eval/{per_case.csv, summary.csv, paired.json, live.json} and
         data/reference_results.csv (base, instructor and attendee-scale models, used by Lab 5). Resumable.

    nohup ~/.venvs/gift_concierge/bin/python scripts/final_eval.py > work/final_eval/run.out 2>&1 &
"""
import json
import sys
import time
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402

from octank import benchmarks, catalog, checks, config, fasteval, pools, serving, state, workflow  # noqa: E402
from octank.prompts import build_prompt  # noqa: E402

S = config.get_settings("final_eval")
WORK = S.local("x").parent
STATE = WORK / "state.json"
ORDER = ["base", "instructor_sft", "instructor_sft_rlvr", "att_sft", "att_sft_rlvr"]
LABEL = {"base": "Base", "instructor_sft": "Instructor SFT", "instructor_sft_rlvr": "Instructor SFT → RLVR",
         "att_sft": "Attendee SFT", "att_sft_rlvr": "Attendee SFT → RLVR"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def st():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(**kv):
    STATE.write_text(json.dumps({**st(), **kv}, indent=2, default=str))


def models():
    out = {}
    for run, prefix in (("instructor", "instructor"), ("attendee", "att")):
        a = state.load(config.get_settings(run)).get("arms", {})
        for arm in ("sft", "sft_rlvr"):
            out[f"{prefix}_{arm}"] = a[arm]["model_package"]
    return out


def paired(df, a, b, tracks):
    sub = df[df.track.isin(tracks) & df.model.isin([a, b])]
    if sub.empty or {a, b} - set(sub.model):
        return None
    p = sub.pivot_table(index=["track", "case"], columns="model", values="hard_pass").dropna()
    x, y = p[a].astype(bool), p[b].astype(bool)
    up, dn = int((~x & y).sum()), int((x & ~y).sum())
    n = up + dn
    pv = min(1.0, 2 * sum(comb(n, k) for k in range(min(up, dn) + 1)) / 2 ** n) if n else 1.0
    t = sub.groupby("model").tokens.mean()
    return {"n": len(p), "a": round(x.mean(), 4), "b": round(y.mean(), 4), "fixed": up, "broke": dn,
            "p": pv, "tokens_a": round(t[a]), "tokens_b": round(t[b])}


def live_check():
    """One Extreme scenario (80 candidates) through every IC on octank-demo."""
    D = config.get_settings("demo")
    DS = state.load(D)
    ics = {"base": DS["base_ic"], **DS.get("ics", {})}
    c = json.loads(open(config.DATA_DIR / "extreme_cases.jsonl").readline())
    pool = pools.build_pool_80(c["request"], catalog.load_clean())
    out = {}
    for key, ic in ics.items():
        r = serving.chat(DS["endpoint"], ic, build_prompt(c["request"], pool), D.region,
                         max_tokens=config.EVAL_MAX_NEW_TOKENS_80)
        ck = checks.check_text(r["text"], c["request"], pool)
        out[key] = {"ic": ic, "seconds": r["latency_s"], "tokens": (r.get("usage") or {}).get("completion_tokens"),
                    "answered": ck["json_valid"], "hard_pass": ck["hard_pass"]}
        log(f"live {key:16s} {out[key]}")
    (WORK / "live.json").write_text(json.dumps(out, indent=2))
    return out


REF_NAMES = {"base": "base", "instructor_sft": "instructor_sft", "instructor_sft_rlvr": "instructor_sft_rlvr",
             "att_sft": "attendee_scale_sft", "att_sft_rlvr": "attendee_scale_sft_rlvr"}


def write_reference(df):
    """data/reference_results.csv for Lab 5: every demo model per case (judge explanations dropped)."""
    ref = df[df.model.isin(list(REF_NAMES))].replace({"model": REF_NAMES, "track": {"natural": "typical"}})
    ref = ref[[c for c in ref.columns if not c.endswith("_why")]]
    ref.to_csv(config.DATA_DIR / "reference_results.csv", index=False)
    return ref


def run():
    WORK.mkdir(parents=True, exist_ok=True)
    if "execs" not in st():
        save(models=models(), execs=workflow.launch_eval(S, models(), "final", base_with="instructor_sft", log=log))
    if not (WORK / "per_case.csv").exists():
        workflow.finish_eval(S, st()["execs"], log=log).to_csv(WORK / "per_case.csv", index=False)
    df = pd.read_csv(WORK / "per_case.csv")
    fasteval.summarize(df).to_csv(WORK / "summary.csv")
    pr = {}
    for a, b in (("instructor_sft", "instructor_sft_rlvr"), ("att_sft", "att_sft_rlvr"), ("att_sft", "instructor_sft"),
                 ("base", "instructor_sft")):
        pr[f"{a}->{b}"] = {"typical": paired(df, a, b, ["typical"]), "hard": paired(df, a, b, benchmarks.HARD),
                           "extreme": paired(df, a, b, ["extreme_80"])}
    (WORK / "paired.json").write_text(json.dumps(pr, indent=2))
    write_reference(df)
    if not (WORK / "live.json").exists():
        live_check()
    log("final eval done\n" + workflow.by_problem_type(df).round(3).to_string(index=False))
    return df, pr, json.loads((WORK / "live.json").read_text())


if __name__ == "__main__":
    run()

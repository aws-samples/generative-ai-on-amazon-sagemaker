"""Generate the workshop notebooks (instructor + attendee lifecycle) from cell definitions.

    python scripts/build_notebooks.py

Notebooks are generated so instructor and attendee stay in lock-step on the shared `octank` package.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


def write(name, cells):
    nb = nbf.v4.new_notebook()
    for i, c in enumerate(cells):            # stable cell ids: regenerating gives no-op diffs
        c["id"] = f"{Path(name).stem}-{i:03d}"
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3 (ipykernel)",
                                     "language": "python"},
                      "language_info": {"name": "python"}}
    path = NB / name
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print("wrote", path.relative_to(ROOT))


INSTALL = code("""
# Pinned SDKs (latest as of 2026-10-28). Restart the kernel after the first install.
%pip install -q -r ../requirements.txt
""")


def boot(run_name):
    return code(f"""
import os, sys, json
sys.path.insert(0, os.path.abspath("../src"))
%load_ext autoreload
%autoreload 2
import pandas as pd
pd.set_option("display.max_colwidth", 120)
from octank import config, state
S = config.get_settings("{run_name}")
ST = state.load(S)
print(f"region={{S.region}}  bucket={{S.bucket}}  prefix={{S.prefix}}  run={{S.run_name}}")
print(f"role={{S.role_arn}}")
""")


SKILL_NOTE = """
> **Agent skill:** this step follows the `aws-ai-ml` skill's **{ref}** workflow. You can ask the
> SageMaker AI agent: *"{prompt}"* and it generates the equivalent cells.
"""

# =============================================================================================
# ATTENDEE 00 - setup
# =============================================================================================
write("00_setup.ipynb", [
    md("""
# Lab 0 - Setup

**Octank Gift Concierge** - you'll take a small gift-shopping model from raw data to a deployed
custom model: generate data, profile it, clean it, label it, review edge cases, fill coverage
gaps, prepare SFT and RLVR datasets, train **SFT, then SFT -> RLVR** with **serverless model
customization**, evaluate with **LLM-as-a-Judge**, and serve it behind a Streamlit demo.

| Lab | Notebook |
|---|---|
| 0 | Setup (this notebook) - and start your endpoint early |
| 1 | `01_generate_profile.ipynb` - generate ~1,000 raw requests, profile them |
| 2 | `02_curate.ipynb` - dedup, toxicity, auto-label, human review, gap-fill |
| 3 | `03_prepare_train.ipynb` - build SFT + RLVR datasets, review the reward function, train SFT then SFT -> RLVR |
| 4 | `04_evaluate_deploy.ipynb` - deploy adapters, live checks, Streamlit demo |
| 5 | `05_compare.ipynb` - compare your models with the instructor's, by problem type |

Everything runs in **your current region** and writes to the SageMaker default bucket under
`s3://sagemaker-<region>-<account>/gift-concierge/attendee/`.
"""),
    INSTALL,
    boot("attendee"),
    md("## Check model access\nEvery model that touches training data is open-weight on Bedrock "
       "(no deployment needed). Claude Haiku 4.5 is used only as the evaluation judge."),
    code("""
from octank import llm
checks = {
    "generator  (gpt-oss-120b)": config.GENERATOR_MODEL,
    "labeler    (gpt-oss-20b)": config.LABELER_MODEL,
    "safety     (gpt-oss-safeguard-20b)": config.SAFETY_MODEL,
    "reward taste (gpt-oss-20b)": config.TASTE_MODEL,
    "eval judge (Claude Haiku 4.5)": config.judge_profile_id(S.region),
}
for name, mid in checks.items():
    try:
        r = llm.converse(mid, "Reply with the single word: ready", max_tokens=200, temperature=0,
                         region=S.region)
        print(f"OK   {name:38s} -> {r.text[:30]!r}")
    except Exception as e:
        print(f"FAIL {name:38s} -> {type(e).__name__}: {str(e)[:120]}")
"""),
    md("""
## Start your endpoint now (it takes a while)

GPU capacity can take time, so we create the `ml.g7.2xlarge` endpoint with the **base model**
now; Lab 4 attaches the instructor's fine-tuned adapters (and yours, if they've finished). This
uses **sagemaker-core** resources
(`Model`, `EndpointConfig`, `Endpoint`, `InferenceComponent`). It is billed hourly until you run
the cleanup cell at the end of Lab 4.
"""),
    code("""
from octank import serving
ENDPOINT = ST.get("endpoint") or "octank-concierge"
state.update(S, endpoint=ENDPOINT, base_ic=f"{ENDPOINT}-base-ic")

print(f"Creating {ENDPOINT} - this cell runs until the endpoint and base model are InService.")
try:
    serving.create_endpoint(ENDPOINT, S.role_arn, S.region)
    out = serving.add_model(ENDPOINT, S.role_arn, S.region)
    state.update(S, endpoint_status="InService", **out)
    print("endpoint ready:", out)
except Exception as e:
    state.update(S, endpoint_status=f"FAILED: {e}")
    print("endpoint failed:", e)
    raise
"""),
])

# =============================================================================================
# ATTENDEE 01 - generate + profile
# =============================================================================================
write("01_generate_profile.ipynb", [
    md("""
# Lab 1 - Generate and profile raw data

1. Load the **Octank product catalog** (~1,200 products, seed data shipped in the repo) and profile it.
2. **Generate ~1,000 raw shopper requests** with `gpt-oss-120b` on Bedrock - the exact technique
   the workshop team used for the 20,000-row instructor dataset.
3. **Profile** them: completeness, distributions, skew, coverage gaps.

The generator deliberately plants data-quality problems (duplicates, missing fields, toxic and
ambiguous requests, contradictions, skew) so Lab 2 has real issues to find and fix.
"""),
    boot("attendee"),
    md("## 1. The catalog"),
    code("""
from octank import catalog
raw_cat = catalog.load_raw()
print(json.dumps(catalog.profile(raw_cat), indent=1, default=str))
raw_cat.sample(5, random_state=1)
"""),
    md("""
The catalog has defects: inconsistent category spellings, duplicate SKUs, items with no interest
tags. Cleaning is deterministic (canonical categories, de-dupe SKUs, back-fill tags from the
description), so everyone gets the same clean catalog.
"""),
    code("""
cat = catalog.clean(raw_cat)
print(len(raw_cat), "->", len(cat), "items")
print(json.dumps(catalog.profile(cat), indent=1, default=str))
"""),
    md("""
## 2. Generate raw requests

**Technique:** sample a structured *spec* per request (relationship, age, occasion, budget,
interests, things to avoid, a note) from configurable distributions, then have `gpt-oss-120b`
write the free-text message a real shopper would type, in varied voices. About 45% of specs also
get a **situation** ("recently got sober", "tiny studio apartment") that the message mentions in
passing - it implies something to avoid that the form never says. Finally, plant flaws.
Generation is **resumable** - re-running reuses cached messages.
"""),
    code("""
from octank import generate
from octank.pipeline import ATTENDEE, Run
run = Run(S, ATTENDEE)
spec_example = generate.sample_spec(__import__("random").Random(0), 0)
print("example spec:", json.dumps(spec_example, indent=1))
"""),
    code("""
%%time
raw = run.generate_raw()          # ~1,000 rows, a few minutes
print(len(raw), "rows")
raw[["request_id", "message", "occasion", "budget_total", "num_items", "interests", "avoid",
     "situation", "hidden_avoid"]].head(8)
"""),
    md("""
## 3. Profile

The hidden `_flaw` column is the **answer key** - we use it only to score curation at the end of
Lab 2. A real dataset doesn't have one, so we profile without looking at it.
"""),
    code("""
from octank import profile
profile.completeness(raw)
"""),
    code("""
print("occasion skew:", profile.skew_report(raw, "occasion"))
print("text stats:", profile.text_stats(raw))
profile.plot_overview(raw, "Raw requests");
"""),
    code("""
gaps = profile.coverage_gaps(raw, min_share=ATTENDEE.gap_min_share)
print(f"{len(gaps)} under-covered segments (target: >= {ATTENDEE.gap_min_share:.0%} of rows each)")
gaps
"""),
    md("""
**What we learned:** ~60% of requests are Christmas or birthdays, several occasions and the
under-$25 budget band barely appear, and some form fields are missing. Next: clean it up.
"""),
    md(SKILL_NOTE.format(ref="dataset-evaluation",
                         prompt="Profile my raw requests file for completeness and skew")),
])

# =============================================================================================
# ATTENDEE 02 - curate
# =============================================================================================
write("02_curate.ipynb", [
    md("""
# Lab 2 - Curate: dedup, toxicity, auto-label, human review, gap-fill

| Step | Technique |
|---|---|
| Exact dedup | hash of normalized text |
| Near-dup | **MinHash + LSH** on word 3-grams (no embedding model) |
| Toxicity | **gpt-oss-safeguard-20b** + a plain-text policy you can edit |
| Auto-label | **gpt-oss-20b** extracts labels + confidence, calibrated on a human gold set |
| Human review | in-notebook review queue for low-confidence and edge cases |
| Gap-fill | generate new requests for under-covered segments (same technique as Lab 1) |
"""),
    boot("attendee"),
    code("""
from octank import catalog, dedup, safety, label, review, profile, gold
from octank.pipeline import ATTENDEE, Run
run = Run(S, ATTENDEE)
raw = run.load_raw()
cat = catalog.load_clean()
print(len(raw), "raw rows")
"""),
    md("""
## 1. De-duplication

**MinHash** turns each message into a 128-number signature; two signatures agree on a fraction of
positions roughly equal to the **Jaccard similarity** of the messages' word 3-grams. **LSH**
buckets signatures so we only compare likely matches. Try a few thresholds.
"""),
    code("""
exact = dedup.exact_duplicates(raw)
print("exact duplicates:", int(exact.sum()))
d1 = raw[~exact]
for t in (0.3, 0.5, 0.7, 0.9):
    print(f"near-dup threshold {t}: {len(dedup.near_duplicates(d1, threshold=t))} pairs")
"""),
    code("""
pairs = dedup.near_duplicates(d1, threshold=ATTENDEE.near_dup_threshold)
msg = d1.set_index("request_id")["message"]
for p in pairs.head(4).itertuples():
    print(f"J~{p.jaccard:.2f}\\n  keep: {msg[p.keep_id][:110]}\\n  drop: {msg[p.dup_id][:110]}\\n")
deduped = d1[~d1["request_id"].isin(pairs["dup_id"])]
print(len(raw), "->", len(deduped))
"""),
    md("""
## 2. Toxicity filter

`gpt-oss-safeguard-20b` is an open-weight **safety classifier**: you give it a *policy* in plain
language and it classifies content against it, with a rationale. Open the policy, read it, and
try editing it (e.g. add a rule) and re-running.
"""),
    code("""
policy = safety.load_policy()
print(policy[:1500])
"""),
    code("""
%%time
screened = run.safety(deduped, policy=policy)
print(screened["safety_decision"].value_counts())
screened[screened["safety_violation"] == 1][["message", "safety_category", "safety_confidence",
                                             "safety_rationale"]].head(10)
"""),
    code("""
kept = screened[screened["safety_decision"] != "drop"]
print(len(deduped), "->", len(kept))
"""),
    md("""
## 3. Auto-labeling - calibrate first, then label at scale

The free-text message is the source of truth; form fields are often missing or wrong.
`gpt-oss-20b` extracts structured labels and a **confidence**. Before trusting it at scale we
measure it on the **gold set** (draft labels from `gpt-oss-120b`, corrected by humans) and pick
the confidence threshold for auto-accept vs. human review.
"""),
    code("""
g = gold.load_gold()
print(len(g), "gold rows;", g["label_source"].value_counts().to_dict())
pred = label.label_df(g[["request_id", "message"]], cache=run.cache("gold_pred"),
                      region=S.region)
pred.columns = [c.replace("lbl_", "") for c in pred.columns]
print("field accuracy vs gold (%):", label.field_accuracy(pred, g))
label.calibration_curve(pred, g)
"""),
    code("""
REVIEW_THRESHOLD = 0.7     # <- pick from the curve above: accuracy vs. how much you send to humans
run.cfg.review_threshold = REVIEW_THRESHOLD
labeled = run.label(kept)
labeled["review_reason"].value_counts()
"""),
    md("""
## 4. Human review

Rows that need a person: unclear intent, form vs. text disagreements, no budget anywhere,
borderline safety and low confidence. Review **~15 items** (approve / edit / reject). Anything you
don't get to is resolved by the default policy, so the pipeline always completes.
"""),
    code("""
queue = labeled[labeled["review_reason"] != ""].sort_values("review_reason")
q = review.ReviewQueue(queue.head(15), save_path=run.path("review_decisions.parquet"))
q.show()
"""),
    code("""
human = q.decisions()
print(len(human), "human decisions")
curated, decisions = run.resolve(labeled, human=human)
curated = run.finalize(curated, cat)
print(len(labeled), "->", len(curated), "rows;", decisions["decision"].value_counts().to_dict())
"""),
    md("## 5. Did curation work? (scored against the hidden answer key)"),
    code("""
run.report(raw, curated)
"""),
    md("""
## 6. Synthetic gap-fill

Generate new requests for segments below 2% of the data, using the same generator - then send
them through the **same** safety, labeling and finalize steps.
"""),
    code("""
%%time
final, gaps = run.gapfill(curated, cat)
print(len(curated), "->", len(final), "rows")
before, after = profile.skew_report(curated), profile.skew_report(final)
print("occasion skew before:", before, "\\nafter: ", after)
profile.plot_overview(final, "After curation + gap-fill");
"""),
    code("""
final.to_parquet(run.path("final.parquet"), index=False)
state.update(S, final_rows=len(final))
"""),
])


# =============================================================================================
# ATTENDEE 03 - prepare + train
# =============================================================================================
write("03_prepare_train.ipynb", [
    md("""
# Lab 3 - Prepare the data, review the reward, train SFT then SFT -> RLVR

**The task:** each request gets **40 candidate products**, listed like a real catalog (name,
category, price, description - no tags). Only ~6 are real matches; the rest are traps: items just
over budget, pairs that break a multi-item total, strong interest matches that carry something the
shopper wants to avoid (or that their situation rules out), near-duplicates and filler.

**Two training stages, each with its own job:**

| Stage | Starts from | Data | Learning signal | What it adds |
|---|---|---|---|---|
| **SFT** | base model | ~700 requests **with** verified teacher answers | imitate the teacher | the task, the answer format, when to stop thinking |
| **SFT -> RLVR** | your SFT model | ~200 prompts, **no answers** | a reward function scores 8 sampled answers per prompt | better decisions on hard cases, fewer tokens |
"""),
    boot("attendee"),
    code("""
from octank import catalog, checks, formats, pools
from octank.prompts import build_prompt
from octank.pipeline import ATTENDEE, Run
run = Run(S, ATTENDEE)
cat = catalog.load_clean()
final = pd.read_parquet(run.path("final.parquet"))
sp = run.split(final)
pool_by_id = run.pools(sp, cat)
sp["split"].value_counts()
"""),
    code("""
ex = sp[(sp["split"] == "sft") & (sp["situation"] != "")].iloc[0].to_dict()
print("situation (never in the form):", ex["situation"], "-> avoid", list(ex["hidden_avoid"]), "\\n")
print(build_prompt(ex, pool_by_id[ex["request_id"]])[:2600])
"""),
    md("## 1. Teacher traces (rejection sampling)"),
    code("""
%%time
taught = run.teach(sp[sp["split"].isin(["sft", "holdout"])], pool_by_id)
print(f"teacher pass rate: {taught['ok'].mean():.1%} of {len(taught)}")
ok = taught[taught["ok"] == True].iloc[0]
print(ok["completion"][:2000])
"""),
    md("""
`gpt-oss-120b` solves each request with its reasoning visible; code checks every answer against
every rule, including needs implied by the situation, and keeps only passing answers whose
reasoning stays under ~1,100 tokens. That guarantees the training answers are *correct* - not that
they are the *best* - and SFT can't learn beyond what the teacher solved. That's what RLVR is for.
"""),
    md("## 2. Write the SFT, RLVR and evaluation files"),
    code("""
%%time
leftovers = set(sp.loc[sp["_flaw"].isin(["toxic", "ambiguous"]), "request_id"])
files = run.write_datasets(sp, pool_by_id, taught, exclude_ids=leftovers)
for k, v in files.items():
    print(f"{k:8s} {sum(1 for _ in open(v)):5d} rows  {v}")
print("RL prompts:", run.filter_stats["rl_prompts"])
"""),
    md("""
- `sft_train.jsonl`: prompt + verified teacher answer (reasoning + JSON).
- `rlvr_train.jsonl`: prompt + the **ground truth the reward function needs**: budget, item count,
  avoid list, needs implied by the situation, every candidate's price, category, attributes and
  interests, a difficulty score and the teacher's reference answer. 20% of the RL prompts are
  freshly generated **Extreme** requests (difficulty 70+), where SFT struggles most.
- `eval.jsonl`: evaluation rows for your own holdout.
"""),
    md("""
## 3. Validate with the agent skill's format detector

The `aws-ai-ml` skill ships `format_detector.py`, which recognizes and validates training and
evaluation formats. Run it on each file before training.
"""),
    code("""
import subprocess
FD = os.path.expanduser("~/.kiro/skills/aws-ai-ml/references/dataset-evaluation/scripts/format_detector.py")
assert os.path.exists(FD), (
    f"format_detector.py not found at {FD}. It ships with the aws-ai-ml agent skill: install the skill "
    "for the SageMaker AI agent (it lives under ~/.kiro/skills/aws-ai-ml/), then re-run this cell.")
for k in ["sft", "rlvr", "eval"]:
    r = subprocess.run([sys.executable, FD, files[k]], capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    print(f"--- {k}: {files[k]}")
    print("\\n".join(l for l in lines if l.startswith(("Format:", "Confidence:"))))
    if r.stderr.strip():
        print("stderr:", r.stderr.strip()[-1500:])
    if r.returncode != 0:
        raise RuntimeError(f"format_detector failed on {k} (exit {r.returncode})")
    valid = next((l for l in lines if l.startswith("Valid:")), None)
    print(valid or "Valid: (no 'Valid:' line in the output)")
    if not valid or "✓" not in valid:
        raise RuntimeError(f"{k} is not a valid {k} dataset:\\n" + r.stdout[-2000:])
"""),
    md(SKILL_NOTE.format(ref="dataset-evaluation",
                         prompt="Validate my SFT and RLVR JSONL files for Qwen3.5-4B")),
    code("""
uris = run.upload(files)
state.update(S, datasets=uris)
uris
"""),
    md("""
## 4. Review the reward function

RLVR samples 8 answers per prompt, scores each one with a **reward function**, and moves the model
toward the better answers. The reward is the most important design decision in the whole workflow:
the model will learn exactly what it pays for - including its loopholes.

Ours is a **hybrid** in `judge/reward_lambda.py`, a pyramid where higher layers only count once the
lower ones pass:

| Layer | Score |
|---|---|
| No parsable answer (cut off) | -1.0 |
| Bad ids / wrong count / duplicates | -0.6 |
| Picks an item the situation rules out (hidden need) | -0.4 |
| Picks an item the shopper said to avoid | -0.3 |
| Over budget | -0.1 |
| All rules pass | 0.30 + 0.30 x fit + 0.10 x variety + 0.30 x taste - length penalty |
| Anti-gaming | echo of the prompt -1.0, repetitive reasoning -0.4, empty "why" -0.25 |

- **Code** checks the rules exactly, and scores *fit* from the product tags the model never sees.
- **An LLM** (`gpt-oss-20b`) scores *taste* - fit, reasoning and the gift message against the
  teacher's answer - only for answers that pass every rule. If it fails, the score falls back to
  the code's fit score, never a garbage reward.
- The **length penalty** is free up to 450-900 tokens (more allowed for harder requests).

Read the function, then predict the score of each answer below before running the cell.
"""),
    code("""
import importlib.util

def load_reward(taste_weight="0"):
    # Load judge/reward_lambda.py. TASTE_WEIGHT=0 runs the deterministic part only (no Bedrock call).
    os.environ["TASTE_WEIGHT"] = taste_weight
    spec = importlib.util.spec_from_file_location("reward_lambda", "../judge/reward_lambda.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

RW = load_reward()
print(open("../judge/reward_lambda.py").read()[:2600])
"""),
    code("""
# A prompt where every test answer means what its label says: the teacher solved it (a reference
# answer exists), it asks for 2+ items (so dropping one is a wrong count), and at least one
# candidate carries an attribute the shopper must avoid (explicitly or through their situation).
def usable(row):
    gt = json.loads(row["reward_model"]["ground_truth"])
    avoid = set(gt["avoid"]) | set((gt.get("implied_by_situation") or {}).get("avoid", []))
    ref = gt.get("reference")
    return (ref is not None and ref.get("picks") and int(gt["num_items"]) >= 2
            and any(avoid & set(c["attributes"]) for c in gt["candidates"].values()))

row = next((r for r in map(json.loads, open(files["rlvr"])) if usable(r)), None)
if row is None:
    raise RuntimeError(f"No RLVR prompt in {files['rlvr']} has a teacher answer, 2+ items and an avoidable "
                       "candidate - check the dataset build (section 2).")
gt = json.loads(row["reward_model"]["ground_truth"])
cands, ref = gt["candidates"], gt["reference"]
ids = [p["id"] for p in ref["picks"]]
avoid = set(gt["avoid"]) | set((gt.get("implied_by_situation") or {}).get("avoid", []))
trap = next(i for i, c in cands.items() if avoid & set(c["attributes"]) and i not in ids)
print(f"prompt: {int(gt['num_items'])} items, budget ${gt['budget_total']:.2f}, avoid {sorted(avoid)}; "
      f"trap {trap} has {sorted(avoid & set(cands[trap]['attributes']))}")

def answer(pick_ids, why="This pick suits the recipient's interests and the occasion well."):
    return json.dumps({"picks": [{"id": i, "why": why} for i in pick_ids],
                       "total": round(sum(cands[i]["price"] for i in pick_ids), 2), "gift_message": "Enjoy!"})

think = "Checking the budget, the item count, the avoid list and what the situation implies."
tests = {
    "excellent (teacher answer)": f"<think>{think}</think>\\n\\n{json.dumps(ref)}",
    "picks a trap item": f"<think>{think}</think>\\n\\n{answer(ids[:-1] + [trap])}",
    "wrong item count": f"<think>{think}</think>\\n\\n{answer(ids[:-1])}",       # one item short
    "empty 'why' everywhere": f"<think>{think}</think>\\n\\n{answer(ids, why='good')}",
    "cut off, no answer": "<think>" + think * 3,
}
batch = [{"id": k, "messages": row["prompt"] + [{"role": "assistant", "content": v}],
          "reward_model": row["reward_model"]} for k, v in tests.items()]
for r in json.loads(RW.lambda_handler({"batch": batch}, None)["body"]):
    print(f"{r['id']:28s} {r['aggregate_reward_score']:+.2f}")
"""),
    md("""
**Try it:** change one rule in a copy of the function (for example drop the empty-"why" penalty or
the length penalty) and think about what a model would learn to do if you trained on it.

Locally the cell above runs the **deterministic** part only (`TASTE_WEIGHT=0`). The deployed Lambda
adds the LLM taste score, exactly as the instructor's models were trained.
"""),
    md("""
## 5. Launch your training: SFT, then SFT -> RLVR

Both stages run as **background jobs on serverless model customization** - no cluster to manage.
SFT trains in ~16 minutes; SFT -> RLVR starts when SFT finishes and takes ~40 minutes, plus any
wait for serverless capacity.

> **For the rest of the session you'll use the instructor's models.** They were trained with this
> exact notebook - same code, settings and reward - on 20x more data, before the session. Lab 4
> deploys them to your endpoint right away, and Lab 5 compares them with the base model and with
> models trained at your data size. Your own jobs keep running; if your SFT model finishes in time,
> Lab 4 adds it next to the instructor's.

Evaluation of your own models is **off by default** (it would finish after the session). Set
`EVALUATE = True` below to run the same fast evaluation the instructor used.
"""),
    code("""
from octank import arms, reward, train, workflow

sm_session = train.session()
MPG = "octank-concierge-attendee"
model_package_group = train.ensure_model_package_group(MPG, "Octank gift concierge (attendee)")
ds = state.load(S)["datasets"]
sft_dataset_arn = train.register_dataset("octank-attendee-sft", ds["sft"])

# The reward Lambda: same code and configuration as the instructor's (Python 3.12, 300 s, 1 GB,
# gpt-oss-20b taste calls). verify_matches() proves the deployed function is the reviewed file.
# Role: config.REWARD_ROLE_ARN (env OCTANK_REWARD_ROLE_ARN) if your account provides one; otherwise
# octank creates octank-hybrid-reward-lambda-role (needs IAM create permissions).
reward_lambda = reward.deploy_hybrid_lambda(S.region, role_arn=config.REWARD_ROLE_ARN)
reward.verify_matches(reward_lambda, S.region)
state.update(S, reward_lambda=reward_lambda)
print("reward Lambda:", reward_lambda)
"""),
    md("### 5a. Trainer recipes (hyperparameters live in `config.SFT_HP` and `config.RLVR_HP`)"),
    code("""
def make_sft(start_from):
    return workflow.sft_trainer(S, start_from, sft_dataset_arn, model_package_group, sm_session)

def make_rlvr(start_from):
    return workflow.rlvr_trainer(S, start_from, ds["rlvr"], MPG, sm_session, reward_lambda,
                                 "octank-attendee-rlvr")

print("SFT:", config.SFT_HP)
print("RLVR:", config.RLVR_HP)
"""),
    md("### 5b. Launch"),
    code("""
EVALUATE = False                                          # True: fast-evaluate each model after training
arms.start(S, "sft", make_sft, evaluate=EVALUATE)                       # base -> SFT
arms.start(S, "sft_rlvr", make_rlvr, after="sft", evaluate=EVALUATE)    # waits for SFT, then SFT -> RLVR
arms.status(S)
"""),
    code("""
# Re-run any time. If the kernel restarted, re-run the cells above: arms resume from saved state.
arms.status(S)
"""),
    md("""
> **Keep this kernel running until `sft_rlvr` shows `phase=training`.** The SFT -> RLVR launch waits
> for SFT in this kernel. If the kernel stops first, Lab 4 detects it and shows a one-line command
> to resume it from there.
"""),
    md(SKILL_NOTE.format(ref="finetuning",
                         prompt="Fine-tune Qwen3.5-4B with SFT, then continue from the SFT model with "
                                "RLVR using my reward Lambda")),
])

# =============================================================================================
# ATTENDEE 04 - deploy
# =============================================================================================
write("04_evaluate_deploy.ipynb", [
    md("""
# Lab 4 - Deploy and try the models

You deploy the **instructor's models** - the same pipeline you just ran, on 20x more data - next to
the base model on your `ml.g7.2xlarge` endpoint. Each LoRA adapter is its own **inference
component** sharing the base model's GPU (up to 4 adapters at once). Here you:
1. **Deploy** the instructor's SFT and SFT -> RLVR adapters (shipped in the repo, `adapters/`),
   and your own adapters if your jobs have finished.
2. **Check live outputs** with the deterministic rules (budget, avoid list, item count).
3. **Try them in the Streamlit demo.**

| Model | Trained on | Where it comes from |
|---|---|---|
| Instructor SFT (20K) | 20,000 raw requests | `adapters/v2-sft` |
| Instructor SFT -> RLVR (20K) | 20,000 raw requests | `adapters/v2-sft-rlvr` |
| Your SFT (1K) / Your SFT -> RLVR (1K) | your 1,000 raw requests | your Lab 3 jobs, when they finish |
"""),
    boot("attendee"),
    code("""
from octank import arms, checks, serving
arms.check_handoff(S)               # tells you if SFT -> RLVR needs resuming: arms.resume_rlvr(S)
arms.status(S)                      # your own jobs keep running in the background
"""),
    md("## 1. Deploy the instructor's adapters (and yours, if finished)"),
    code("""
ST = state.load(S)
ENDPOINT, BASE_IC = ST["endpoint"], ST["base_ic"]
if ST.get("endpoint_status") != "InService":
    print("waiting for endpoint ...")
    serving.create_endpoint(ENDPOINT, S.role_arn, S.region)
    serving.add_model(ENDPOINT, S.role_arn, S.region)

def deploy_ready():
    ics = dict(state.load(S).get("ics", {}))
    # 1. the instructor's adapters: copy from the repo to your bucket, attach as inference components
    for key, folder in serving.INSTRUCTOR_ADAPTERS.items():
        if key not in ics:
            uri = serving.upload_adapter(folder, S.s3("adapters", folder), S.region)
            ic = f"{ENDPOINT}-{key.replace('_', '-')}"
            serving.add_adapter(ENDPOINT, BASE_IC, ic, uri, S.region)
            ics[key] = ic
    # 2. your own adapters, once their training job has finished
    for arm, a in state.load(S).get("arms", {}).items():
        if a.get("model_package") and arm not in ics:
            ic = f"{ENDPOINT}-{arm.replace('_', '-')}"
            serving.add_adapter(ENDPOINT, BASE_IC, ic, serving.adapter_uri_from_job(a["job"], S.region), S.region)
            ics[arm] = ic
    state.update(S, ics=ics)
    return ics

ics = deploy_ready()        # re-run later to add your own models when they finish training
ics
"""),
    md("## 2. Deterministic checks on live outputs"),
    code("""
from concurrent.futures import ThreadPoolExecutor
from octank.prompts import build_prompt
cases = [json.loads(l) for l in open(config.DATA_DIR / "benchmark_cases.jsonl")][:12]
targets = state.load(S).get("ics", {})          # deployed adapters (the base model rarely finishes)

def score(args):
    name, ic, c = args
    out = serving.chat(ENDPOINT, ic, build_prompt(c["request"], c["candidates"]), S.region)
    r = checks.check_text(out["text"], c["request"], c["candidates"])
    return {"model": name, **{k: r[k] for k in ["json_valid", "hard_pass", "within_budget",
            "no_avoid", "item_count_ok"]}, "tokens": (out.get("usage") or {}).get("completion_tokens"),
            "latency_s": out["latency_s"]}

work = [(n, ic, c) for n, ic in targets.items() for c in cases]
with ThreadPoolExecutor(2) as ex:                   # the endpoint streams one request at a time
    res = pd.DataFrame(list(ex.map(score, work)))
res.groupby("model").mean(numeric_only=True).round(3)
"""),
    md("""
## 3. Streamlit demo

Run the cell, then open **`<your Studio URL>/jupyterlab/default/proxy/8501/`** in a new tab. Pick a
random easy / medium / hard / extreme scenario, compare the models side by side, and open the
Catalog page to see the traps each request hides. Your own models show up in **Compare** once you
re-run the deploy cell after they finish; until then the sidebar shows their training status.
"""),
    code("""
!cd .. && OCTANK_APP_RUN=attendee nohup streamlit run app/app.py --server.port 8501 --server.headless true --server.enableCORS false --server.enableXsrfProtection false > work/streamlit.log 2>&1 &
print("open: /jupyterlab/default/proxy/8501/")
"""),
    md("## 4. Clean up (stops endpoint billing)"),
    code("""
# serving.delete_all(ST["endpoint"], S.region, confirm_release_endpoint=True)
"""),
])

# =============================================================================================
# ATTENDEE 05 - compare
# =============================================================================================
write("05_compare.ipynb", [
    md("""
# Lab 5 - Compare: accuracy, tokens and latency by problem type

Every model scored on the **same 1,184 benchmark cases** (`data/reference_results.csv`, from the
instructor's final evaluation):

| Model | Trained on |
|---|---|
| Base | - |
| Attendee-scale SFT / SFT -> RLVR (1K) | 1,000 raw requests: your data size, same pipeline |
| Instructor SFT / SFT -> RLVR (20K) | 20,000 raw requests |

| Track | What it tests |
|---|---|
| Typical | the natural request mix |
| Hard, 40 / 80 candidates | hard requests, with known and never-seen products, and with twice the traps |
| Extreme | the hardest scenarios (difficulty 70+) |

**Accuracy** = every hard rule passes (checked in code). **Latency** is estimated from tokens at the
endpoint's measured ~24 tokens/s. Look at the problem types separately: the overall average hides
where RL helps. If you turned on `EVALUATE` in Lab 3, your own models join the tables when their
evaluation finishes.
"""),
    boot("attendee"),
    code("""
from octank import arms, benchmarks, workflow
jobs = arms.status(S)
print("Your jobs:"); display(jobs if len(jobs) else "none launched (Lab 3)")
ref = pd.read_csv(config.DATA_DIR / "reference_results.csv")
mine = arms.results(S)                                   # only if you evaluated your models
df = pd.concat([ref, mine.assign(model="my_" + mine["model"]) if len(mine) else mine], ignore_index=True)
ORDER = [m for m in ["base", "attendee_scale_sft", "attendee_scale_sft_rlvr", "my_sft", "my_sft_rlvr",
                     "instructor_sft", "instructor_sft_rlvr"] if m in set(df.model)]
print("models:", ORDER, "| cases per model:", df.groupby("model").size().to_dict())
"""),
    code("""
t = workflow.by_problem_type(df)
TPS = 24.0
t["latency_s"] = t["tokens"] / TPS
pivot = lambda col: t.pivot(index="problem_type", columns="model", values=col).reindex(
    [p for p, _ in benchmarks.PROBLEM_TYPES])[ORDER]
print("Accuracy (rule pass)"); display(pivot("rule_pass").style.format("{:.1%}"))
print("Average tokens");       display(pivot("tokens").round(0))
print("Estimated latency (s)"); display(pivot("latency_s").round(0))
print("Cut off (no answer)");  display(pivot("cut_off").style.format("{:.1%}"))
"""),
    md("## 1. Where does each stage help?"),
    code("""
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
pivot("rule_pass").plot.bar(ax=axes[0], rot=15, width=0.85)
axes[0].set_ylim(0, 1); axes[0].set_ylabel("rule pass"); axes[0].set_title("Accuracy by problem type")
pivot("tokens").plot.bar(ax=axes[1], rot=15, width=0.85)
axes[1].set_ylabel("tokens"); axes[1].set_title("Average tokens by problem type")
for ax in axes: ax.legend(fontsize=8)
plt.tight_layout()
"""),
    md("""
## 2. SFT vs SFT -> RLVR on the same cases

Paired comparison: *fixed* = cases the RLVR model passes and SFT fails; *broke* = the reverse.
p is an exact sign test - with only a few hundred cases, small differences are noise. Compare the
two scales: with 20x the data RL has a strong SFT model to sharpen; at 1K data (one RL step) the
gain is smaller and mostly within noise.
"""),
    code("""
from math import comb
def paired(a, b, tracks):
    p = df[df.track.isin(tracks) & df.model.isin([a, b])].pivot_table(index=["track", "case"], columns="model", values="hard_pass").dropna()
    if p.empty or a not in p or b not in p:
        return None
    x, y = p[a].astype(bool), p[b].astype(bool); up, dn = int((~x & y).sum()), int((x & ~y).sum())
    pv = min(1.0, 2 * sum(comb(up + dn, k) for k in range(min(up, dn) + 1)) / 2 ** (up + dn)) if up + dn else 1.0
    return {"n": len(p), "SFT": f"{x.mean():.1%}", "SFT -> RLVR": f"{y.mean():.1%}", "fixed": up, "broke": dn, "p": round(pv, 4)}
rows = []
for who, a, b in (("instructor (20K)", "instructor_sft", "instructor_sft_rlvr"),
                  ("attendee scale (1K)", "attendee_scale_sft", "attendee_scale_sft_rlvr"),
                  ("yours", "my_sft", "my_sft_rlvr")):
    for name, tracks in benchmarks.PROBLEM_TYPES:
        r = paired(a, b, tracks)
        if r:
            rows.append({"models": who, "problem type": name, **r})
pd.DataFrame(rows)
"""),
    md("## 3. Same request, side by side"),
    code("""
from octank import checks as ck
cases = {c["request"]["request_id"]: c for c in benchmarks.cases_40()}
case = df[df.track == "hard_known"]["case"].sample(1, random_state=3).iloc[0]   # change random_state to browse
c = cases[case]
print("SHOPPER:", c["request"]["message"])
print("IMPLIED BY THE SITUATION:", c["request"].get("hidden_avoid") or "-", "\\n")
by_id = {x["id"]: x for x in c["candidates"]}
for m in ORDER:
    r = df[(df.model == m) & (df.case == case) & (df.track == "hard_known")]
    if r.empty:
        continue
    r = r.iloc[0]
    print(f"{m:24s} {'PASS' if r.hard_pass else 'FAIL'} {'(cut off)' if r.cut_off else '':10s} {int(r.tokens):5d} tok")
"""),
])

# =============================================================================================
# INSTRUCTOR
# =============================================================================================
write("instructor.ipynb", [
    md("""
# Instructor run (prepared before the session)

For the workshop team, not attendees. **Same `octank` package, same steps and settings as the
attendee notebooks** - only the dataset is 20x larger:

1. **Data build** - 20,000 raw requests -> curation -> gap-fill -> teacher traces -> SFT / RLVR / eval
   files (`scripts/build_data.py instructor`).
2. **Train and evaluate** - SFT, then SFT -> RLVR with the same reward Lambda and hyperparameters
   (`scripts/train_eval.py instructor`).
3. **Final evaluation** of every demo model in one pass, which also writes
   `data/reference_results.csv` for Lab 5 (`scripts/final_eval.py`).
4. **Demo endpoint** with the instructor and attendee-scale adapters (`scripts/deploy_demo.py`).

Seed data (catalog, gold set, benchmarks, scenarios) is committed in `data/`. The run's LLM caches and
state are version-controlled in `instructor/`, so a rebuild only calls Bedrock for inputs that changed;
derived files (parquet, datasets, logs) go to `work/instructor/`. The long steps run as background
scripts; this notebook monitors them.
"""),
    INSTALL,
    boot("instructor"),
    md("## 1. Data build (background, resumable)"),
    code("""
STATUS = S.local("logs", "status.json")
# !cd .. && nohup ~/.venvs/gift_concierge/bin/python scripts/build_data.py instructor > work/instructor/logs/build.out 2>&1 &
print(open(STATUS).read() if STATUS.exists() else "not started")
"""),
    code("""
from octank import profile
from octank.pipeline import INSTRUCTOR, Run
run = Run(S, INSTRUCTOR)
raw = pd.read_parquet(run.path("raw_requests.parquet"))
final = pd.read_parquet(run.path("final.parquet"))
print(len(raw), "raw ->", len(final), "final")
pd.read_csv(run.path("curation_report.csv"))
"""),
    code("""
print("raw skew:", profile.skew_report(raw), "\\nfinal skew:", profile.skew_report(final))
profile.plot_overview(final, "Instructor dataset (final)");
"""),
    md("""
## 2. Gold set and human annotation slice

`data/gold_labels.jsonl` (150 rows, `gpt-oss-120b` drafts) calibrates the labeler in Lab 2.
`data/human_annotation_slice.csv` (50 rows) has the message, the `draft_*` labels and empty `human_*`
columns: annotators fill in `human_*` only where the draft is wrong, plus `annotator`. Merge it back:
"""),
    code("""
from octank import gold
g = gold.load_gold()
print(len(g), g["_flaw"].value_counts().to_dict())
# g = gold.merge_human(g, gold.HUMAN_SLICE_PATH.with_name("human_annotation_slice_done.csv"))
# gold.save_gold(g)
pd.read_csv(gold.HUMAN_SLICE_PATH).head()
"""),
    md("""
## 3. Train and evaluate: SFT, then SFT -> RLVR

```bash
cd .. && nohup ~/.venvs/gift_concierge/bin/python scripts/train_eval.py instructor > work/instructor/logs/train_eval.out 2>&1 &
```
Same arms as the attendee Lab 3 (`octank.arms`), with evaluation on (`evaluate=True`); resumable from `instructor/state.json`.
"""),
    code("""
from octank import arms, workflow
arms.status(S)
"""),
    code("""
res = arms.results(S)
workflow.by_problem_type(res).pivot(index="problem_type", columns="model", values="rule_pass").round(3) if len(res) else "no results yet"
"""),
    md("""
## 4. Final evaluation and demo endpoint

```bash
cd .. && nohup ~/.venvs/gift_concierge/bin/python scripts/final_eval.py > work/final_eval/run.out 2>&1 &
cd .. && nohup ~/.venvs/gift_concierge/bin/python scripts/deploy_demo.py > work/demo/deploy.out 2>&1 &
```
`final_eval.py` scores base, instructor and attendee-scale models in one pass (same cases) and
refreshes `data/reference_results.csv`.
"""),
])

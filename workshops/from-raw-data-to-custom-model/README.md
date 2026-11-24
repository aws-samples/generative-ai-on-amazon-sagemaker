# From raw data to custom model: dataset prep with Amazon SageMaker AI

In this workshop you take a small open-weight model from **raw,
messy synthetic data** to a **custom model you can try live**. You generate data, profile it,
curate it, prepare it for supervised fine-tuning and reinforcement learning, launch **SFT, then
SFT → RLVR** training with **serverless model customization**, and deploy fine-tuned models behind
a demo app. The demo and the comparison use the **instructor's models**, trained with the same
notebooks on 20× more data before the session, so nothing in the session waits for training.

> Great models start with great data. Most of this workshop is about the data; the training,
> evaluation and deployment steps show what that data buys you.

---

## The scenario: Octank Gift Concierge

**Octank** is a fictional gift shop. A shopper describes who they're shopping for. The
**concierge** model gets that request plus **40 candidate products** from the catalog, listed the
way a real catalog lists them: name, category, price and description, with no tags. It thinks
through the request, then answers with its picks, a short "why" for each, the total, and a
gift-card message.

**Example request**

> "Two housewarming gifts for my sister, $60 total. She loves cooking and plants, and she just
> moved into a tiny studio with almost no storage."

**Example answer** (illustrative; it follows the model's reasoning, which is wrapped in `<think>…</think>`)

```json
{
  "picks": [
    {"id": "OCT-1412", "why": "A compact windowsill herb kit: plants she can cook with, and it fits a tiny studio"},
    {"id": "OCT-1157", "why": "A slim magnetic spice set that mounts on the fridge, so it takes no cupboard space"}
  ],
  "total": 54.50,
  "gift_message": "Small space, big flavor. Happy housewarming!"
}
```

**What makes this hard.** Only about 6 of the 40 candidates are real matches. The rest are traps:
items just over budget, pairs that break a multi-item total, strong interest matches that carry
something the shopper wants to avoid, near-duplicates and filler. About 45% of requests also
contain a **need implied only by the shopper's situation** ("a tiny studio with almost no storage"
means nothing bulky); the form never states it. Every answer must pass these **hard
rules**, which code checks deterministically:

- valid JSON
- every id is in the candidate list
- the requested number of items
- no duplicates
- total within budget
- nothing the shopper asked to avoid, and nothing their situation rules out

Out of the box, the base model (**Qwen3.5-4B**, thinking on) almost never finishes: it reasons
until its token budget runs out on 99.5% of requests. A fine-tuned model answers in a fraction
of that and follows the rules.

---

## What you'll learn

| Stage | You will... | Techniques |
|---|---|---|
| Generate | Create ~1,000 realistic shopper requests | Spec sampling + LLM writing (`gpt-oss-120b`), hidden needs, planted data flaws |
| Profile | Measure completeness, skew and coverage gaps | Distributions, entropy, segment coverage |
| Curate | Fix the problems you found | Exact + **MinHash/LSH** dedup, **toxicity filtering** with a plain-language policy, **auto-labeling** with confidence calibrated on a gold set, **human review** of edge cases, **synthetic gap-fill** |
| Prepare | Build training data for SFT and RLVR | Teacher traces with **rejection sampling**; **SFT**, **RLVR** and evaluation datasets; validation with the `aws-ai-ml` agent skill |
| Reward | Review the function RL will optimize | A **hybrid reward**: exact rule checks in code plus an LLM taste score, with anti-gaming penalties |
| Train | Fine-tune without managing infrastructure | Serverless **SFT** with LoRA, then **SFT → RLVR** from your SFT model |
| Evaluate | Measure quality two ways | **Deterministic rule checks** and **LLM-as-a-Judge** (custom metrics with the real ground truth); optional for your own models |
| Deploy | Serve fine-tuned adapters next to the base model | SageMaker real-time endpoint with **inference components**, one per LoRA adapter |
| Compare | See what each step and data scale bought you | Base vs 1K vs 20K models: accuracy, tokens and latency **by problem type**, paired comparisons |

---

## How it fits together

```
 Lab 1                     Lab 2                                   Lab 3
 ┌────────────┐  ┌──────────────────────────────────────┐  ┌────────────────────────────┐
 │ catalog    │  │ dedup → toxicity → auto-label →      │  │ candidates → teacher traces│
 │ + generate │→ │ human review → resolve → gap-fill    │→ │ → SFT / RLVR / eval files  │
 │ + profile  │  │ (scored against a hidden answer key) │  │ → reward review → launch   │
 └────────────┘  └──────────────────────────────────────┘  └─────────────┬──────────────┘
                                                                         │ your jobs keep
                                                                         ▼ running in the background
 Lab 0                     Lab 4                                   Lab 5
 ┌─────────────────┐  ┌──────────────────────────────────┐  ┌──────────────────────────────┐
 │ endpoint + base │→ │ instructor adapters (from repo), │  │ base vs 1K vs 20K models,    │
 │ model (early)   │  │ live checks, Streamlit demo      │  │ by problem type (same cases) │
 └─────────────────┘  └──────────────────────────────────┘  └──────────────────────────────┘
```

**Your models and the instructor's.** You build your own data and launch your own training. The
instructor ran the same notebooks, with the same code, settings and reward, on 20,000 raw requests
instead of 1,000. Those models (`adapters/` in the repo) are what you deploy in Lab 4 and compare
in Lab 5, next to models trained at your data size. Your own SFT model can join the demo if it
finishes in time; your SFT → RLVR job usually finishes after the session.

---

---

## Models and why each one was chosen

| Role | Model | Why |
|---|---|---|
| **The model you train** | Qwen3.5-4B (`huggingface-vlm-qwen3-5-4b`), thinking on | Small, open (Apache 2.0), supported by the serverless SFT and RLVR recipes |
| Data generator and teacher | `gpt-oss-120b` on Amazon Bedrock | Open-weight, so its outputs can be used as training data |
| Auto-labeler | `gpt-oss-20b` on Bedrock | Fast and accurate once calibrated (94–98% per field on the gold set) |
| Toxicity filter | `gpt-oss-safeguard-20b` on Bedrock | A safety classifier you steer with a plain-language policy |
| Reward taste score | `gpt-oss-20b` on Bedrock | Open-weight and available in every workshop region; the rules themselves are checked in code |
| Evaluation judge | Claude Haiku 4.5 | Supported LLM-as-a-Judge model. **Used only for scoring, never for training data or rewards** |

Every model is available in both us-east-1 and us-west-2. No embedding models are used:
near-duplicate detection is MinHash on word 3-grams.

---

## Prerequisites

- A Workshop Studio account (provided at the event). Everything runs in **the account's current
  region** (us-east-1 or us-west-2) and writes to the SageMaker default bucket under
  `s3://sagemaker-<region>-<account>/gift-concierge/attendee/`.
- SageMaker Studio with JupyterLab. Open the `workshops/from-raw-data-to-custom-model/notebooks/` folder.
- Bedrock model access to the five models above. Lab 0 checks this for you.
- Some Python familiarity. You run cells and change a few values; you don't write code from
  scratch.

---

## Agenda

The session is **2 hours**. Nothing waits on training: Labs 4 and 5 use the instructor's models.

| Time (approx.) | You | In the background |
|---|---|---|
| 0:00 – 0:10 | **Lab 0** Setup | Endpoint and base model start (~15 min) |
| 0:10 – 0:25 | **Lab 1** Generate and profile | |
| 0:25 – 0:55 | **Lab 2** Curate | |
| 0:55 – 1:15 | **Lab 3** Prepare, review the reward, launch training | Your SFT trains (~16 min), then SFT → RLVR (~40 min), plus any capacity wait |
| 1:15 – 1:40 | **Lab 4** Deploy the instructor's models and try them | Add your SFT model if it has finished |
| 1:40 – 2:00 | **Lab 5** Compare | |

Times are planning targets.

---

---

## Lab 0: Setup (`00_setup.ipynb`)

**Goal:** prepare your environment and start your endpoint early, so GPU start-up time overlaps
with Labs 1–3.

**Steps**
1. Open `00_setup.ipynb` and select the Python 3 kernel.
2. Run the install cell (pinned SDKs), then **restart the kernel**.
3. Run the setup cell. It prints your region, bucket, prefix and execution role.
4. Run **Check model access**. All five models should print `OK`.
5. Run **Start your endpoint now**. This creates an `ml.g7.2xlarge` endpoint with the base model
   in the background, using SageMaker Python SDK resources (`Model`, `EndpointConfig`,
   `Endpoint`, `InferenceComponent`).

**Checkpoint:** five `OK` lines, and "Creating … in the background - continue with Lab 1."

**If something fails:** a `FAIL` on model access usually means Bedrock model access isn't
enabled for that model in your region. Ask a facilitator.

---

## Lab 1: Generate and profile raw data (`01_generate_profile.ipynb`)

**Goal:** create a realistic raw dataset and find out what's wrong with it before you train on
it.

**Steps**
1. **Catalog.** Load the Octank catalog (about 1,200 products, shipped with the repo) and
   profile it. It has planted defects: inconsistent category spellings, duplicate SKUs, items
   with missing tags. Clean it. Cleaning is deterministic, so everyone gets the same catalog.
2. **Generate.** Look at an example *spec*: relationship, age, occasion, budget, number of items,
   interests, things to avoid, a note, a writing voice and, for about 45% of requests, a
   **situation** ("recently got sober", "tiny studio apartment") that implies something to avoid.
   `gpt-oss-120b` turns each spec into the free-text message a real shopper would type; the
   situation appears only in passing. Run generation: about 1,050 rows, typically under a minute.
3. **Profile.** Check completeness per field, the occasion distribution and skew, text
   statistics, and **coverage gaps** (segments under 2% of rows).

**Checkpoint:** you should see that about 60% of requests are Christmas or birthdays, that
several occasions and the under-$25 band barely appear, and that some form fields are missing.

**Think about it:** the generator planted flaws deliberately: duplicates, missing fields, toxic
and ambiguous requests, contradictions between the form and the message, and skew. A hidden
`_flaw` column records them as an **answer key**. You won't use it until the end of Lab 2, where
it scores how well your curation worked. A real dataset doesn't come with one.

> **Agent skill:** you can ask the SageMaker AI agent *"Profile my raw requests file for
> completeness and skew"*.

---

## Lab 2: Curate (`02_curate.ipynb`)

**Goal:** turn the raw data into a clean, balanced, correctly labeled dataset.

**Steps**
1. **Deduplicate.**
   - Remove exact duplicates (hash of normalized text).
   - Find near-duplicates with **MinHash + LSH**. MinHash gives each message a 128-number
     signature; the share of matching positions approximates the Jaccard similarity of the two
     messages' word 3-grams.
   - Try the threshold sweep (0.3 / 0.5 / 0.7 / 0.9) and read a few pairs to see what each
     threshold catches.
2. **Filter toxicity.**
   - Read the plain-language policy that `gpt-oss-safeguard-20b` classifies against.
   - Run the filter and read the rationales for flagged rows.
   - *Try it:* add a rule to the policy and re-run.
3. **Auto-label, calibrated first.**
   - The free-text message is the source of truth; form fields are often missing or wrong.
   - Measure `gpt-oss-20b`'s label accuracy on the **gold set** (150 rows: drafts by
     `gpt-oss-120b`, to be corrected by humans).
   - Use the calibration curve to choose `REVIEW_THRESHOLD`. It decides how many rows are
     accepted automatically and how many go to a human. The default is 0.7.
   - Label the full dataset.
4. **Review as a human.** Work through about 15 queued items: approve, edit or reject. They
   include unclear intent, form vs. text disagreements, no budget, borderline safety and low
   confidence. Anything you skip is resolved by a default policy, so the pipeline always
   completes.
5. **Score your curation** against the hidden answer key: how many planted toxic, ambiguous,
   duplicate, missing and contradictory rows did you catch?
6. **Fill coverage gaps.** Generate new requests for under-covered segments with the same
   generator. They go through the same safety, labeling and finalize steps. Compare the skew
   before and after.

**Checkpoint:**
- The curation report shows roughly 95–100% of each planted flaw type caught.
- The Christmas + birthday share drops after gap-fill.
- Your final dataset has about 1,000 rows.

**Think about it:** what would you lose by skipping calibration and accepting every label? What
would it cost to send every row to a human?

---

## Lab 3: Prepare data, review the reward, train (`03_prepare_train.ipynb`)

**Goal:** turn curated requests into training data, understand the reward that RL will optimize,
and start training.

**Steps**
1. **Build candidate lists.** Each request gets 40 untagged candidates: ~6 real matches plus
   traps. Print one full prompt to see exactly what the model sees.
2. **Generate teacher traces with rejection sampling.**
   - `gpt-oss-120b` solves each request with its reasoning visible.
   - Code checks every answer against every rule, including needs implied by the situation, and
     keeps only passing answers whose reasoning stays under ~1,100 tokens.
   - Expect a pass rate of about 90%.
3. **Write the datasets:**

   | File | Used by | What it contains |
   |---|---|---|
   | `sft_train.jsonl` | SFT | prompt + verified teacher answer (reasoning and JSON), ~700 rows |
   | `rlvr_train.jsonl` | RLVR | prompt + the ground truth the reward needs (budget, item count, avoid list, implied needs, every candidate's price and tags, difficulty, teacher reference), 200 prompts; **no answers to imitate**. 20% are freshly generated **Extreme** requests |
   | `eval.jsonl` | Evaluation | evaluation rows for your own holdout |

4. **Validate** every file with the `aws-ai-ml` skill's `format_detector.py`. Each should report
   `Valid: ✓`. Then upload to S3.
5. **Review the reward function** (`judge/reward_lambda.py`). It is a pyramid: higher layers count
   only once the lower ones pass.

   | Layer | Score |
   |---|---|
   | No parsable answer (cut off) | −1.0 |
   | Bad ids / wrong count / duplicates | −0.6 |
   | Picks an item the situation rules out | −0.4 |
   | Picks an item the shopper said to avoid | −0.3 |
   | Over budget | −0.1 |
   | All rules pass | 0.30 + 0.30 × fit + 0.10 × variety + 0.30 × taste − length penalty |
   | Anti-gaming | echo of the prompt −1.0, repetitive reasoning −0.4, empty "why" −0.25 |

   Predict the score of five crafted answers, then run the function locally (deterministic part
   only). Then deploy it as a Lambda function. `reward.verify_matches` proves the deployed code is
   the file you reviewed.
6. **Launch training.** Both stages run as background jobs:

   | Arm | Starts from | Data | Learning signal | Time |
   |---|---|---|---|---|
   | `sft` | base model | ~700 prompts **with** verified answers | imitate the teacher | ~16 min |
   | `sft_rlvr` | your SFT model | 200 prompts (1 RL step), **no answers** | the reward function scores 8 sampled answers per prompt | ~40 min |

   `sft_rlvr` starts as soon as SFT finishes. **For the rest of the session you use the
   instructor's models**, so you don't wait for these jobs. Evaluating your own models is off by
   default (`EVALUATE = False`), because it would finish after the session; set it to `True` to run
   the same fast evaluation the instructor used.

**Checkpoint:** `arms.status(S)` shows `sft` queued or training. Re-run that cell any time. If your
kernel restarts, re-run the launch cells: the arms resume from saved state and don't resubmit.
**Keep the Lab 3 kernel running until `sft_rlvr` shows `phase=training`**: the SFT → RLVR launch
waits for SFT in that kernel. If it stops first, Lab 4 tells you and resumes it with
`arms.resume_rlvr(S)`.

> **Agent skill:** *"Fine-tune Qwen3.5-4B with SFT, then continue from the SFT model with RLVR
> using my reward Lambda."*

**Think about it:** SFT needs *answers*; RLVR needs only *prompts and a reward*. The teacher's
rejection sampling guarantees the training answers are correct, not that they're the best, so SFT
can't learn beyond what the teacher solved. What would a model learn if you removed the length
penalty or the empty-"why" penalty?

---

## Lab 4: Deploy and try the models (`04_evaluate_deploy.ipynb`)

**Goal:** serve fine-tuned adapters next to the base model and see the difference live.

**Steps**
1. **Deploy the instructor's adapters.** The deploy cell copies `adapters/v2-sft` and
   `adapters/v2-sft-rlvr` from the repo to your bucket and attaches each as its own **inference
   component** that shares the base model's GPU. If your own SFT model has finished training, the
   same cell adds it; re-run it later to add more.

   | Model | Trained on |
   |---|---|
   | Instructor SFT (20K) | 20,000 raw requests, same pipeline |
   | Instructor SFT → RLVR (20K) | 20,000 raw requests, same pipeline |
   | Your SFT (1K), Your SFT → RLVR (1K) | your Lab 3 data, when the jobs finish |

2. **Check live outputs.** Run 12 benchmark requests against each deployed adapter and apply the
   deterministic checks: valid JSON, hard pass, within budget, no avoided items, item count,
   tokens and latency.
3. **Open the Streamlit demo** at `<your Studio URL>/jupyterlab/default/proxy/8501/`.
   - Fill in the form, or use **🎲 Random scenario** / **🟢 easy** / **🟠 medium** / **🔴 hard** /
     **🟣 extreme** to load a held-out request of that difficulty. Benchmark scenarios show the
     audience the hidden need the models must infer.
   - Every request gets **80 candidates**: the same ~6 real matches among twice the traps of the
     40-candidate lists the models were trained on. That crowded setting is where SFT → RLVR beats
     SFT most.
   - Above **Find gifts**, the **request difficulty** (Easy / Medium / Hard / Extreme, 0–100)
     explains what makes the request hard: implied needs, items to fit in one budget, budget
     tightness, few good options, trap items.
   - **Browse the catalog** (sidebar or the 📚 link) to search all products, or to see the
     current request's candidates labeled by role: real match, trap, over budget, filler.
   - Models run one after another, each with its own timer and live-streamed reasoning. The
     "done in … · tokens · tok/s" header stays on each column after it finishes.
   - Each answer shows product cards, rule badges, total vs. budget, the gift message, and a judge
     score (the evaluation metrics) with a **Why this score** explanation. The score is gated by
     the rules: 0 if there's no answer, at most 0.2 if a rule is broken. If a model runs out of
     tokens, its unfinished reasoning is still shown.
   - **Token use on this request** compares tokens and time against a reference model, with a
     running tally for your session.
   - Your own models appear under **Compare** once deployed; until then the sidebar shows their
     training status.
4. **Own account only:** when you're done, uncomment and run the last cell to delete the
   endpoint. In Workshop Studio there's nothing to clean up.

**Checkpoint:** the base model is cut off on nearly every request; the instructor's models answer
and pass most of them, SFT → RLVR in fewer tokens.

---

## Lab 5: Compare every model (`05_compare.ipynb`)

**Goal:** see, with numbers, what data preparation, data scale and each training stage bought
you, **by problem type**.

**Steps**
1. Run the collection cell. It shows your jobs' status and loads the reference results
   (`data/reference_results.csv`): base, **attendee-scale** models (1,000 raw requests, your data
   size) and the **instructor's** models (20,000), all scored on the same 1,184 cases. If you
   turned on `EVALUATE` in Lab 3, your own models join when their evaluation finishes.
2. **By problem type:** accuracy (every hard rule passes), average tokens, estimated latency and
   cut-off rate for typical, hard (40 and 80 candidates) and Extreme requests.
3. **SFT vs SFT → RLVR on the same cases:** fixed, broke and an exact sign-test p-value, at both
   data scales.
4. **Same request, side by side:** every model's pass/fail and tokens for one hard request. Change
   `random_state` to browse.

**What to expect** (accuracy · average tokens, same 1,184 cases for every model):

| Problem type | Base | Attendee-scale SFT | Attendee-scale SFT → RLVR | Instructor SFT | Instructor SFT → RLVR |
|---|---|---|---|---|---|
| Typical (261) | 1.5% · 4,093 | 85.8% · 1,297 | 87.0% · 1,192 | 96.6% · 839 | 95.8% · 554 |
| Hard, 40 candidates (432) | 0.2% · 4,092 | 65.7% · 1,673 | 71.1% · 1,549 | 83.6% · 1,091 | 84.7% · 686 |
| Hard, 80 candidates (431) | 0.0% · 3,072 | 57.5% · 1,792 | 58.0% · 1,693 | 77.5% · 1,363 | 84.5% · 756 |
| Extreme, 80 candidates (60) | 0.0% · 3,072 | 46.7% · 2,280 | 63.3% · 2,028 | 71.7% · 1,750 | 86.7% · 1,069 |

- **SFT does the heavy lifting:** from almost never answering to passing most requests.
- **Data scale matters most:** with 20× more curated data, SFT gains 11 points on typical and 19
  on hard requests (p < 10⁻⁶).
- **RLVR's gain hides in the average.** It is flat on typical requests and grows with difficulty:
  for the instructor's models +4.1 points on the hard tracks (p = 0.002) and +15 on Extreme
  (p = 0.012), with 34–45% fewer tokens on every problem type. With one RL step at 1K data the
  gain is smaller and mostly within noise.

**Think about it:**
- Why would RL help most where SFT is weakest?
- Where do the judge and the deterministic checks agree, and where do they disagree?

---

## Clean up

- **Workshop Studio:** nothing to do. The account and everything in it is removed after the
  session.
- **Your own account:** run the cleanup cell at the end of Lab 4 to delete the endpoint (endpoints
  are billed per hour until deleted). Training jobs and evaluations stop costing compute once they
  finish. Your S3 data is under `s3://sagemaker-<region>-<account>/gift-concierge/attendee/`. The reward
  Lambda (`SageMaker-octank-hybrid-reward`) costs nothing when idle.

---

## Troubleshooting

| Symptom | What to do |
|---|---|
| A model shows `FAIL` in Lab 0 | Bedrock model access isn't enabled for that model in your region; ask a facilitator |
| The endpoint takes a long time or fails with capacity errors | GPU capacity is shared; Lab 4 waits for it. Continue with Labs 1–3 |
| The kernel restarted mid-lab | Re-run the setup cell. Generation, labeling and teacher steps are **cached and resume**; training arms resume from saved state |
| A training job shows `Pending` | Serverless capacity is being allocated; it starts automatically |
| My own model isn't in the demo | Its training job hasn't finished; check `arms.status(S)` and re-run Lab 4's deploy cell later |
| `sft_rlvr` never started | The Lab 3 kernel stopped while it waited for SFT; Lab 4's first cell detects this - run `arms.resume_rlvr(S)` |
| Reward Lambda deploy fails with `PermissionError` / AccessDenied | Your role can't create IAM roles. Set `OCTANK_REWARD_ROLE_ARN` to the pre-provisioned reward role (ask a facilitator), restart the kernel and re-run Lab 3 section 5 |
| Format check fails: `format_detector.py not found` | The `aws-ai-ml` agent skill isn't installed under `~/.kiro/skills/`; ask a facilitator |
| The Streamlit page doesn't load | Check the URL ends in `/proxy/8501/`, then re-run the Streamlit cell |

---

## Glossary

| Term | Meaning |
|---|---|
| **SFT** | Supervised fine-tuning: the model learns to imitate example answers |
| **RLVR** | Reinforcement learning with verifiable rewards: a reward function scores each sampled answer, and the model moves toward the better ones |
| **Hybrid reward** | A reward that checks rules exactly in code and adds an LLM score for qualities code can't measure |
| **Hidden need** | Something to avoid that only the shopper's situation implies ("a tiny studio with almost no storage" → nothing bulky) |
| **LoRA adapter** | A small set of trained weights layered on the base model; many adapters can share one base |
| **Inference component** | A SageMaker hosting unit; here, one for the base model plus one per adapter on the same endpoint |
| **Rejection sampling** | Generate answers, keep only the ones that pass the checks |
| **MinHash / LSH** | A fast way to find near-duplicate text without embeddings |
| **LLM-as-a-Judge** | Using a strong LLM to score outputs against custom metrics |
| **Cut off** | The model used its whole token budget without producing an answer |
| **Hard pass / accuracy** | The answer satisfies every hard rule |

---

## Repository map

```
from-raw-data-to-custom-model/
  README.md                this file
  notebooks/               00_setup … 05_compare (attendee), instructor
  src/octank/              the shared Python package the notebooks call
  data/                    catalog, gold labels, benchmarks, demo scenarios, reference results
  instructor/              the instructor run's LLM caches and state (version-controlled)
  judge/reward_lambda.py   the hybrid RLVR reward function (deployed as a Lambda function)
  app/app.py               Streamlit demo (app/pages/1_Catalog.py: catalog browser)
  adapters/                the instructor's LoRA adapters (bf16)
  scripts/                 instructor data build, training, final evaluation, demo deployment
  tests/test_offline.py    offline unit tests (no AWS)
```

---

## Instructor appendix: the full-scale demo

The instructor runs **the same code, settings, reward and evaluation** on **20,000 raw requests**
(attendees use 1,000) to produce the reference results in Lab 5 and the live demo. Run
`notebooks/instructor.ipynb` or the scripts directly:

| Step | Command | Output |
|---|---|---|
| Data build | `python scripts/build_data.py instructor` | `work/instructor/datasets/`; LLM caches in `instructor/cache/` |
| Train and evaluate | `python scripts/train_eval.py instructor` | SFT → SFT + RLVR, each fast-evaluated (`evaluate=True`) |
| Final evaluation | `python scripts/final_eval.py` | all demo models on 1,184 cases; `data/reference_results.csv` (base, instructor, attendee-scale) |
| Demo endpoint | `python scripts/deploy_demo.py` | `octank-demo` with base + four adapters |
| Demo app | `bash scripts/restart_app.sh demo` | Streamlit on port 8501 |

- **Instructor data in version control:** `instructor/cache/` holds the run's LLM results
  (generated messages, safety verdicts, labels, gap-fill, teacher answers). `instructor/state.json`
  (datasets, jobs and model packages in your own account) is created by the build and is git-ignored. A rebuild reuses every cached result and only calls Bedrock
  for inputs that changed; parquet files, datasets and logs are derived into `work/instructor/` in
  minutes. Commit `instructor/cache/` after a build. Attendee runs stay in `work/attendee/` (not tracked).
- **Results** (paired, same cases): instructor SFT → RLVR vs SFT is −0.8 points on typical
  requests (p = 0.75), **+4.1 on the four hard tracks** (fixed 80 / broke 45, p = 0.002) and
  **+15.0 on Extreme** (fixed 10 / broke 1, p = 0.012), with tokens down from 1,168 to 702 on
  average.
- **Maintenance:** `pip install -r requirements-dev.txt` adds `nbformat`, `nbclient` and `pytest`
  for `scripts/build_notebooks.py`, `scripts/run_attendee_test.py` and
  `python -m pytest tests/test_offline.py -q` (15 offline tests).
- **Live demo:** the Streamlit app against the shared `octank-demo` endpoint
  (`OCTANK_APP_RUN=demo`), serving the base model, **Instructor SFT**, **Instructor SFT → RLVR**,
  **Attendee-scale SFT (1K samples)** and **Attendee-scale SFT → RLVR (1K samples)**. The app
  preselects the first three; add the attendee-scale models under **Compare**. Requests always use
  **80 candidates**: the models were trained on 40, and that's where SFT and RLVR separate most.
- **Adapters:** `adapters/v2-sft` and `adapters/v2-sft-rlvr` (bf16) are the instructor's
  adapters that every attendee deploys in Lab 4 (`serving.upload_adapter`). If you retrain them, update these folders. The RL recipe saves LoRA adapters
  in float32; `scripts/adapter_to_bf16.py` converts them.

### Notes for writing workshop instructions from this README

- Each lab section follows the same pattern: **Goal → Steps → Checkpoint → Think about it /
  Agent skill**. Use these as the page structure in Workshop Studio.
- The notebooks are generated by `scripts/build_notebooks.py`. Change the script, not the
  `.ipynb` files, and regenerate. `scripts/run_attendee_test.py --dry-run` runs Labs 1–3 and 5
  headless (training replaced by trainer dry runs).
- The "What to expect" numbers come from runs in us-west-2. Refresh them after the dry run in
  us-east-1, and confirm the agenda times there.

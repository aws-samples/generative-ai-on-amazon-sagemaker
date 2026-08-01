# Lab 1 — Supervised Fine-Tuning: contract review that cites its evidence

An end-to-end SFT lab built on a real, human-annotated task where a **fine-tuned
4B model beats Claude Sonnet 5 and Claude Opus 4.8** — on accuracy and on evidence
grounding.

---

## The story this lab tells

**1. A boring, valuable, repetitive job.** Every company signs NDAs, and someone
has to answer the same 17 questions about each one: can the counterparty share
our information with their employees? their subcontractors? must they notify us if
a court compels disclosure? must they return or destroy our material at the end?
can they keep a copy anyway?

**2. The output has to be verifiable.** A verdict alone is not useful — a reviewer
would have to re-read fifteen pages to trust it. The model must also **cite the
clause** that justifies each answer.

**3. The obvious approach half-works.** Claude Sonnet 5 reaches 82.1% accuracy
zero-shot. Respectable, and not enough: its errors are concentrated in *claiming a
protection exists when it does not*.

**4. Prompting does not fix it — and for the small model it makes things worse.**
Five frontier prompting configurations were tested (0/3/5-shot; one call per item
with reasoning). The best was 83.9% accuracy. For the base 4B model, adding
examples *reduced* contradiction-F1 from 46.1 to 8.9.

**5. Fine-tuning does fix it.** 423 annotated contracts, one LoRA job, 34 minutes,
about $3. The result: 88.1% accuracy, 76.1 evidence-F1 — better than both frontier
models on every metric, and better than the published supervised state of the art
for this dataset.

---

## Results

Full 123-contract held-out test set, 2,091 checklist decisions:

| Model | Accuracy | Macro-F1 | **Evidence-F1** | Contradiction-F1 |
|---|---|---|---|---|
| Base Qwen3-4B | 68.0 | 60.5 | 40.4 | 44.0 |
| Claude Sonnet 5 | 82.1 | 77.1 | 68.0 | 63.3 |
| Claude Opus 4.8 | 82.7 | 76.1 | 69.5 | 57.6 |
| **Fine-tuned Qwen3-4B (LoRA)** | **88.1** | **83.9** | **76.1** | **71.3** |

Document-clustered bootstrap (2,000 resamples): **+4.8 accuracy [95% CI +2.7, +7.1]**
and **+8.4 evidence-F1 [+4.4, +12.3]** over Sonnet 5, p < 0.0001.

### Why evidence-F1 is the metric to lead with

NDAs are boilerplate-heavy and several checklist items are 77–86% skewed to one
answer, so a model can reach the sixties on accuracy largely by learning the label
distribution rather than reading the contract.

Evidence-F1 cannot be gamed that way — pointing at the right clause in *this*
document requires reading it. Contradiction-F1 is the other number worth leading
with: it measures whether the model can spot a clause that actively *conflicts* with
a requirement, which is the expensive mistake in contract review.

---

## Notebooks

| Notebook | What it does | Runtime |
|---|---|---|
| `1-prepare-data.ipynb` | The problem, the dataset, build and register training records | ~5 min |
| `2-fine-tune-llm.ipynb` | Serverless LoRA fine-tuning job | ~34 min |
| `3-evaluation.ipynb` | Label-skew check, frontier baseline, then base + tuned Qwen via the managed evaluation pipeline, failure analysis | ~35 min |
| `4-deployment.ipynb` | Deploy to a SageMaker real-time endpoint | ~15 min |
| `4a-deployment-bedrock.ipynb` | Deploy via Bedrock Custom Model Import | ~15 min |

Supporting modules: `contractnli.py` (task, prompt, scoring, baselines, and the
Bedrock call used for the frontier baseline), `contractnli_scorer.py` (the
deterministic scorer registered with the managed evaluation pipeline), `config.py`.

The deployment notebooks deliberately call `boto3` directly rather than hiding the
request behind a helper module — invoking a served model is a thing participants
should see spelled out. Each defines its own small `ask_*` function in the notebook.

Run notebooks **1 → 2 → 3 → (4 or 4a)**.

Notebook 3 needs no deployed model: the base and fine-tuned Qwen are both scored by
the managed evaluation pipeline straight from the registered model package, in one
job (`evaluate_base_model=True`). Endpoints are for **serving** (notebooks 4 and 4a),
not for measuring.

---

## The dataset

[ContractNLI](https://stanfordnlp.github.io/contract-nli/) — Koreeda & Manning,
*Findings of EMNLP 2021*. **CC-BY-4.0.** 607 real NDAs from EDGAR filings and the
public web, annotated against a fixed 17-point checklist with
`Entailment` / `Contradiction` / `NotMentioned` **plus evidence spans**.

Document-level splits stratified by source format: **423 train / 61 dev / 123 test**.
`contractnli.ensure_dataset()` downloads it automatically.

Chosen because the labels are human expert annotations (not LLM-generated), the
task is scored programmatically with no judge model, and the shape generalises:
vendor security questionnaires, claims adjudication, regulatory filing checks,
trial eligibility screening.

### What makes the task hard

> Span [3]: *"...shall not disclose or cause to be disclosed ... any information
> concerning said trade secret or property **to any person, entity, business or
> other individual or company** without the prior written permission..."*

Checklist item `nda-5` is "Receiving Party may share some Confidential Information
with some of Receiving Party's employees." The answer is **`Contradiction`**: the
prohibition is absolute and contains **no employee carve-out**, though most NDAs
do have one.

The model must notice the *absence of an exception*. The dataset authors call this
*negation by exception* and identify it as the core difficulty; the prohibition and
its carve-out are sometimes pages apart. This single distinction is what separates
the models.

---

## Choosing a base model

`config.py` sets `BASE_MODEL_ID = "huggingface-reasoning-qwen3-4b"`. Verified
end-to-end: trains, deploys to a SageMaker endpoint, and imports into Bedrock.

**Verify the deployment path before committing to a base model.** Qwen3.5-4B
trains without error but **cannot be served**: no current LMI/DJL container
recognises model type `qwen3_5` (*"Transformers does not recognize this
architecture"*), including the newest image, and the `cu130` images will not start
on `ml.g5` at all (driver 470 vs the ≤570 expected). Serving its LoRA adapter on
vLLM directly is also blocked, because the adapter targets Qwen3.5's hybrid
`linear_attn` modules.

Interesting aside, measured: fine-tuning Qwen3.5-4B *did* produce a better model
(90.2% accuracy, 81.6 evidence-F1) — a stronger base yields a stronger specialist
where the task has headroom. But it cannot be deployed through either path in this
lab today.

---

## Managed evaluation with a custom scorer

Notebook 3 scores everything locally, which is what makes the frontier comparison
possible: Claude on Bedrock and the tuned model on SageMaker go through identical
code. Notebook 3a shows the other path — registering `contractnli_scorer.py` as an
AI Registry evaluator and letting `CustomScorerEvaluator` run inference and scoring
as a managed pipeline, with results in S3, MLflow and registry lineage.

It emits nine metrics per record via `metrics_list`, and the two agree closely:

| Metric | Managed pipeline | Notebook 3 (local) |
|---|---|---|
| label accuracy | 87.76 | 88.1 |
| evidence F1 | 75.96 | 76.1 |
| JSON valid | 100.0 | 100 |

Measured on the fine-tuned Qwen3-4B, all 123 test contracts, `byoc_failure_count` 0.

Three things to know before using it:

- **It cannot evaluate a Bedrock model.** `CustomScorerEvaluator` accepts only
  `dataset` and `evaluator`; the model under test must be a SageMaker model package.
  `InspectAIEvaluator` does take a `bedrock_model_id`, but scores through Inspect AI
  rather than a custom reward function. Frontier baselines stay in notebook 3.
- **Metrics are per-record averages.** For label accuracy that is not a
  difference — every contract has exactly 17 items, so the per-document mean *is*
  the corpus figure. For evidence F1 it is a real difference: macro (per-contract)
  rather than the micro figure notebook 3 reports.
- **The evaluator type is `REWARD_FUNCTION`.** That name comes from RLVR; using it
  purely for evaluation is fine.

## Gotchas encountered building this lab

- **A custom scorer must read `model_response`, not `response`.** The managed
  pipeline passes each record with `model_response` (the model's answer),
  `reference_answer` (the gold) **and `response` — which is also the gold**, carried
  over from the dataset's `response` column. Reading `response` as the model output
  returns 1.0 on every metric with no error raised. Sanity-check any scorer with a
  deliberately wrong answer before registering it.

Each of these is called out in the relevant notebook.

| | |
|---|---|
| **Optimizer steps, not epochs** | `global_batch_size` is floored at 64. With 423 records that is ~6 steps/epoch, so the default `max_epochs=1` learns nothing. We use 10 epochs (~60 steps). |
| **Overwriting S3 does not refresh a registered DataSet** | Name and path are unchanged, so the registration looks valid. Delete and re-register. |
| **The SDK resolves the control plane from the ambient region** | Not from the `boto3.Session` you pass to `Session(...)`. Mismatched regions fail at `CreateTrainingJob`. |
| **Container pin matters** | Architecture support *and* driver compatibility. See "Choosing a base model". |
| **Endpoint creation is flaky** | Failed transiently 2 of 6 times; succeeded on retry. |
| **Evaluate on a random sample** | The dataset is ordered. `test_docs[:40]` gave 86.5% accuracy versus 88.2% for a seeded random 40 and 88.1% for the full test set. |
| **Bedrock CMI cold start** | ~109 s after idle, then 3.7 s warm. Fine for batch, disqualifying for interactive unless kept warm. |
| **Idle GPUs dominate cost** | On the reference run an idle evaluation instance cost more than **twice** all the fine-tuning combined. |

---

## Cost, measured

| Item | Cost |
|---|---|
| One LoRA fine-tuning job | **~$3**, 34 minutes |
| Managed evaluation run (base + tuned, 123 contracts) | ~30 minutes of managed compute |

Serving cost is deliberately out of scope for this lab. If a customer asks, use the
[AWS Pricing Calculator](https://calculator.aws/) with the measured token counts:
Sonnet 5 uses 4,399 in / 1,023 out per contract, the fine-tuned model 3,126 / 389 —
the tuned model is cheaper per contract largely because it emits only the JSON.

---

## Honest limitations

Read these before presenting the result to a customer.

**It is not a like-for-like comparison.** A model fine-tuned on 423 in-domain
contracts is compared against frontier models that have never seen the checklist.
This is *not* a claim that a 4B model is more capable than Opus. The defensible
claim is about a deployment decision: *given a stable, repetitive review task and
a few hundred labelled examples, adapting a small model beats calling a large one*
— on quality, not merely on cost.

**Possible pretraining contamination.** ContractNLI has been public since October
2021. Two arguments limit the concern: the base model scores only 68.0%, and if
test labels were recoverable from pretraining the frontier models would score far
above 82%. It cannot be fully excluded.

**Near-duplicate contracts.** 6 of 123 test contracts have a near-identical train
twin (5-gram Jaccard ≥ 0.9). Excluding every test contract at J ≥ 0.5 moves the
tuned model from 88.0 → 88.2 accuracy, so it does not explain the result.

**Annotation is the real barrier to reproducing this.** 423 contracts × 17
verdicts *with* evidence spans is ~7,200 expert judgements. The dataset authors
used crowd workers plus expert review at $18.31/hour. For a customer with no
historical review records, annotation dwarfs the ~$3 of training compute. The
realistic customer already has completed checklists in a case management system.
**Quote the $3 as the training cost, never as the project cost.**

**One contract type.** The dataset covers NDAs only. The authors state that
practitioners would need a comparable dataset per contract type; nothing here
tests cross-type generalisation.

**44% of gold answers are "NotMentioned."** Much of what fine-tuning fixed was
learning *when to abstain*. A customer whose checklist items are nearly always
answered has a different prior structure and should expect a smaller gain.

---

## Where this leads

Labels here are exact-match verifiable, so the same dataset drives the RLVR lab.
The natural target is the contradiction class, where SFT plateaus at 71.3 F1, and
the carve-out items the tuned model still finds hardest (`nda-20` at 64%, `nda-7`
at 76%).

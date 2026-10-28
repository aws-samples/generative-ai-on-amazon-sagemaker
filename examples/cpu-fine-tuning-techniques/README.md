# Fine-tuning techniques on CPU: SFT, continued pre-training, DPO and GRPO

Four short labs that run the main LLM customization techniques end to end on **CPU-only SageMaker training jobs**,
so they work in any account without waiting for GPU capacity. Each lab fine-tunes `Qwen/Qwen3-0.6B` (Apache-2.0)
with a LoRA adapter and evaluates base vs fine-tuned in the same job, in about 4-14 minutes per lab.

The models and runs are deliberately small: the labs show what each technique does and how it runs on SageMaker
(ModelTrainer, capacity options, metrics, deployment). The multi-GPU recipes (QLoRA, FSDP, vLLM rollouts) are in
`workshops/distributed-training-deployment-on-sagemaker-ai/`.

| Notebook | Technique | Data | What the job measures |
|---|---|---|---|
| `01-sft.ipynb` | supervised fine-tuning | booking request → strict JSON (generated) | exact JSON match; also compares **on-demand / Spot / fallback** training and **real-time / instance pools / serverless** endpoints |
| `02-continued-pretraining.ipynb` | continued pre-training | raw text about a fictional restaurant | fill-in-the-blank on unseen phrasings, perplexity on held-out sentences |
| `03-dpo.ipynb` | direct preference optimization | the booking task as chosen/rejected pairs | exact match, preference accuracy; plain DPO vs DPO + SFT term |
| `04-grpo.ipynb` | RL with a verifiable reward | the booking task with a reward function | exact match, mean reward, reward per step |

All data is generated in code (`scripts/booking_data.py`): nothing to download except the model, no licensing
questions. 01 uploads its dataset to S3 and passes it to the job as input channels, as you would with your own
data; 02-04 generate theirs inside the job. Every job logs to MLflow (experiment `cpu-fine-tuning-techniques`)
when an MLflow App is available. Run 01 first; 02-04 are independent of each other.

## Layout

| Path | What |
|---|---|
| `scripts/booking_data.py` | the tasks: booking requests, restaurant facts and documents, preference pairs, reward function |
| `scripts/train.py` | SFT job (01) |
| `scripts/train_cpt.py`, `train_dpo.py`, `train_grpo.py` | CPT, DPO, GRPO jobs (02-04); `common.py` holds what they share |
| `scripts/requirements.txt`, `requirements-trl.txt` | job dependencies (DPO/GRPO use TRL, same versions as the GPU workshop) |
| `sm_helpers.py` | notebook helpers for 02-04: session, MLflow, a CPU job in one of the capacity modes, waiting, metrics |
| `workshop_mlflow.py` | finds or creates the MLflow App (copied from the serverless customization workshop) |
| `serving/inference.py` | CPU endpoint handler (01): base model from the Hub + adapter, merged at startup |

## Measured (us-east-1, `ml.c7i.4xlarge`)

| Lab | Result (30 held-out examples unless noted) | Job time |
|---|---|---|
| 01 SFT, 20 steps | exact JSON 0% → 100% | ~4 min (Spot: 57 s billed vs ~180 s on-demand) |
| 02 CPT, 60 steps | questions answered 0% → 100% (8 questions, unseen phrasings), held-out perplexity 148 → 8.4 | 5.0 min |
| 03 DPO, 40 steps | DPO + SFT: exact 0% → 90%, prefers chosen 53% → 100%. Plain DPO: prefers chosen 100% but exact 0% (it drops JSON altogether) | 9.4 min (both jobs in parallel) |
| 04 GRPO, 40 steps | exact 0% → 40%, mean reward 0.46 → 0.79; name/party/day 100%, 24-hour time 40% (rarely sampled, so slowly learned) | 13.5 min |

Serving (01), same model on three endpoints: real-time and real-time with instance pools ~4.5 s per request,
serverless 46 s cold start then 3.6 s.

## Requirements

* A SageMaker execution role (Studio's role works) and training quota for one of the CPU types in
  `sm_helpers.FALLBACK_TYPES` (types without quota are dropped automatically if the role can read Service Quotas).
  01 also needs a Spot training quota for `ml.c7i.4xlarge` and endpoint quotas for `ml.m5.xlarge`, `ml.m7i.xlarge`,
  `ml.c7i.2xlarge`, with serverless endpoints enabled.
* Internet access from the jobs and endpoints to download `Qwen/Qwen3-0.6B`.
* 01's last cell deletes its endpoints; 02-04 create no endpoints.

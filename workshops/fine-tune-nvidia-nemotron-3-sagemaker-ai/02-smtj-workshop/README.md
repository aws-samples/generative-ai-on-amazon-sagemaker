# Fine-tune NVIDIA Nemotron 3 Nano 4B with SageMaker Training jobs

Fine-tune `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` with LoRA to review non-disclosure agreements:
for each of 17 checklist items, return a verdict and the contract clauses that support it. The
data is [ContractNLI](https://stanfordnlp.github.io/contract-nli/). Training runs as a SageMaker
AI Training job, the model is served from a SageMaker AI real-time endpoint, and evaluation
scores that endpoint.

## Notebooks

Run them in order.

| Notebook | What it does | Time |
| --- | --- | --- |
| `1-prepare-data.ipynb` | Installs dependencies, builds prompt/completion records, uploads train and validation splits to S3, keeps the test split locally | ~1 min |
| `2-fine-tune-llm.ipynb` | Writes the training recipe, launches `scripts/train.py` as a Training job on `ml.g5.2xlarge`, and waits for it | ~35 min |
| `3-deployment.ipynb` | Deploys the merged checkpoint to an `ml.g5.xlarge` endpoint with vLLM and sends a test request | ~11 min |
| `4-evaluation.ipynb` | Scores the endpoint on 123 held-out contracts, compares against the untuned model and a frontier model, runs an LLM-as-a-Judge evaluation, and deletes the endpoint | seconds to ~25 min |

Notebook 4 has two switches, both `False` by default, which read pre-computed results from
`baselines/`:

| Switch | When `True` | Time |
| --- | --- | --- |
| `RUN_ENDPOINT_EVAL` | scores all 123 contracts against your endpoint | ~3 min |
| `RUN_JUDGE_EVAL` | runs two Amazon Bedrock evaluation jobs over 30 contracts | ~20 min |

## Results

Reference run, 123 held-out contracts:

| Model | Accuracy | Evidence F1 | Contradiction F1 | Valid JSON |
| --- | --- | --- | --- | --- |
| Nemotron 3 Nano 4B, untuned | 13.6 | 7.8 | 3.8 | 30.1 |
| Nemotron 3 Nano 4B, fine-tuned | 87.9 | 76.6 | 70.1 | 100.0 |
| Claude Sonnet 5 | 84.1 | 68.6 | 62.3 | 100.0 |

## Prerequisites

- A SageMaker Studio JupyterLab space on SageMaker Distribution 4.5.0. `requirements.txt` only
  adds `transformers`; the SageMaker Python SDK, boto3, pandas and matplotlib come with the image.
- Quota for one `ml.g5.2xlarge` Training job and one `ml.g5.xlarge` endpoint.
- Optional: a SageMaker MLflow App. Notebook 2 logs the run to it when one exists, and the
  execution role needs `sagemaker-mlflow:*`.
- For `RUN_JUDGE_EVAL = True`: Bedrock access to the
  `us.anthropic.claude-sonnet-4-5-20250929-v1:0` inference profile, and an execution role that
  trusts `bedrock.amazonaws.com` and allows `bedrock:CreateEvaluationJob`, `GetEvaluationJob`,
  `ListEvaluationJobs`, `StopEvaluationJob` and `bedrock:InvokeModel`.

## Files

| Path | Purpose |
| --- | --- |
| `config.py` | Base model, S3 data prefix and job name prefix shared by the notebooks |
| `contractnli.py` | Dataset download, prompt template and record builder |
| `contractnli_scorer.py` | Scores a model answer against the annotated reference |
| `scripts/train.py`, `scripts/requirements.txt` | Training script and the dependencies installed inside the Training job |
| `baselines/` | Pre-computed evaluation results read by notebook 4 |

## Clean up

The endpoint bills per instance-hour until it is deleted. The last cell of notebook 4 deletes the
endpoint, endpoint config and model. If you stop after notebook 3, run that cell anyway. The
Training job stops on its own; its artifact and checkpoints stay in your S3 bucket.

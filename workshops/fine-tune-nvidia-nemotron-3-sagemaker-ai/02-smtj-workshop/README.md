# Fine-tune NVIDIA Nemotron 3 Nano 4B with SageMaker Training jobs

Fine-tune `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` with LoRA to review non-disclosure agreements:
for each of 17 checklist items, return a verdict and the contract clauses that support it. The
data is [ContractNLI](https://stanfordnlp.github.io/contract-nli/). Training runs as a SageMaker
AI Training job, the model is served from a SageMaker AI real-time endpoint, and evaluation
scores that endpoint.

## Notebooks

Run them in order.

| Notebook                | What it does                                                                                                                                                        | Time               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| `1-prepare-data.ipynb`  | Installs dependencies, builds prompt/completion records, uploads train and validation splits to S3, keeps the test split locally                                    | ~1 min             |
| `2-fine-tune-llm.ipynb` | Writes the training recipe, launches `scripts/train.py` as a Training job (first choice `ml.g5.xlarge`), and waits for it                                          | ~35 min            |
| `3-deployment.ipynb`    | Deploys the merged checkpoint to an endpoint (first choice `ml.g5.xlarge`) with vLLM and sends a test request                                                        | ~11 min            |
| `4-evaluation.ipynb`    | Scores the endpoint on 123 held-out contracts, compares against the untuned model and a frontier model, runs an LLM-as-a-Judge evaluation, and deletes the endpoint | seconds to ~25 min |

Notebook 4 has two switches, both `False` by default, which read pre-computed results from
`baselines/`:

| Switch              | When `True`                                               | Time    |
| ------------------- | --------------------------------------------------------- | ------- |
| `RUN_ENDPOINT_EVAL` | scores all 123 contracts against your endpoint            | ~3 min  |
| `RUN_JUDGE_EVAL`    | runs two Amazon Bedrock evaluation jobs over 30 contracts | ~20 min |

## Results

Reference run, 123 held-out contracts:

| Model                          | Accuracy | Evidence F1 | Contradiction correctness | Valid JSON |
| ------------------------------ | -------- | ----------- | ---------------- | ---------- |
| Nemotron 3 Nano 4B, untuned    | 13.6     | 7.8         | 3.8              | 30.1       |
| Nemotron 3 Nano 4B, fine-tuned | 87.9     | 76.6        | 70.1             | 100.0      |
| Claude Sonnet 5                | 84.1     | 68.6        | 62.3             | 100.0      |

## Evaluation notebook structure

The evaluation notebook imports [evaluation_utils.py](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/evaluation_utils.py) as `E`. It delegates endpoint requests, gold construction, score collection/summaries, baseline loading, judge rubrics/input uploads/result reading, charts, and diagnostics. The separate `contractnli_scorer.py` retains parsing and per-contract equations. Core Bedrock submission and execution choices remain visible in the notebook. Keep the helper and notebook together; importing the helper creates no clients or jobs.

`contradiction_correct` is the mean per-contract fraction of gold contradictions predicted correctly, including zero for contracts with none, not F1. Evidence F1 is averaged per contract. `json_valid` means tolerant object extraction, not complete checklist/schema validation. Reference tables are historical; neither switch regenerates the base/frontier statistical baselines. Live quality and judge results remain in memory, with judge inputs/results in S3; there is no automatic local results export.

The endpoint and judge switches default independently to `False`. `RUN_PERFORMANCE_BENCHMARK` also defaults to `False` and uses the separate streaming helper; cost calculations require a user-supplied regional SageMaker rate. Setup still accesses AWS. Cleanup is unguarded by these switches and attempts hosting deletion even in reference-only mode; inspect it before Run All.

Before live judging, inspect the utility's rubric and dataset builder: the judge input currently omits the system checklist, and the carve-out rubric has conflicting silence/exception wording. Judge means exclude null scores; the table reports coverage but the chart does not. Retain raw output IDs for paired comparisons.

## Prerequisites

- A SageMaker Studio JupyterLab space on SageMaker Distribution 4.5.0 with Python 3.12. Install `requirements.txt` and restart the kernel. It uses `../constraints-notebook.txt` to pin the SDK modules and adds the tokenizer dependency. See the [workshop dependency notes](../README.md#notebook-dependencies) for validation limits and the native Studio smoke test.
- Training job quota of 1 for each of `ml.g5.xlarge`, `ml.g5.2xlarge`, `ml.g6.2xlarge`, `ml.g6e.2xlarge` and `ml.g6.xlarge`. The job lists all five as instance preferences and is rejected if any of them has no quota.
- Endpoint quota of 1 for the endpoint instance pools: `ml.g5.xlarge`, `ml.g6.xlarge`, `ml.g6.2xlarge` and `ml.g6e.xlarge`.
- MLflow App permissions as described in [MLflow setup](../README.md#monitor-training-and-mlflow). `MLFLOW_APP_ARN` is optional and always wins. If empty, notebook 2 first reuses the uniquely matching tagged stack app, ahead of any named duplicate. Only if none matches does it reuse `MLFLOW_APP_NAME` in the current account/Region or create it. Ambiguity stops setup; pending apps are waited for. Creation uses the notebook execution role and bucket's `mlflow-artifacts` prefix unless overridden by `MLFLOW_ROLE_ARN` and `MLFLOW_ARTIFACT_STORE_URI`. No domain/tracking server is required and no defaults are changed. Notebook 2 prints the app ARN, link and experiment; metrics appear after training startup.
- For `RUN_JUDGE_EVAL = True`: Bedrock access to the
  `us.anthropic.claude-sonnet-4-5-20250929-v1:0` inference profile, and an execution role that
  trusts `bedrock.amazonaws.com` and allows `bedrock:CreateEvaluationJob`, `GetEvaluationJob`,
  `ListEvaluationJobs`, `StopEvaluationJob` and `bedrock:InvokeModel`.

## Files

| Path                                           | Purpose                                                                |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| `config.py`                                    | Base model, S3 data prefix and job name prefix shared by the notebooks |
| `contractnli.py`                               | Dataset download, prompt template and record builder                   |
| `contractnli_scorer.py`                        | Scores a model answer against the annotated reference                  |
| `scripts/train.py`, `scripts/requirements.txt` | Training script and the dependencies installed inside the Training job |
| `baselines/`                                   | Pre-computed evaluation results read by notebook 4                     |
| `evaluation_utils.py` | Endpoint scoring orchestration, baselines, judge input/results, plots and diagnostics |
| `inference_performance.py` | Optional streaming latency, TTFT, final-usage token rates and compute allocation |

## Clean up

The endpoint bills per instance-hour until it is deleted. The last cell of notebook 4 deletes the
endpoint, endpoint config and model. If you stop after notebook 3, run that cell anyway. The
Training job stops on its own; its artifact and checkpoints stay in your S3 bucket.

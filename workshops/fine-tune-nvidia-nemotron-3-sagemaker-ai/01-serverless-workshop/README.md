# Fine-tune NVIDIA Nemotron 3 Nano 30B-A3B with SageMaker AI serverless model customization

Fine-tune `huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16` with LoRA to review
non-disclosure agreements: for each of 17 checklist items, return a verdict and the contract
clauses that support it. The data is [ContractNLI](https://stanfordnlp.github.io/contract-nli/).
Training runs as serverless model customization, so there is no training instance to choose or
manage. The result is registered as a Model Package and served from a SageMaker AI real-time
endpoint.

## Notebooks

Run them in order.

| Notebook | What it does |
| --- | --- |
| `1-prepare-data.ipynb` | Installs dependencies, builds prompt/completion training records and query/response test records, uploads them to S3 and registers them as SageMaker AI datasets |
| `2-fine-tune-llm.ipynb` | Runs serverless LoRA fine-tuning with `SFTTrainer` and registers the result in a Model Package Group |
| `3-evaluation.ipynb` | Scores the base and fine-tuned models with a custom scorer, runs a Claude Sonnet 4.6 baseline on Amazon Bedrock, and runs an LLM-as-a-Judge evaluation |
| `4-deployment.ipynb` | Deploys the fine-tuned model to an `ml.g5.12xlarge` endpoint with vLLM and an inference component, tests streaming inference, and deletes the endpoint |

Wait for each training and evaluation job to finish before running the next notebook.

## Prerequisites

- A SageMaker Studio JupyterLab space. Notebook 1 installs `requirements.txt`, including the
  SageMaker Python SDK v3.
- Access to serverless model customization and evaluation in your Region, and quota for one
  `ml.g5.12xlarge` endpoint.
- Bedrock access to `global.anthropic.claude-sonnet-4-6` for the baseline and
  `amazon.nova-pro-v1:0` for the judge.
- An execution role that trusts `bedrock.amazonaws.com` and allows `bedrock:CreateEvaluationJob`,
  `GetEvaluationJob`, `ListEvaluationJobs`, `StopEvaluationJob` and `bedrock:InvokeModel` on the
  judge model.

## Files

| Path | Purpose |
| --- | --- |
| `config.py` | Base model and dataset prefix shared by the notebooks |
| `contractnli.py` | Dataset download, prompt template and record builder |
| `contractnli_scorer.py` | Scores a model answer against the annotated reference |

## Clean up

The endpoint bills per instance-hour until it is deleted. The last cells of notebook 4 delete the
inference component, endpoint, endpoint config and model. Datasets, Model Packages and S3
objects remain until you delete them.

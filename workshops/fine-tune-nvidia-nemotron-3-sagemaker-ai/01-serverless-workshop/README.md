# Fine-tune NVIDIA Nemotron 3 Nano 30B-A3B with SageMaker AI serverless model customization

Fine-tune `huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16` with LoRA to review
non-disclosure agreements: for each of 17 checklist items, return a verdict and the contract
clauses that support it. The data is [ContractNLI](https://stanfordnlp.github.io/contract-nli/).
Training runs as serverless model customization, so there is no training instance to choose or
manage. The result is registered as a Model Package and served from a SageMaker AI real-time
endpoint.

## Notebooks

Run them in order.

| Notebook                | What it does                                                                                                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1-prepare-data.ipynb`  | Installs dependencies, builds prompt/completion training records and query/response test records, uploads them to S3 and registers them as SageMaker AI datasets |
| `2-fine-tune-llm.ipynb` | Runs serverless LoRA fine-tuning with `SFTTrainer` and registers the result in a Model Package Group                                                             |
| `3-evaluation.ipynb`    | Scores the base and fine-tuned models with a custom scorer, runs a Claude Sonnet 4.6 baseline on Amazon Bedrock, and runs an LLM-as-a-Judge evaluation           |
| `4-deployment.ipynb`    | Deploys the fine-tuned model to an `ml.g5.12xlarge` endpoint with vLLM and an inference component, sends a non-streaming smoke test, and deletes the endpoint           |

Wait for each training and evaluation job to finish before running the next notebook.

## Evaluation notebook structure

The evaluation notebook imports [evaluation_utils.py](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/01-serverless-workshop/evaluation_utils.py) as `E`. It delegates frontier Converse requests/retries, local parsing/scoring, sanity checks, judge rubrics, polling, charts, and the `EvaluationResults` S3 readers/exports. The managed evaluator constructors and `.evaluate()` submissions remain in the notebook. `contractnli_scorer.py` remains the separate self-contained scorer upload. Keep the helper and notebook together; importing the helper creates no clients or jobs, but this notebook has no reference-only run switch.

Local frontier evidence F1 pools spans across contracts; managed evidence F1 averages per-contract scores. Local `contradiction_f1` is not managed `contradiction_correct`. Parseability does not validate checklist completeness. Inspect the actual configured identities and generated results, not the helper's historical comparison printout. Verify execution provenance because result lookup can fall back to other matching runs, and review the carve-out rubric's conflicting wording before live judging.

Local exports overwrite `./eval_result/frontier_metrics.json`, `frontier_rows.json`, `scorer_base.json`, `scorer_tuned.json`, `summary.json`, `judge_base.json`, and `judge_tuned.json`. Preserve package/execution ARNs and raw S3 records separately; the snapshots are not a complete provenance manifest.

## Prerequisites

- A SageMaker Studio JupyterLab space with Python 3.12. Notebook 1 installs `requirements.txt`, using `../constraints-notebook.txt` to pin SageMaker SDK 3.21.0 and its modules. Restart the kernel and follow the [dependency validation notes](../README.md#notebook-dependencies).
- MLflow App permissions as described in [MLflow setup](../README.md#monitor-training-and-mlflow). `MLFLOW_APP_ARN` is optional and always wins. If empty, setup first reuses the uniquely matching tagged stack app, then the named workshop app, before falling back to native SDK discovery/create. Ambiguity stops setup; pending apps are waited for before training. It does not require a domain/tracking server or change defaults. Set both `MLFLOW_ROLE_ARN` and `MLFLOW_ARTIFACT_STORE_URI` to use the named workshop reuse/create path instead of SDK creation defaults. Notebook 2 prints the app ARN and links; save that ARN in `config.py` so notebook 3 uses the same app. Select experiment `contractnli-serverless-sft` after training starts.
- Access to serverless model customization and evaluation in your Region, and quota for one
  `ml.g5.12xlarge` endpoint.
- Bedrock access to `global.anthropic.claude-sonnet-4-6` for the baseline and
  `amazon.nova-pro-v1:0` for the judge.
- An execution role that trusts `bedrock.amazonaws.com` and allows `bedrock:CreateEvaluationJob`,
  `GetEvaluationJob`, `ListEvaluationJobs`, `StopEvaluationJob` and `bedrock:InvokeModel` on the
  judge model.

## Files

| Path                    | Purpose                                               |
| ----------------------- | ----------------------------------------------------- |
| `config.py`             | Base model and dataset prefix shared by the notebooks |
| `contractnli.py`        | Dataset download, prompt template and record builder  |
| `contractnli_scorer.py` | Scores a model answer against the annotated reference |
| `evaluation_utils.py` | Supporting evaluation implementations used by the notebook |

## Clean up

The endpoint bills per instance-hour until it is deleted. The last cells of notebook 4 delete the
inference component, endpoint, endpoint config and model. Datasets, Model Packages and S3
objects remain until you delete them.

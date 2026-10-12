# Fine-Tuning NVIDIA Nemotron 3 Nano with Amazon SageMaker AI

Welcome to the **NVIDIA Nemotron 3 Nano fine-tuning workshop**! This hands-on, self-paced workshop explores supervised fine-tuning (SFT) for evidence-based contract review using two alternative SageMaker AI training approaches.

Both tracks use [ContractNLI](https://stanfordnlp.github.io/contract-nli/) to review a non-disclosure agreement (NDA) against the same 17 legal hypotheses. The target is a JSON object containing a `label` (`Entailment`, `Contradiction`, or `NotMentioned`) and an `evidence` list of supporting span numbers for each hypothesis, with an empty list for `NotMentioned`.

Choose **serverless customization of Nemotron 3 Nano 30B-A3B** or **SageMaker Training jobs for Nemotron 3 Nano 4B**. These are alternative SFT tracks, not consecutive labs.

## Learning Objectives

By working through your chosen track, you'll learn to:

- Prepare ContractNLI prompts, reference JSON answers, and held-out data for SFT
- Fine-tune with LoRA using the SageMaker SDK's serverless `SFTTrainer` or `ModelTrainer` with a custom training script
- Distinguish AI Registry datasets and model packages from S3 input channels and Training job artifacts
- Assess label correctness, evidence overlap, contradiction handling, and JSON validity using the serverless evaluation workflow
- Deploy a merged checkpoint on a SageMaker real-time endpoint with vLLM and test inference
- Review service quotas, isolate training data, and clean up billable resources

## Workshop Structure

### Option 1: Serverless Model Customization

Customize the 30B-A3B model without selecting or managing training instances. This track uses registered datasets and model packages for training and managed evaluation, then provisions GPU hosting for inference.

**Model:** NVIDIA Nemotron 3 Nano 30B-A3B BF16 | **Dataset:** ContractNLI | **Training:** SageMaker serverless `SFTTrainer` with LoRA

| Step | Notebook                                                              | Description                                                                                                                                  |
| ---- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | [1-prepare-data.ipynb](01-serverless-workshop/1-prepare-data.ipynb)   | Create string-valued prompt/completion training records and query/response test records, upload to S3, and register datasets in AI Registry. |
| 2    | [2-fine-tune-llm.ipynb](01-serverless-workshop/2-fine-tune-llm.ipynb) | Run serverless LoRA SFT and associate the result with a Model Package Group.                                                                 |
| 3    | [3-evaluation.ipynb](01-serverless-workshop/3-evaluation.ipynb)       | Compare base and fine-tuned models with a custom scorer, run a Bedrock frontier baseline, and evaluate with LLM-as-a-Judge.                  |
| 4    | [4-deployment.ipynb](01-serverless-workshop/4-deployment.ipynb)       | Deploy merged weights from the model package using vLLM and an Inference Component, test streaming inference, and clean up.                  |

**Current notebook settings:** JumpStart model ID `huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16`, 10 training epochs, LoRA rank 32, and learning rate `1e-4`. Hosting uses one `ml.g5.12xlarge` with four GPUs and the vLLM `0.22.0` container.

The custom scorer reports `label_correct`, `evidence_f1`, `contradiction_correct`, and `json_valid`. The judge uses Amazon Nova Pro for `EvidenceSupportsVerdict`, `CarveOutHandling`, and `ChecklistCompleteness`; the separate frontier baseline calls Claude Sonnet 4.6 through Bedrock.

---

### Option 2: SageMaker Training Jobs

Customize the 4B model using `ModelTrainer`, S3 input channels, and a training script running TRL `SFTTrainer` with PEFT LoRA. Deployment reads the merged `model.tar.gz` artifact from a completed Training job rather than from a model package.

**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | **Dataset:** ContractNLI | **Training:** SageMaker Training jobs with BF16 LoRA, not QLoRA

| Step | Notebook                                                        | Description                                                                                                                                                                                                             |
| ---- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | [1-prepare-data.ipynb](02-smtj-workshop/1-prepare-data.ipynb)   | Build chat-based prompt/completion records with `enable_thinking=False`, upload train/validation splits to S3, and retain test records locally.                                                                         |
| 2    | [2-fine-tune-llm.ipynb](02-smtj-workshop/2-fine-tune-llm.ipynb) | Generate the training recipe, launch `scripts/train.py` with `ModelTrainer`, and save merged weights to S3.                                                                                                             |
| 3    | [3-deployment.ipynb](02-smtj-workshop/3-deployment.ipynb)       | Discover a completed Training job, deploy its merged artifact directly with vLLM, and check a response. The endpoint stays up for notebook 4.                                                                           |
| 4    | [4-evaluation.ipynb](02-smtj-workshop/4-evaluation.ipynb)       | Score the endpoint over the 123 held-out contracts with the custom scorer, compare against the untuned model and a frontier baseline, run LLM-as-a-Judge as a managed Bedrock evaluation job, then delete the endpoint. |

Deployment comes before evaluation in this track because evaluation scores the endpoint. Both are
gated by switches that default to reading pre-computed results from `02-smtj-workshop/baselines/`,
so the notebook can be read without running anything:

| Switch              | `False` (default)          | `True`                                         | Time when `True` |
| ------------------- | -------------------------- | ---------------------------------------------- | ---------------- |
| `RUN_ENDPOINT_EVAL` | reads pre-computed results | scores all 123 contracts against your endpoint | ~3 min           |
| `RUN_JUDGE_EVAL`    | reads pre-computed results | runs two Bedrock judge jobs over 30 contracts  | ~20 min          |

The judge reaches Bedrock directly rather than through `LLMAsJudgeEvaluator`, because this track
registers no Model Package. The endpoint's answers are supplied to the evaluation job as
precomputed inference responses, using the same three custom metrics as Option 1.

**Current notebook settings:** One `ml.g5.2xlarge` training instance, PyTorch `2.8.0`, 1 training epoch, maximum sequence length 8,192, LoRA rank 32, and learning rate `1e-4`. Direct hosting uses one `ml.g5.xlarge` and the vLLM `0.28.0` container, without an Inference Component. Notebook 3 selects the newest completed job matching `TRAIN_JOB_PREFIX` unless `TRAINING_JOB_NAME` is set explicitly. Training logs to the account's default SageMaker MLflow App when one exists; without one the job trains untracked.

## Prerequisites

- An AWS account and a SageMaker Studio JupyterLab environment, or an equivalent notebook environment with AWS credentials and an execution role
- A Region supporting your chosen training approach, model, and inference container; verify availability before starting
- IAM permissions for the resources you use: SageMaker training and hosting, S3 data/artifact access, ECR image pulls, CloudWatch logs, and `iam:PassRole` for the execution role
- For Option 1, permissions for AI Registry datasets/evaluators, Model Registry, and the managed evaluation workflow, including its SageMaker Pipelines and Lambda scorer resources
- **Separate quotas:** Option 1 needs serverless customization/evaluation capacity and `ml.g5.12xlarge` endpoint capacity; Option 2 needs `ml.g5.2xlarge` Training job capacity and `ml.g5.xlarge` endpoint capacity. Training and hosting quotas are not interchangeable; check warm-pool capacity if retaining that setting.
- For baseline/judge evaluation, Bedrock invocation access to the frontier model and evaluation access to the judge model, including the applicable model/inference-profile permissions and quotas. Option 1 judges with `amazon.nova-pro-v1:0`; Option 2 judges with the `us.anthropic.claude-sonnet-4-5-20250929-v1:0` inference profile, and its execution role additionally needs a trust relationship for `bedrock.amazonaws.com` plus `bedrock:CreateEvaluationJob`, `GetEvaluationJob`, `ListEvaluationJobs`, `StopEvaluationJob` and `bedrock:InvokeModel`. Neither is required to run Option 2 with both evaluation switches left `False`.
- Network access to download ContractNLI, Python dependencies, and Hugging Face tokenizers/weights; review applicable model terms before training, especially the serverless notebook's `accept_eula=True` setting
- Install the selected directory's `requirements.txt` using notebook 1. Both tracks require SageMaker SDK v3. Option 1 installs `sagemaker>=3.16.0`; Option 2 uses the SDK that ships with the SageMaker Distribution 4.5.0 JupyterLab image and installs only `transformers`, with its training dependencies installed separately from `scripts/requirements.txt` inside the job.

## How to Run

1. Choose one option and open notebooks with that option's directory as the working directory so local imports and relative paths resolve correctly.
2. Review its `config.py`, AWS Region, execution role, S3 bucket/prefix, training settings, and hosting instance type before launching resources.
3. Install dependencies in notebook 1 and restart the kernel if needed, then prepare the data.
4. **Option 1:** Run notebooks **1 → 2 → 3 → 4**, waiting for training and evaluation jobs to finish before consuming their outputs.
5. **Option 2:** Run notebooks **1 → 2 → 3 → 4**, waiting for the Training job to reach `Completed` before deployment.
6. Run the cleanup cells when finished and verify that the resources were deleted. For Option 1 these are in notebook 4; for Option 2 they are at the end of notebook 4, since notebook 3 leaves the endpoint running for evaluation.

**S3 isolation warning:** Both configurations default to `DATASET_PREFIX = "contractnli-nda-review"` and write train/validation objects under the same `datasets/contractnli-nda-review` prefix when using the same bucket and session prefix. Option 1 stores strings, while Option 2 stores chat-message lists plus chat-template arguments, so running both can overwrite data with an incompatible schema. Choose one track, or set distinct dataset prefixes in both configurations before preparing data; Option 2 derives `DATA_PREFIX` from `DATASET_PREFIX`.

**Costs and cleanup:** Option 2 sets `KEEP_ALIVE_SECONDS = 0`, so no training warm pool is retained; a non-zero value keeps the instance warm, and billable, for that many seconds after the job. Real-time endpoints remain billable until deleted, even after closing a notebook or stopping a polling cell. Remove endpoints, endpoint configurations, models, and Option 1 Inference Components using the cleanup cells, and review retained S3 artifacts, logs, registry resources, and any other resources you created. Training, evaluation, Bedrock calls, hosting, and storage can incur charges; this workshop makes no cost or quality guarantees.

## Repository Structure

The tree below shows the workshop source files, excluding downloaded data, temporary files, caches, and generated training configuration.

```text
.
├── README.md
├── 01-serverless-workshop/
│   ├── README.md
│   ├── 1-prepare-data.ipynb
│   ├── 2-fine-tune-llm.ipynb
│   ├── 3-evaluation.ipynb
│   ├── 4-deployment.ipynb
│   ├── config.py
│   ├── contractnli.py
│   ├── contractnli_scorer.py
│   └── requirements.txt
└── 02-smtj-workshop/
    ├── README.md
    ├── 1-prepare-data.ipynb
    ├── 2-fine-tune-llm.ipynb
    ├── 3-deployment.ipynb
    ├── 4-evaluation.ipynb
    ├── baselines/
    ├── config.py
    ├── contractnli.py
    ├── contractnli_scorer.py
    ├── requirements.txt
    └── scripts/
        ├── train.py
        └── requirements.txt
```

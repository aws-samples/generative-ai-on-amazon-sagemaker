# Serverless Model Customization with SageMaker AI

Welcome to the **Serverless Model Customization with SageMaker AI** workshop! This hands-on, self-paced workshop is designed for ML practitioners who want to customize Large Language Models (LLMs) using different fine-tuning techniques — without managing any training infrastructure — and deploy them on AWS.

You will explore the full model customization lifecycle — from dataset preparation and serverless training, to evaluation and real-time deployment — across four progressively advanced labs using the SageMaker AI serverless customization APIs.

## Learning Objectives

By the end of this workshop, you'll be able to:

- Prepare datasets for different model customization techniques (SFT, DPO, RLVR, RLAIF)
- Run serverless fine-tuning jobs using the SageMaker Python SDK v3 trainers (`SFTTrainer`, `DPOTrainer`, `RLVRTrainer`, `RLAIFTrainer`)
- Register datasets, models, and evaluators in the SageMaker AI Registry
- Evaluate fine-tuned models using LLM-as-a-Judge with custom metrics and benchmark evaluations
- Deploy fine-tuned models using DJL LMI with vLLM on SageMaker Real-time Endpoints with Inference Components

## Workshop Structure

### Lab 1: Supervised Fine-Tuning (SFT)

Train a model to answer a fixed 17-point legal checklist about a non-disclosure agreement and cite the clauses that justify each answer.

**Model:** Qwen 3 - 4B | **Dataset:** [ContractNLI](https://stanfordnlp.github.io/contract-nli/) | **Use case:** Contract review with evidence citations

| Step | Notebook                        | Description                                                                                                                                                                                               |
| ---- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `1-prepare-data.ipynb`          | Load the ContractNLI NDA corpus, render the 17-point checklist prompt, upload to S3, and register in AI Registry                                                                                          |
| 2    | `2-fine-tune-llm.ipynb`         | Run serverless SFT with LoRA using `SFTTrainer`, track metrics with MLflow                                                                                                                                |
| 2b   | `2b-import-model-from-s3.ipynb` | _(Alternative to step 2)_ Skip training: copy the published checkpoint from S3 and register it as a Model Package                                                                                         |
| 3    | `3-evaluation.ipynb`            | Score with a custom reward function (label_correct, evidence_f1, contradiction_correct, json_valid), then judge with Claude Sonnet 4.5 (EvidenceSupportsVerdict, CarveOutHandling, ChecklistCompleteness) |
| 4    | `4-deployment.ipynb`            | Deploy with DJL LMI + vLLM on a SageMaker Endpoint and test streaming inference                                                                                                                           |

---

### Lab 2: Direct Preference Optimization (DPO)

Align model outputs to human preferences using chosen/rejected response pairs — without training a separate reward model.

**Model:** Llama 3.2 - 1B Instruct | **Dataset:** [Human-Like DPO Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset) | **Use case:** Human-like conversational tone

| Step | Notebook                            | Description                                                                                                                          |
| ---- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | `1-dpo-prepare-data.ipynb`          | Load the preference dataset, format as prompt/chosen/rejected triplets, upload to S3, and register in AI Registry                    |
| 2    | `2-dpo-trainer.ipynb`               | Run serverless DPO training with LoRA using `DPOTrainer`, track metrics with MLflow                                                  |
| 2b   | `2b-dpo-import-model-from-s3.ipynb` | _(Alternative to step 2)_ Skip training: copy the published checkpoint from S3 and register it as a Model Package                    |
| 3    | `3-dpo-evaluation.ipynb`            | Evaluate with LLM-as-a-Judge (Claude Sonnet 4.5) using custom metrics: HumanLikeTone, ConversationalEngagement, AvoidRoboticPatterns |
| 4    | `4-dpo-deployment.ipynb`            | Deploy with DJL LMI + vLLM on a SageMaker Endpoint and test streaming inference                                                      |

---

### Lab 3a: Reinforcement Learning from Verifiable Rewards (RLVR)

Fine-tune a model using rule-based reward signals from tasks with objectively verifiable answers — no human annotators or reward model needed.

**Model:** Qwen 3 - 0.6B | **Dataset:** [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | **Use case:** Mathematical reasoning

| Step | Notebook                        | Description                                                                                                       |
| ---- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 1    | `1-prepare-data.ipynb`          | Transform GSM8K math problems into RLVR format with rule-based reward verification, upload to S3                  |
| 2    | `2-trainer.ipynb`               | Run serverless RLVR training using `RLVRTrainer`, track metrics with MLflow                                       |
| 2b   | `2b-import-model-from-s3.ipynb` | _(Alternative to step 2)_ Skip training: copy the published checkpoint from S3 and register it as a Model Package |
| 3    | `3-evaluation.ipynb`            | Evaluate with the MATH benchmark using `BenchMarkEvaluator`, compare fine-tuned vs. base model                    |
| 4    | `4-deployment.ipynb`            | Deploy with DJL LMI + vLLM on a SageMaker Endpoint and test streaming inference                                   |

---

### Lab 3b: RLVR with a Custom Reward Function (Optional Extension)

This optional extension builds on Lab 3a. It reuses the same GSM8K dataset and RLVR workflow, but replaces the built-in rule verifier with a custom Python reward function registered as a SageMaker AI Registry Evaluator.

**Model:** Qwen 3 - 0.6B Instruct | **Dataset:** [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | **Use case:** Custom reward design for mathematical reasoning

| Step | Notebook                                    | Description                                                                                                       |
| ---- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 1    | `1-prepare-data.ipynb`                      | Prepare GSM8K records with stable IDs and reference answers for custom reward scoring                             |
| 2    | `2-create-reward-function.ipynb`            | Test and register a Lambda-style Python reward function as a `REWARD_FUNCTION` Evaluator in AI Registry           |
| 3    | `3-train-with-custom-reward-function.ipynb` | Run serverless RLVR training using `RLVRTrainer` with `custom_reward_function`                                    |
| 3b   | `3b-import-model-from-s3.ipynb`             | _(Alternative to step 3)_ Skip training: copy the published checkpoint from S3 and register it as a Model Package |
| 4    | `4-evaluation.ipynb`                        | Evaluate the custom-reward fine-tuned model with benchmark evaluation                                             |
| 5    | `5-deployment.ipynb`                        | Deploy with DJL LMI + vLLM on a SageMaker Endpoint and test streaming inference                                   |

---

### Lab 4: Reinforcement Learning from AI Feedback (RLAIF)

Use an AI judge model to provide reward signals during training, enabling preference optimization on tasks without verifiable answers.

**Model:** Qwen 2.5 - 7B Instruct | **Dataset:** [Human-Like DPO Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset) | **Use case:** Human-like conversational tone

| Step | Notebook                        | Description                                                                                                       |
| ---- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 1    | `1-prepare-data.ipynb`          | Format the dataset in VERL post-training format with LLM-judge reward style, upload to S3                         |
| 2    | `2-prepare-reward-model.ipynb`  | Create a reward prompt for human-likeness scoring and register it as an Evaluator in AI Registry                  |
| 3    | `3-fine-tune-model.ipynb`       | Run serverless RLAIF training using `RLAIFTrainer` with a GPT-based reward model                                  |
| 3b   | `3b-import-model-from-s3.ipynb` | _(Alternative to step 3)_ Skip training: copy the published checkpoint from S3 and register it as a Model Package |
| 4    | `4-evaluation.ipynb`            | Run dual LLM-as-a-Judge evaluation (base vs. fine-tuned) with a custom human-like-alignment metric                |
| 5    | `5-deployment.ipynb`            | Deploy with DJL LMI + vLLM on a SageMaker Endpoint and test streaming inference                                   |

## Prerequisites

- An AWS account with access to Amazon SageMaker AI
- A SageMaker Studio domain with JupyterLab or Code Editor
- IAM permissions for SageMaker Training, Endpoints, Model Registry, and S3
- Service quota for `ml.g5.xlarge` (all labs) for inference
- Access to Claude Sonnet 4.5 in Amazon Bedrock, the LLM-as-a-Judge evaluator model in labs 1, 2 and 4. Lab 1 also calls Claude Sonnet 4.6 for its frontier baseline

## MLflow automation in every lab

Training and live managed evaluations use the corrected standalone MLflow App resolver
from the Nemotron workshop. Each lab contains a local `config.py` and `workshop_mlflow.py`
so the lab also works when checked out independently.

1. A configured `MLFLOW_APP_ARN` always wins; invalid or inaccessible overrides fail.
2. With no override, discover the uniquely matching tagged CloudFormation app before
   considering a named workshop app or native SDK fallback. The tag/name, account,
   and Region checks are preserved. Ambiguous or failed discovery stops setup.
3. Wait for the app to be `Created` or `Updated` before constructing a trainer or
   live evaluator, then pass the resolved `mlflow_resource_arn` explicitly.
4. Optional `MLFLOW_ROLE_ARN` and `MLFLOW_ARTIFACT_STORE_URI` configure discovery
   scope and named creation. No event identifiers or home-directory handoff files
   are used. The helper does not request an SDK upgrade of shared defaults.

| Lab                | Training experiment            |
| ------------------ | ------------------------------ |
| SFT                | `contractnli-serverless-sft`   |
| DPO                | `humanlike-llama3-2-1b-dpo`    |
| Built-in RLVR      | `qwen3-06b-rlvr`               |
| Custom-reward RLVR | `qwen3-06b-custom-reward-rlvr` |
| RLAIF              | `humanlike-rlaif-qwen25-7b`    |

Training notebooks print the exact job name, console/log links, and a temporary MLflow
app link. Open it within five minutes and select the printed experiment. Rerun only the
link cell when it expires. Metrics appear after training startup, not immediately after
submission.

DPO, built-in RLVR, custom-reward RLVR, and RLAIF evaluation notebooks resolve MLflow
inside their `RUN_EVALUATIONS` live branch. Their existing saved-result switches and
saved-result loading remain available. The SFT evaluation notebook prepares its app in
setup and uses it for both managed evaluators.

The caller needs app discovery/readiness permissions (`DescribeMlflowApp`,
`ListMlflowApps`, `ListTags`) and `CreatePresignedMlflowAppUrl` for links. Creation
fallback also needs `CreateMlflowApp`, role passing and the artifact-store permissions
described by the configured SDK path. The app role must trust SageMaker and access its
S3/KMS artifacts; training/evaluation roles need MLflow tracking permissions.

Run notebooks from their own lab directory and keep the local helper/config with them.
For reproducible app selection across kernels in a shared account, set the same explicit
app ARN in the labs you use. No app, job, or quota is created by editing these files.

## How to Run

This workshop follows a hands-on, self-paced format. Each lab is independent — you can start with any lab that matches your interest.

- **Lab 1 (SFT)** is the best starting point: covers the core workflow of data prep, training, evaluation, and deployment
- **Lab 2 (DPO)** introduces preference-based alignment without a reward model
- **Lab 3a (RLVR)** demonstrates reinforcement learning with rule-based rewards for tasks with verifiable answers
- **Lab 3b (Custom Reward RLVR)** is an optional Lab 3a extension that shows how to register and use your own Python reward function evaluator
- **Lab 4 (RLAIF)** is the most advanced: combines RL training with an AI judge for subjective tasks

Within each lab, run the notebooks in order (1 through 4 or 5). Lab 3b is optional and can be run after Lab 3a, or independently if you want to focus on custom reward functions.

Each notebook contains:

- Step-by-step instructions and explanations
- Code samples that you can run and modify
- Cleanup cells to delete deployed resources

## Repository Structure

```
.
├── lab-1-supervised-fine-tuning/
│   ├── 1-prepare-data.ipynb
│   ├── 2-fine-tune-llm.ipynb
│   ├── 2b-import-model-from-s3.ipynb
│   ├── 3-evaluation.ipynb
│   ├── 4-deployment.ipynb
│   ├── 4a-deployment-bedrock.ipynb
│   ├── config.py
│   ├── contractnli.py
│   ├── contractnli_scorer.py
│   ├── workshop_mlflow.py
│   └── requirements.txt
├── lab-2-direct-preference-optimization-DPO/
│   ├── 1-dpo-prepare-data.ipynb
│   ├── 2-dpo-trainer.ipynb
│   ├── 2b-dpo-import-model-from-s3.ipynb
│   ├── 3-dpo-evaluation.ipynb
│   ├── 4-dpo-deployment.ipynb
│   ├── config.py
│   ├── workshop_mlflow.py
│   └── requirements.txt
├── lab-3a-reinforcement-learning-from-verifiable-rewards/
│   ├── 1-prepare-data.ipynb
│   ├── 2-trainer.ipynb
│   ├── 2b-import-model-from-s3.ipynb
│   ├── 3-evaluation.ipynb
│   ├── 4-deployment.ipynb
│   ├── config.py
│   ├── workshop_mlflow.py
│   └── requirements.txt
├── lab-3b-custom-reward-function-rlvr/
│   ├── 1-prepare-data.ipynb
│   ├── 2-create-reward-function.ipynb
│   ├── 3-train-with-custom-reward-function.ipynb
│   ├── 3b-import-model-from-s3.ipynb
│   ├── 4-evaluation.ipynb
│   ├── 5-deployment.ipynb
│   ├── reward_function.py
│   ├── config.py
│   ├── workshop_mlflow.py
│   └── requirements.txt
└── lab-4-reinforcement-learning-from-ai-feedback/
    ├── 1-prepare-data.ipynb
    ├── 2-prepare-reward-model.ipynb
    ├── 3-fine-tune-model.ipynb
    ├── 3b-import-model-from-s3.ipynb
    ├── 4-evaluation.ipynb
    ├── 5-deployment.ipynb
    ├── config.py
    ├── workshop_mlflow.py
    └── requirements.txt
```

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

Train a model on instruction-response pairs to specialize it for medical reasoning with chain-of-thought outputs.

**Model:** Qwen 2.5 - 7B Instruct | **Dataset:** [Medical O1 Reasoning SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) | **Use case:** Medical expert reasoning

| Step | Notebook              | Description                                                                                                   |
| ---- | --------------------- | ------------------------------------------------------------------------------------------------------------- |
| 1    | `1-prepare-data.ipynb` | Load the medical reasoning dataset, format as prompt/completion pairs, upload to S3, and register in AI Registry |
| 2    | `2-fine-tune-llm.ipynb` | Run serverless SFT with LoRA using `SFTTrainer`, track metrics with MLflow                                    |
| 3    | `3-evaluation.ipynb`   | Evaluate with LLM-as-a-Judge (Amazon Nova Pro) using custom metrics: MedicalReasoningQuality, ClinicalAccuracy, ThinkTagStructure |
| 4    | `4-deployment.ipynb`   | Deploy with DJL LMI + vLLM on a SageMaker Endpoint and test streaming inference                               |

---

### Lab 2: Direct Preference Optimization (DPO)

Align model outputs to human preferences using chosen/rejected response pairs — without training a separate reward model.

**Model:** Llama 3.2 - 1B Instruct | **Dataset:** [Human-Like DPO Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset) | **Use case:** Human-like conversational tone

| Step | Notebook                  | Description                                                                                                  |
| ---- | ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 1    | `1-dpo-prepare-data.ipynb` | Load the preference dataset, format as prompt/chosen/rejected triplets, upload to S3, and register in AI Registry |
| 2    | `2-dpo-trainer.ipynb`      | Run serverless DPO training with LoRA using `DPOTrainer`, track metrics with MLflow                          |
| 3    | `3-dpo-evaluation.ipynb`   | Evaluate with LLM-as-a-Judge (Amazon Nova Pro) using custom metrics: HumanLikeTone, ConversationalEngagement, AvoidRoboticPatterns |
| 4    | `4-dpo-deployment.ipynb`   | Deploy with DJL LMI + vLLM on a SageMaker Endpoint and test streaming inference                              |

---

### Lab 3: Reinforcement Learning from Verifiable Rewards (RLVR)

Fine-tune a model using rule-based reward signals from tasks with objectively verifiable answers — no human annotators or reward model needed.

**Model:** Qwen 2.5 - 7B Instruct | **Dataset:** [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | **Use case:** Mathematical reasoning

| Step | Notebook              | Description                                                                                      |
| ---- | --------------------- | ------------------------------------------------------------------------------------------------ |
| 1    | `1-prepare-data.ipynb` | Transform GSM8K math problems into RLVR format with rule-based reward verification, upload to S3 |
| 2    | `2-trainer.ipynb`      | Run serverless RLVR training using `RLVRTrainer`, track metrics with MLflow                      |
| 3    | `3-evaluation.ipynb`   | Evaluate with the MATH benchmark using `BenchMarkEvaluator`, compare fine-tuned vs. base model    |
| 4    | `4-deployment.ipynb`   | Deploy with DJL LMI + vLLM on a SageMaker Endpoint and test streaming inference                  |

---

### Lab 4: Reinforcement Learning from AI Feedback (RLAIF)

Use an AI judge model to provide reward signals during training, enabling preference optimization on tasks without verifiable answers.

**Model:** Qwen 2.5 - 7B Instruct | **Dataset:** [Human-Like DPO Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset) | **Use case:** Human-like conversational tone

| Step | Notebook                      | Description                                                                                     |
| ---- | ----------------------------- | ----------------------------------------------------------------------------------------------- |
| 1    | `1-prepare-data.ipynb`         | Format the dataset in VERL post-training format with LLM-judge reward style, upload to S3        |
| 2    | `2-prepare-reward-model.ipynb` | Create a reward prompt for human-likeness scoring and register it as an Evaluator in AI Registry |
| 3    | `3-fine-tune-model.ipynb`      | Run serverless RLAIF training using `RLAIFTrainer` with a GPT-based reward model                 |
| 4    | `4-evaluation.ipynb`           | Run dual LLM-as-a-Judge evaluation (base vs. fine-tuned) with a custom human-like-alignment metric |
| 5    | `5-deployment.ipynb`           | Deploy with DJL LMI + vLLM on a SageMaker Endpoint and test streaming inference                  |

## Prerequisites

- An AWS account with access to Amazon SageMaker AI
- A SageMaker Studio domain with JupyterLab or Code Editor
- IAM permissions for SageMaker Training, Endpoints, Model Registry, and S3
- Service quota for `ml.g5.2xlarge` (Labs 1, 3, 4) and `ml.g5.12xlarge` (Lab 2) for inference
- Access to Amazon Nova Pro (used as the LLM-as-a-Judge evaluator model)

## How to Run

This workshop follows a hands-on, self-paced format. Each lab is independent — you can start with any lab that matches your interest.

- **Lab 1 (SFT)** is the best starting point: covers the core workflow of data prep, training, evaluation, and deployment
- **Lab 2 (DPO)** introduces preference-based alignment without a reward model
- **Lab 3 (RLVR)** demonstrates reinforcement learning with rule-based rewards for tasks with verifiable answers
- **Lab 4 (RLAIF)** is the most advanced: combines RL training with an AI judge for subjective tasks

Within each lab, run the notebooks in order (1 through 4 or 5) — each notebook builds on the resources created by the previous one.

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
│   ├── 3-evaluation.ipynb
│   ├── 4-deployment.ipynb
│   └── requirements.txt
├── lab-2-direct-preference-optimization-DPO/
│   ├── 1-dpo-prepare-data.ipynb
│   ├── 2-dpo-trainer.ipynb
│   ├── 3-dpo-evaluation.ipynb
│   ├── 4-dpo-deployment.ipynb
│   └── requirements.txt
├── lab-3-reinforcement-learning-from-verifiable-rewards/
│   ├── 1-prepare-data.ipynb
│   ├── 2-trainer.ipynb
│   ├── 3-evaluation.ipynb
│   ├── 4-deployment.ipynb
│   └── requirements.txt
└── lab-4-reinforcement-learning-from-ai-feedback/
    ├── 1-prepare-data.ipynb
    ├── 2-prepare-reward-model.ipynb
    ├── 3-fine-tune-model.ipynb
    ├── 4-evaluation.ipynb
    ├── 5-deployment.ipynb
    └── requirements.txt
```

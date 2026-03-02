# Amazon Nova Customization on SageMaker AI

Welcome to the **Amazon Nova Customization on SageMaker AI** workshop! This hands-on workshop is designed for ML practitioners who want to customize Amazon Nova models using SageMaker Training jobs and deploy them on Amazon Bedrock.

You will explore the full model customization lifecycle — from dataset preparation and fine-tuning with the Amazon Nova Customization SDK, to deployment on Bedrock and evaluation using LLM-as-a-Judge with Amazon Nova Pro.

## Learning Objectives

By the end of this workshop, you'll be able to:

- Prepare datasets for Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) using the Amazon Nova Converse format
- Fine-tune Amazon Nova models with LoRA using the Amazon Nova Customization SDK on SageMaker Training jobs
- Deploy fine-tuned models on Amazon Bedrock using custom model on-demand inference
- Evaluate model quality using LLM-as-a-Judge with Amazon Nova Pro

## Workshop Structure

### Nova 1.0

Customize **Amazon Nova Micro 1.0** with SFT and DPO using the Amazon Nova Customization SDK.

#### Lab 1 — Supervised Fine-Tuning (SFT) with LoRA

Train the model on multilingual reasoning data to reason in a specified language and respond in English.

| Step | Notebook               | Description                                                                              |
| ---- | ---------------------- | ---------------------------------------------------------------------------------------- |
| 1    | `1-prepare-data.ipynb` | Format the Multilingual-Thinking dataset with `<think>` tags and upload to S3            |
| 2    | `2-fine-tune.ipynb`    | Fine-tune Nova Micro with SFT and LoRA using the Nova Customization SDK                  |
| 3    | `3-deploy.ipynb`       | Import to Bedrock and deploy for on-demand inference                                     |
| 4    | `4-evaluate.ipynb`     | Evaluate with LLM-as-a-Judge (Amazon Nova Pro) comparing base vs fine-tuned              |

**Model:** Amazon Nova Micro 1.0 | **Dataset:** [HuggingFaceH4/Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking) | **Use case:** Multilingual chain-of-thought reasoning

#### Lab 2 — Direct Preference Optimization (DPO) with LoRA

Align model outputs to prefer correct tool calls over incorrect ones using chosen/rejected response pairs.

| Step | Notebook               | Description                                                                              |
| ---- | ---------------------- | ---------------------------------------------------------------------------------------- |
| 1    | `1-prepare-data.ipynb` | Format the When2Call dataset into DPO preference pairs and upload to S3                  |
| 2    | `2-fine-tune.ipynb`    | Fine-tune Nova Micro with DPO and LoRA using the Nova Customization SDK                  |
| 3    | `3-deploy.ipynb`       | Import to Bedrock and deploy for on-demand inference                                     |
| 4    | `4-evaluate.ipynb`     | Evaluate tool-calling accuracy with LLM-as-a-Judge (Amazon Nova Pro)                     |

**Model:** Amazon Nova Micro 1.0 | **Dataset:** [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) | **Use case:** Function calling / tool use

---

### Nova 2.0

Customize **Amazon Nova Lite 2.0** with SFT using the Nova 2.0 Converse format with `reasoningContent`.

> **Note:** The Nova 2.0 labs require `ml.p5.48xlarge` instances and can only be run in your own AWS account. They are not available in AWS-hosted workshop events.

#### Lab 1 — Supervised Fine-Tuning (SFT) with LoRA

Train the model on multilingual reasoning data using the Nova 2.0 `reasoningContent` format for structured chain-of-thought reasoning.

| Step | Notebook               | Description                                                                              |
| ---- | ---------------------- | ---------------------------------------------------------------------------------------- |
| 1    | `1-prepare-data.ipynb` | Format the Multilingual-Thinking dataset with `reasoningContent` and upload to S3        |
| 2    | `2-fine-tune.ipynb`    | Fine-tune Nova Lite 2.0 with SFT and LoRA using the Nova Customization SDK               |
| 3    | `3-deploy.ipynb`       | Import to Bedrock, deploy, and test with extended thinking enabled                       |
| 4    | `4-evaluate.ipynb`     | Evaluate with LLM-as-a-Judge (Amazon Nova Pro) comparing base vs fine-tuned              |

**Model:** Amazon Nova Lite 2.0 | **Dataset:** [HuggingFaceH4/Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking) | **Use case:** Multilingual chain-of-thought reasoning

## Prerequisites

### Nova 1.0 Labs

- An AWS account with access to Amazon SageMaker AI and Amazon Bedrock
- A SageMaker Studio domain with JupyterLab
- IAM permissions for SageMaker Training, Bedrock, and S3
- Service quota for `ml.g5.12xlarge` instances for training

### Nova 2.0 Labs

- All of the above, plus:
- Service quota for **4x `ml.p5.48xlarge`** instances for training
- Access to Amazon Nova 2 Lite model in Amazon Bedrock

## How to Run

This workshop follows a hands-on format. Each Nova version is independent — you can start with either based on your needs.

- **Nova 1.0** covers two techniques (SFT and DPO) on Nova Micro using `ml.g5.12xlarge` instances
- **Nova 2.0** covers SFT on Nova Lite 2.0 with the new `reasoningContent` format using `ml.p5.48xlarge` instances

Each lab contains:

- Step-by-step Jupyter notebooks with instructions and explanations
- Code samples that you can run and modify
- Evaluation notebooks to validate fine-tuning results

## Repository Structure

```
.
├── nova_1_0/
│   ├── lab-1-supervised-fine-tuning-nova/
│   │   ├── requirements.txt
│   │   ├── 1-prepare-data.ipynb
│   │   ├── 2-fine-tune.ipynb
│   │   ├── 3-deploy.ipynb
│   │   └── 4-evaluate.ipynb
│   └── lab-2-dpo/
│       ├── requirements.txt
│       ├── 1-prepare-data.ipynb
│       ├── 2-fine-tune.ipynb
│       ├── 3-deploy.ipynb
│       └── 4-evaluate.ipynb
└── nova_2_0/
    └── lab-1-supervised-fine-tuning/
        ├── requirements.txt
        ├── 1-prepare-data.ipynb
        ├── 2-fine-tune.ipynb
        ├── 3-deploy.ipynb
        └── 4-evaluate.ipynb
```

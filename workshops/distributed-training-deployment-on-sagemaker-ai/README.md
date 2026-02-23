# Advanced Model Customization on SageMaker AI

Welcome to the **Advanced Model Customization on SageMaker AI** workshop! This hands-on, self-paced workshop is designed for ML practitioners who want to customize Large Language Models (LLMs) using different techniques and deploy them on AWS.

You will explore the full model customization lifecycle — from dataset preparation and distributed training, to real-time deployment and evaluation — across three progressively advanced solutions using Amazon SageMaker AI.

## Learning Objectives

By the end of this workshop, you'll be able to:

- Prepare datasets for different model customization techniques (CPT, SFT, DPO)
- Run distributed fine-tuning workloads using SageMaker JumpStart, SageMaker Training Jobs, and SageMaker HyperPod with EKS
- Deploy fine-tuned models using vLLM on SageMaker Real-time Endpoints and HyperPod Inference Operator
- Evaluate model quality using statistical metrics (ROUGE, BERTScore) and LLM-as-a-Judge with Amazon Nova

## Workshop Structure

### Solution 1: SageMaker JumpStart

A managed, low-code approach to fine-tuning using pre-built model recipes.

| Notebook                                  | Description                                                                             |
| ----------------------------------------- | --------------------------------------------------------------------------------------- |
| `jumpstart-llama3.1-8b-instruct-ft.ipynb` | Fine-tune Meta Llama 3.1 8B Instruct with LoRA and FSDP, deploy to a real-time endpoint |

**Model:** Meta Llama 3.1 8B Instruct | **Use case:** Telecom customer promotions

---

### Solution 2: SageMaker Training Jobs

Full control over training scripts, hyperparameters, and infrastructure using custom SageMaker Training Jobs with QLoRA and FSDP.

**Model:** Qwen3-4B-Instruct

#### Option 1 — Continued Pre-Training (CPT)

Inject new domain knowledge into a foundation model using unsupervised training on raw text.

| Step | Notebook                        | Description                                                             |
| ---- | ------------------------------- | ----------------------------------------------------------------------- |
| 1    | `1-prepare-data.ipynb`          | Tokenize the FineMath dataset into fixed-length blocks and upload to S3 |
| 2    | `2-continued-pretraining.ipynb` | Run distributed CPT with QLoRA and FSDP on SageMaker                    |
| 3    | `3-deployment.ipynb`            | Deploy with vLLM and test streaming inference on math reasoning tasks   |

**Use case:** Mathematical reasoning

#### Option 2 — Supervised Fine-Tuning (SFT)

Train the model on instruction-response pairs to specialize it for a specific task.

| Step | Notebook                         | Description                                                                               |
| ---- | -------------------------------- | ----------------------------------------------------------------------------------------- |
| 1    | `1-prepare-data.ipynb`           | Format the Medical O1 Reasoning dataset into chat messages and upload to S3               |
| 2    | `2-supervised-fine-tuning.ipynb` | Run distributed SFT with QLoRA and FSDP on SageMaker                                      |
| 3    | `3-deployment.ipynb`             | Deploy with vLLM and test streaming inference on medical questions                        |
| 4    | `4-evaluation.ipynb`             | Evaluate with ROUGE, BERTScore, semantic similarity, and LLM-as-a-Judge (Amazon Nova Pro) |

**Use case:** Medical expert reasoning

#### Option 3 — Direct Preference Optimization (DPO)

Align model outputs to human preferences using chosen/rejected response pairs.

| Step | Notebook                                 | Description                                                                               |
| ---- | ---------------------------------------- | ----------------------------------------------------------------------------------------- |
| 1    | `1-prepare-data.ipynb`                   | Format the NVIDIA When2Call dataset into preference pairs and upload to S3                |
| 2    | `2-direct-preference-optimization.ipynb` | Run DPO training with TRL's DPOTrainer on SageMaker                                       |
| 3    | `3-deployment.ipynb`                     | Deploy with vLLM (tool-calling enabled) and test function-calling inference               |
| 4    | `4-evaluation.ipynb`                     | Evaluate tool-calling accuracy and response quality with LLM-as-a-Judge (Amazon Nova Pro) |

**Use case:** Function calling / tool use

---

### Solution 3: SageMaker HyperPod with EKS

Run the same customization techniques on a Kubernetes-native infrastructure using SageMaker HyperPod with EKS orchestration. Training jobs run as Kubernetes pods with Karpenter autoscaling, FSx for Lustre shared storage, and deployment via the HyperPod Inference Operator.

**Model:** Qwen3-4B-Instruct

Each option includes a README with step-by-step instructions for deploying training pods and inference endpoints using `kubectl`.

#### Option 1 — Continued Pre-Training (CPT)

| Step | File                                | Description                                          |
| ---- | ----------------------------------- | ---------------------------------------------------- |
| 1    | `1-prepare-data.ipynb`              | Prepare dataset (same as Solution 2)                 |
| 2    | `pod-finetuning.yaml` + `args.yaml` | Deploy distributed CPT training pod on HyperPod      |
| 3    | `deployment.yaml`                   | Deploy trained model via HyperPod Inference Operator |

#### Option 2 — Supervised Fine-Tuning (SFT)

| Step | File                                | Description                                          |
| ---- | ----------------------------------- | ---------------------------------------------------- |
| 1    | `1-prepare-data.ipynb`              | Prepare dataset (same as Solution 2)                 |
| 2    | `pod-finetuning.yaml` + `args.yaml` | Deploy distributed SFT training pod on HyperPod      |
| 3    | `deployment.yaml`                   | Deploy trained model via HyperPod Inference Operator |
| 4    | `3-evaluation.ipynb`                | Evaluate the deployed model                          |

#### Option 3 — Direct Preference Optimization (DPO)

| Step | File                                | Description                                          |
| ---- | ----------------------------------- | ---------------------------------------------------- |
| 1    | `1-prepare-data.ipynb`              | Prepare dataset (same as Solution 2)                 |
| 2    | `pod-finetuning.yaml` + `args.yaml` | Deploy distributed DPO training pod on HyperPod      |
| 3    | `deployment.yaml`                   | Deploy trained model via HyperPod Inference Operator |
| 4    | `3-evaluation.ipynb`                | Evaluate the deployed model                          |

## Prerequisites

### Solutions 1 and 2

- An AWS account with access to Amazon SageMaker AI
- A SageMaker Studio domain with JupyterLab or Code Editor
- IAM permissions for SageMaker Training, Endpoints, and S3
- Service quota for `ml.g5.12xlarge` (or equivalent GPU instances) for training and inference

### Solution 3

- All of the above, plus:
- An Amazon SageMaker HyperPod cluster with EKS orchestration
- `kubectl` and `aws` CLI configured with cluster access
- FSx for Lustre filesystem shared between SageMaker Studio and HyperPod

## How to Run

This workshop follows a hands-on, self-paced format. Each solution is independent — you can start with any solution that matches your use case.

- **Solution 1** is the quickest path: a single notebook covers fine-tuning and deployment
- **Solution 2** provides the most depth: data prep, training, deployment, and evaluation across three customization techniques
- **Solution 3** demonstrates the same techniques on Kubernetes-native infrastructure for teams operating at scale

Each module contains:

- Step-by-step instructions and explanations
- Code samples that you can run and modify
- Links to additional resources

## Repository Structure

```
.
├── solution-1-sagemaker-jumpstart/
│   └── jumpstart-llama3.1-8b-instruct-ft.ipynb
├── solution-2-sagemaker-training/
│   ├── option-1-continued-pre-training/
│   │   ├── 1-prepare-data.ipynb
│   │   ├── 2-continued-pretraining.ipynb
│   │   └── 3-deployment.ipynb
│   ├── option-2-supervised-fine-tuning/
│   │   ├── 1-prepare-data.ipynb
│   │   ├── 2-supervised-fine-tuning.ipynb
│   │   ├── 3-deployment.ipynb
│   │   └── 4-evaluation.ipynb
│   └── option-3-dpo/
│       ├── 1-prepare-data.ipynb
│       ├── 2-direct-preference-optimization.ipynb
│       ├── 3-deployment.ipynb
│       └── 4-evaluation.ipynb
└── solution-3-sagemaker-hyperpod-k8/
    ├── option-1-continued-pre-training/
    │   ├── 1-prepare-data.ipynb
    │   ├── pod-finetuning.yaml, args.yaml
    │   └── deployment.yaml
    ├── option-2-supervised-fine-tuning/
    │   ├── 1-prepare-data.ipynb
    │   ├── pod-finetuning.yaml, args.yaml
    │   ├── deployment.yaml
    │   └── 3-evaluation.ipynb
    └── option-3-dpo/
        ├── 1-prepare-data.ipynb
        ├── pod-finetuning.yaml, args.yaml
        ├── deployment.yaml
        └── 3-evaluation.ipynb
```

# Mastering foundation model customization with Amazon SageMaker AI

This workshop demonstrates end-to-end fine-tuning, evaluation, and deployment of vision-language models using Amazon SageMaker AI.

## Overview

- Model: [Qwen/Qwen3-VL-2B-Instruct](Qwen/Qwen3-VL-2B-Instruct) (or [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct))
- Dataset: [pranavvmurthy26/DoclingMatix_500](https://huggingface.co/datasets/pranavvmurthy26/DoclingMatix_500)
- License: `Apache-2.0`

## Workflow

### Lab 1: Data Preparation & Fine-Tuning Setup
**Notebook:** `1-2-finetune--Qwen--Qwen3-VL-2B-Instruct.ipynb`

1. Load and prepare the DoclingMatix dataset for multimodal instruction tuning
2. Convert dataset to messages format with base64-encoded images
3. Upload training data to S3
4. Configure and launch SageMaker training job using:
   - PyTorch Estimator with ModelTrainer
   - PeFT/LoRA strategy (4-bit quantization)
   - DeepSpeed ZeRO-3 optimization
   - Optional MLflow tracking for metrics
5. Monitor training progress and retrieve model artifacts

**Key Outputs:**
- Training dataset in JSONL format
- Fine-tuned model artifacts in S3
- Training job name for downstream tasks

### Lab 2: Model Evaluation
**Notebook:** `3-evaluation-stream--Qwen--Qwen3-VL-2B-Instruct.ipynb`

1. Generate predictions from both base and fine-tuned models
2. Run statistical evaluation metrics:
   - ROUGE scores (rouge1, rouge2, rougeL)
   - BERTScore (precision, recall, F1)
   - Semantic similarity using sentence transformers
3. Run qualitative evaluation using LLM-as-a-Judge:
   - Amazon Nova Pro evaluates outputs on factual accuracy, completeness, relevance, and clarity
   - Scores generated on 0-100 scale
4. Compare results between base and fine-tuned models
5. Visualize improvements with histograms and box plots

**Key Outputs:**
- Statistical metrics logged to MLflow
- LLM judge evaluation results
- Comparative analysis showing fine-tuning improvements

### Lab 3: Model Deployment
**Notebook:** `4-deploy--Qwen--Qwen3-VL-2B-Instruct.ipynb`

1. Clean up pre-existing endpoints (if any)
2. Create SageMaker endpoint configuration
3. Deploy endpoint with vLLM inference container
4. Create inference component with:
   - Custom vLLM image for Qwen3-VL
   - GPU memory optimization (85% utilization)
   - Streaming response support
5. Test deployed endpoint with sample image + text prompts

**Key Outputs:**
- Production-ready SageMaker endpoint
- Streaming inference capability
- Real-time multimodal inference API

## Quick Start with SageMaker

The recommended approach is to use the Jupyter notebooks in sequence (Lab 1 → Lab 2 → Lab 3).

## Key Features

- Multimodal fine-tuning with vision + text inputs
- Efficient training with PeFT/LoRA and 4-bit quantization
- Comprehensive evaluation (statistical + LLM-as-a-Judge)
- Production deployment with vLLM for optimized inference
- MLflow integration for experiment tracking
- Streaming inference support for real-time applications





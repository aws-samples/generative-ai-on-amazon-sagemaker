# Customizing Amazon Nova Models for Agentic Workflows with Strands

This workshop demonstrates end-to-end fine-tuning, deployment, evaluation, and agentic integration of Amazon Nova models using Amazon SageMaker AI and Strands Agents SDK.

## Overview

- Model: [Amazon Nova Micro](https://aws.amazon.com/ai/generative-ai/nova/)
- Dataset: [HuggingFaceH4/Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)
- License: `Apache-2.0`

## Workflow

### Lab 1: Prerequisites

**Notebook:** `lab-1-2-3-prepare-data-fine-tune.ipynb`

1. Set up SageMaker session and IAM role
2. Configure S3 bucket for training artifacts
3. Initialize AWS clients (SageMaker, S3, Bedrock)

**Key Outputs:**

- SageMaker session configuration
- IAM role ARN
- S3 bucket paths

### Lab 2: Data Preparation

**Notebook:** `lab-1-2-3-prepare-data-fine-tune.ipynb`

1. Load the Multilingual-Thinking dataset from HuggingFace
2. Split dataset into train, validation, and test sets
3. Format data for Amazon Nova's conversation format:
   - System prompts with language-specific reasoning instructions
   - User messages with queries
   - Assistant responses with `<think>...</think>` tags for reasoning
4. Apply token-based truncation to ensure compliance with model limits (3072 tokens)
5. Upload formatted datasets to S3

**Key Outputs:**

- Training dataset in JSONL format (S3)
- Validation dataset in JSONL format (S3)
- Test dataset in JSONL format (S3)

### Lab 3: Model Fine-Tuning

**Notebook:** `lab-1-2-3-prepare-data-fine-tune.ipynb`

1. Configure PyTorch Estimator with SageMaker ModelTrainer
2. Set up training parameters:
   - Instance type: `ml.g5.12xlarge`
   - LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
   - Learning rate: 2e-5 with warmup steps
   - Custom training recipe for Nova Micro
3. Launch SageMaker training job with train and validation data
4. Monitor training progress via SageMaker console and CloudWatch logs

**Key Outputs:**

- Fine-tuned model artifacts in S3
- Training job name for downstream tasks
- Model checkpoints

### Lab 4: Model Deployment

**Notebook:** `lab-4-5-deploy-evaluate.ipynb`

1. Retrieve the latest completed training job
2. Extract checkpoint S3 URI from training job metadata
3. Create Bedrock Custom Model:
   - Import fine-tuned model from S3
   - Register model with Amazon Bedrock
4. Deploy custom model for on-demand inference (ODI)
5. Monitor deployment status until ACTIVE
6. Test deployed model with sample queries

**Key Outputs:**

- Bedrock Custom Model ARN
- Custom Model Deployment ARN
- Production-ready inference endpoint

### Lab 5: Model Evaluation

**Notebook:** `lab-4-5-deploy-evaluate.ipynb`

1. Generate predictions from both base Nova Micro and fine-tuned models
2. Create evaluation dataset with:
   - System prompts
   - User queries
   - Base model responses
   - Fine-tuned model responses
3. Run LLM-as-a-Judge evaluation using Amazon Nova Pro:
   - Format Compliance Score (60 points): Language separation and think tags
   - Instruction Following Score (30 points): System prompt adherence
   - Content Quality Score (10 points): Reasoning coherence and completeness
4. Extract preferences and scores from judge evaluations
5. Visualize results with:
   - Preference distribution pie chart
   - Score comparison bar chart

**Key Outputs:**

- LLM judge evaluation results (JSON)
- Preference distribution visualization
- Score comparison charts
- Performance metrics comparing base vs fine-tuned models

### Lab 6: Agentic Integration with Strands

**Notebook:** `lab-6-nova-strands-agent.ipynb`

1. Install Strands Agents SDK and dependencies
2. Define custom tools for the agent:
   - `cultural_context_tool`: Provides cultural context and explanations
3. Create agent with:
   - Fine-tuned Nova Micro model from Bedrock
   - System prompt with multilingual reasoning instructions
   - Tool integration for enhanced capabilities
   - Optional guardrail configuration
4. Test agent with multilingual queries:
   - French cuisine
   - Spanish flamenco culture
   - German Oktoberfest
   - Italian literature (Dante's Divina Commedia)
5. Observe agent's reasoning in target language with English responses

**Key Outputs:**

- Functional Strands agent with fine-tuned Nova model
- Tool-augmented responses
- Multilingual reasoning demonstrations

## Quick Start with SageMaker

The recommended approach is to use the Jupyter notebooks in sequence (Lab 1 → Lab 2 → Lab 3 → Lab 4 → Lab 5 → Lab 6).

## Key Features

- Multilingual reasoning with structured thinking (`<think>` tags)
- Parameter-efficient fine-tuning with LoRA
- Token-based truncation for context window management
- Bedrock Custom Model integration for serverless inference
- Comprehensive LLM-as-a-Judge evaluation framework
- Agentic workflows with Strands SDK
- Tool integration for enhanced agent capabilities
- Guardrail support for responsible AI

## Architecture

The workshop follows this architecture:

1. **Data Preparation**: Format multilingual reasoning dataset
2. **Fine-Tuning**: Train Nova Micro with LoRA on SageMaker
3. **Deployment**: Import to Bedrock as Custom Model
4. **Evaluation**: Compare base vs fine-tuned with Nova Pro as judge
5. **Agentic Integration**: Build Strands agent with custom tools

## Requirements

See `requirements.txt` for Python dependencies.

Key packages:

- `sagemaker` - SageMaker SDK
- `boto3` - AWS SDK
- `datasets` - HuggingFace datasets
- `tiktoken` - Token counting
- `strands-agents` - Strands Agents SDK
- `matplotlib` - Visualization

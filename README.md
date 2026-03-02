# Generative AI on Amazon SageMaker and Amazon Bedrock

This repository provides comprehensive resources for working with generative AI models using Amazon SageMaker and Amazon Bedrock. Whether you're looking to fine-tune foundation models, build RAG applications, create agents, or implement responsible AI practices, you'll find practical examples and workshops here.

## Repository Structure

### Workshops

#### [Building RAG Workflows with SageMaker and Bedrock](./workshops/building-rag-workflows-with-sagemaker-and-bedrock/)

- Build experimental RAG applications
- Implement RAG with SageMaker and OpenSearch
- Fine-tune embedding models
- Customize models with RAFT
- Apply guardrails to LLM outputs

#### [Advanced Model Customization on SageMaker AI](./workshops/distributed-training-deployment-on-sagemaker-ai/)

- Explore different customization techniques, such as Continued Pre-Training (CPT), Supervised Fine-Tuning (SFT), and Reinforcement Learning Techniques such as Direct Preference Optimization (DPO)
- Implement distribution strategies such as Fully-Sharded Data Parallel (FSDP) and optimization techniques such as LoRA
- Execute workloads on SageMaker Training jobs and SageMaker HyperPod
- Deploy models on SageMaker Managed Inference and on SageMaker HyperPod with Inference Operator
- Evaluate models using Statistical evaluations and LLM as Judge with Amazon Bedrock

#### [DIY Agents with SageMaker and Bedrock](./workshops/diy-agents-with-sagemaker-and-bedrock/)

- Basic inference with Bedrock and SageMaker
- Implement tool calling capabilities
- Build agent patterns (autonomous, orchestrator-worker, etc.)
- Use agent frameworks (LangGraph, CrewAI, Strands, etc.)
- Add observability with Langfuse and MLflow

#### [Fine-tuning with SageMaker AI and Bedrock](./workshops/fine-tuning-with-sagemakerai-and-bedrock/)

- Set up a Foundation Model Playground
- Customize foundation models
- Evaluate models with LightEval
- Implement responsible AI with Bedrock Guardrails
- Develop FMOps fine-tuning workflows with SageMaker Pipelines

#### [Partner AI Apps with SageMaker AI](./workshops/partner-ai-apps-with-sagemakerai/)

- Experiment Management with Comet [Image Classification, Fraud Detection]
- Evaluting LLM applications with Comet Opik
- Evaluating Agents with Opik
- LLM Evaluation with Comet
- Model Monitoring with Fiddler
- RAG chatbot evaluation with Deepchecks

#### [Amazon Nova Customization on SageMaker AI](./workshops/nova-customization-on-sagemaker-ai/)

- Customize Amazon Nova models (Micro 1.0, Lite 2.0) using SageMaker Training jobs with the Amazon Nova Customization SDK
- Fine-tune with Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) using LoRA
- Deploy fine-tuned models on Amazon Bedrock with custom model on-demand inference
- Evaluate models using LLM-as-a-Judge with Amazon Nova Pro

#### [Serverless Model Customization with SageMaker AI](./workshops/serverless-model-customization-with-sagemaker-ai/)

- Fine-tune LLMs using serverless customization APIs (no infrastructure management)
- Explore four techniques: Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), Reinforcement Learning from Verifiable Rewards (RLVR), and Reinforcement Learning from AI Feedback (RLAIF)
- Register datasets, models, and evaluators in the SageMaker AI Registry
- Evaluate models using LLM-as-a-Judge with custom metrics and benchmark evaluations
- Deploy fine-tuned models with DJL LMI and vLLM on SageMaker Real-time Endpoints

## Getting Started

1. Clone this repository
2. Navigate to the workshop of your choice
3. Follow the instructions in the workshop's README.md file
4. Each workshop contains Jupyter notebooks that guide you through the process

## Prerequisites

- AWS account with appropriate permissions
- Basic understanding of machine learning concepts
- Familiarity with Python programming
- Access to Amazon SageMaker and Amazon Bedrock services

## Contributing

We welcome contributions! Please see [CONTRIBUTING](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or suggest improvements.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information about reporting security issues.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file for details.

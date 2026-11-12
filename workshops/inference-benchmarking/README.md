# SageMaker AI Automated Benchmarking & Inference Recommendations Workshop

This workshop walks you through the complete lifecycle of benchmarking and optimizing generative AI model deployments on Amazon SageMaker AI. You'll deploy a model, benchmark it under various conditions, use Automated Inference Recommendations to find the optimal configuration, and then compare results.

## 🎯 What You'll Learn

- How to deploy a foundation model on SageMaker AI using JumpStart
- How to use **SageMaker Automated Benchmarking** (`CreateAIBenchmarkJob`) to measure endpoint performance with NVIDIA AIPerf
- How workload parameters (concurrency, token lengths, request rates) impact inference metrics
- How to use **SageMaker Automated Inference Recommendations** (`CreateAIRecommendationJob`) to find optimal deployment configurations
- How to deploy an optimized recommendation and compare against your baseline
- Cost-per-token analysis across different configurations

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SageMaker AI Benchmarking Service                  │
│                                                                       │
│  ┌───────────────┐    ┌───────────────────┐    ┌─────────────────┐  │
│  │  AI Workload  │───▶│  AI Benchmark Job │───▶│  S3 Results     │  │
│  │  Config       │    │  (NVIDIA AIPerf)  │    │  (JSON metrics) │  │
│  └───────────────┘    └────────┬──────────┘    └─────────────────┘  │
│                                │                                      │
│                                ▼                                      │
│                     ┌──────────────────┐                             │
│                     │  SageMaker AI    │                             │
│                     │  Endpoint        │                             │
│                     └──────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│              SageMaker AI Inference Recommendations                   │
│                                                                       │
│  ┌───────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────┐  │
│  │  Model    │──▶│  Analyze &   │──▶│  Benchmark │──▶│  Ranked  │  │
│  │  in S3    │   │  Optimize    │   │  on GPU    │   │  Results  │  │
│  └───────────┘   └──────────────┘   └────────────┘   └──────────┘  │
│                   • Spec. Decoding    • Real infra     • Deploy-    │
│                   • Kernel Tuning     • AIPerf         │  ready     │
│                   • Tensor Parallel                    │  configs   │
└─────────────────────────────────────────────────────────────────────┘
```

## 📋 Labs

| Lab | Title | Duration | Cost Estimate |
|-----|-------|----------|---------------|
| [Lab 1](./lab1/) | Deploy and Benchmark Your First Model | ~45 min | ~$15-20 |
| [Lab 2](./lab2/) | Explore Benchmarking Nuances | ~60 min | ~$25-35 |
| [Lab 3](./lab3/) | Automated Inference Recommendations | ~90 min | ~$40-60 |
| [Lab 4](./lab4/) | Deploy the Recommendation and Compare | ~45 min | ~$20-30 |

### Lab 1: Deploy and Benchmark Your First Model
Deploy Meta Llama 3.1 8B Instruct from JumpStart, create a workload configuration, and run your first automated benchmark job. Learn to interpret TTFT, ITL, throughput, and latency percentiles.

### Lab 2: Explore Benchmarking Nuances
Systematically vary workload parameters — concurrency, token lengths, request rates, and streaming modes — to understand what drives each performance metric. Build comparison tables and identify optimal operating points.

### Lab 3: Automated Inference Recommendations
Use SageMaker AI Inference Recommendations to automatically find the best deployment configuration. Compare across instance types with automatic optimization (speculative decoding, kernel tuning). Understand how performance targets change the optimization strategy.

### Lab 4: Deploy the Recommendation and Compare
Deploy the recommended optimized configuration and benchmark it with the same workload as Lab 1. Compare baseline vs. optimized performance side-by-side and calculate cost-per-token improvements.

## 🔧 Prerequisites

- AWS account with SageMaker AI access
- IAM role with permissions for SageMaker, S3, and Secrets Manager
- Service quota for `ml.g6.12xlarge` (or `ml.g5.12xlarge`) endpoint instances
- HuggingFace account with access token (for gated model tokenizers)
- Python 3.10+

## 🌎 Supported Regions

SageMaker AI Automated Benchmarking and Inference Recommendations are available in:
- US East (N. Virginia) — `us-east-1`
- US East (Ohio) — `us-east-2`
- US West (Oregon) — `us-west-2`
- Asia Pacific (Singapore) — `ap-southeast-1`
- Asia Pacific (Tokyo) — `ap-northeast-1`
- Europe (Frankfurt) — `eu-central-1`
- Europe (Ireland) — `eu-west-1`

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/aws-samples/sagemaker-genai-hosting-examples.git
cd sagemaker-genai-hosting-examples/04-workshops/inference-benchmarking

# Install dependencies
pip install -r requirements.txt

# Open Lab 1 in JupyterLab or SageMaker Studio
jupyter lab lab1/lab1_deploy_and_benchmark.ipynb
```

## 📖 Key Concepts

| Term | Definition |
|------|-----------|
| **AI Workload Config** | Defines traffic patterns and benchmark parameters (token distributions, concurrency, request rate) |
| **AI Benchmark Job** | Runs NVIDIA AIPerf against an existing endpoint and produces performance metrics |
| **AI Recommendation Job** | Analyzes a model, applies optimizations, benchmarks across instance types, returns ranked deployment configs |
| **TTFT** | Time to First Token — latency before the first output token is generated |
| **ITL** | Inter-Token Latency — time between consecutive output tokens |
| **Speculative Decoding** | Optimization that uses a lightweight model to propose tokens, verified by the main model in parallel |
| **Kernel Tuning** | Optimizes GPU kernel parameters for the specific model + hardware + workload combination |

## 🧹 Cost Management

Each lab includes a cleanup section. To avoid unexpected charges:
1. Always run the cleanup cells at the end of each lab
2. Verify no endpoints remain: `aws sagemaker list-endpoints --region us-west-2`
3. Benchmark jobs are billed only for the duration of the benchmark run (typically 2-5 minutes)
4. Recommendation jobs provision temporary endpoints for benchmarking — these are automatically cleaned up

## 📚 References

- [Benchmark generative AI inference endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/generative-ai-inference-recommendations-benchmark.html)
- [Optimized generative AI inference recommendations](https://docs.aws.amazon.com/sagemaker/latest/dg/generative-ai-inference-recommendations.html)
- [Set up a workload configuration](https://docs.aws.amazon.com/sagemaker/latest/dg/generative-ai-inference-recommendations-workload-config.html)
- [Deploy a generative AI inference recommendation](https://docs.aws.amazon.com/sagemaker/latest/dg/generative-ai-inference-recommendations-deploy.html)
- [Blog: Amazon SageMaker AI now supports optimized generative AI inference recommendations](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-ai-now-supports-optimized-generative-ai-inference-recommendations/)

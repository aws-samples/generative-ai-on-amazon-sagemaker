## Prerequisites

- Amazon SageMaker Studio domain with a user profile
- Access to Amazon SageMaker Hyperpod with EKS (Elastic Kubernetes Service)

Please follow [SageMaker Studio + Hyperpod Integration](https://catalog.workshops.aws/sagemaker-hyperpod-eks/en-US/11-tips/08-studio-integration)

## Required Changes Before Running

Before running the scripts, you need to make the following changes:

### 1. Copy `solution-3-sagemaker-hyperpod-k8` folder in the FSx for Lustre volume

In order to leverage the integration through shared FSx for Lustre volume between Amazon SageMaker Studio and SageMaker Hyperpod, copy the folder and the content of `solution-3-sagemaker-hyperpod-k8` in the FSx for Lustre volume mounted on both SageMaker Studio and the Hyperpod cluster.

### 2. Update `args.yaml`

Replace all instances of `<STUDIO_USER_PROFILE>` with your SageMaker Studio user profile name:

```yaml
model_id: "/data/<STUDIO_USER_PROFILE>/lab-hp-k8-sft/DeepSeek-R1-Distill-Qwen-7B"
output_dir: "/data/<STUDIO_USER_PROFILE>/lab-hp-k8-sft/model/"
train_dataset_path: "/data/<STUDIO_USER_PROFILE>/lab-hp-k8-sft/data/train/"
test_dataset_path: "/data/<STUDIO_USER_PROFILE>/lab-hp-k8-sft/data/test/"
```

### 2. Update `pod-finetuning.yaml`

Replace all instances of `<STUDIO_USER_PROFILE>` with your SageMaker Studio user profile name:

```yaml
command:
  - /bin/bash
  - -c
  - |
    pip install -r /data/<STUDIO_USER_PROFILE>/lab-hp-k8-sft/requirements.txt && \
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    /data/<STUDIO_USER_PROFILE>/lab-hp-k8-sft/scripts/train.py \
    --config /data/<STUDIO_USER_PROFILE>/lab-hp-k8-sft/args.yaml
```

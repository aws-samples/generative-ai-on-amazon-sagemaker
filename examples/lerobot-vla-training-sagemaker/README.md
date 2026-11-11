# Training Vision-Language-Action models with Hugging Face LeRobot on Amazon SageMaker AI

Companion code for the blog post *Hugging Face LeRobot Meets Amazon SageMaker: Building the Complete VLA
Training Pipeline*. One container image and one launcher script fine-tune NVIDIA GR00T N1.7, π₀.₅ and ACT on
[LeRobot](https://github.com/huggingface/lerobot) 0.6.1, on either **SageMaker Training Jobs** or
**SageMaker HyperPod (EKS)**, from a single GPU to multi-node with EFA.

```
training-jobs/   image (Dockerfile on the AWS Deep Learning Container), SDK v3 launcher, train.sh, presets
hyperpod/        PyTorchJob template + launcher that runs the same image and train.sh on HyperPod EKS
```

Dataset used throughout: `sreetz-nv/so101_teleop_vials_rack_left` from NVIDIA's
[Sim-to-Real SO-101 learning path](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/)
(75 sim teleoperation episodes of a vial pick-and-place). Any LeRobot v3 dataset works via `--dataset-repo-id`.

## Quick start (Training Jobs)

```bash
cd training-jobs
pip install "sagemaker>=3.22" boto3
export AWS_REGION=us-east-2
./build_and_push.sh                                          # builds + pushes to ECR, prints TRAINING_IMAGE_URI
export TRAINING_IMAGE_URI=... SAGEMAKER_ROLE_ARN=... HF_TOKEN=hf_xxx   # token needed for GR00T and π₀.₅

python launch_training_job.py --model groot --instance-type ml.g7e.2xlarge
python launch_training_job.py --model pi05  --instance-type ml.g7e.4xlarge
python launch_training_job.py --model act   --instance-type ml.g6.12xlarge --instance-count 2 \
       --subnets subnet-a,subnet-b --security-group-ids sg-xxxx                  # multi-node, EFA
```

## Quick start (HyperPod EKS)

```bash
cd hyperpod
export IMAGE=<your ECR image> HF_TOKEN=hf_xxx
./launch.sh --model groot --instance-type ml.g7e.2xlarge
./launch.sh --model act   --instance-type ml.g6.12xlarge --gpus 4 --efa 1 --nodes 2
```

## What was validated

Measured on 300-step runs, us-east-2, September 2026:

| Model | 96 GB (ml.g7e) | 48 GB (ml.g6e) | 24 GB (ml.g6 / ml.g5) |
|---|---|---|---|
| ACT | 15 GB | 15 GB | 15 GB |
| GR00T N1.7 fine-tune | 41.5 GB @ bs32 | 39 GB @ bs16 | projector only (21.5 GB @ bs4) |
| π₀.₅ full fine-tune | 41 GB @ bs16 | 40.5 GB @ bs8 | LoRA (10.7 GB @ bs4) |

Multi-GPU DDP and 2-node EFA (NCCL over Libfabric) verified on both backends; checkpoint resume across GPU types
verified on Training Jobs. Details and gotchas are in `training-jobs/README.md`.

## Requirements

- Hugging Face token with the gates accepted for `nvidia/Cosmos-Reason2-2B` (GR00T backbone) and
  `google/paligemma-3b-pt-224` (π₀.₅ backbone). Passed with `--hf-token` / `HF_TOKEN`; it ends up as a plain
  environment variable on the job or pod (visible in `DescribeTrainingJob` / the pod spec).
- SageMaker execution role that `sagemaker.amazonaws.com` can assume, with ECR pull and S3 access.
- For multi-node EFA on Training Jobs: a VPC subnet set and a security group that allows all traffic to itself.

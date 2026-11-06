# LeRobot VLA fine-tuning on SageMaker Training Jobs

One image, one launcher script, three policies (NVIDIA GR00T N1.7, π₀.₅, ACT). The same image and `src/train.sh`
also run on HyperPod EKS; that part lives in `../hyperpod/`.

```
training-jobs/
├── docker/Dockerfile.dlc      # PRIMARY: AWS DLC pytorch-training 2.10.0 (py3.13, cu130; EFA + aws-ofi-nccl included)
│                              # + lerobot[groot,pi,peft,training]==0.6.1 (torch pinned), minus torchcodec/flash-attn
├── docker/Dockerfile          # ALTERNATIVE: nvidia/cuda:13.0.3 + uv + torch 2.11 cu130 + EFA 1.50.0 + aws-ofi-nccl
│                              # (for a torch newer than the DLC ships; needs the libnccl-net.so symlink)
├── build_and_push.sh          # docker build (linux/amd64, context = this dir) + push to ECR
├── launch_training_job.py     # SageMaker Python SDK v3 ModelTrainer (Training Jobs path)
├── results.py                 # summarise jobs: status, wall, peak VRAM, errors from CloudWatch
└── src/                       # synced into the job at start by SDK v3 SourceCode (no rebuild on edit)
    ├── train.sh               # topology (resourceconfig.json | Kubeflow env) -> [accelerate launch] lerobot-train
    └── presets/{act,groot,pi05}.env   # per-model --policy.* flags, default batch, LoRA mechanism
```

## Training Jobs

```bash
pip install "sagemaker>=3.22" boto3
export AWS_REGION=us-east-2
DOCKERFILE=docker/Dockerfile.dlc ./build_and_push.sh   # prints TRAINING_IMAGE_URI (~10 min on a c6i.4xlarge)
# or DOCKERFILE=docker/Dockerfile for the NVIDIA-base variant
export TRAINING_IMAGE_URI=<acct>.dkr.ecr.us-east-2.amazonaws.com/lerobot-vla:0.6.1
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<acct>:role/<role sagemaker.amazonaws.com can assume>
export HF_TOKEN=hf_xxx   # or --hf-token on each command; needed for groot and pi05 (gated backbones)

# single GPU (Blackwell 96 GB)
python launch_training_job.py --model act   --instance-type ml.g7e.2xlarge
python launch_training_job.py --model groot --instance-type ml.g7e.2xlarge
python launch_training_job.py --model pi05  --instance-type ml.g7e.4xlarge
# 48 GB (L40S): half batch
python launch_training_job.py --model groot --instance-type ml.g6e.2xlarge --batch-size 16
python launch_training_job.py --model pi05  --instance-type ml.g6e.4xlarge --batch-size 8 
# 24 GB (L4 / A10G): LoRA for pi05, projector-only for groot
python launch_training_job.py --model pi05  --instance-type ml.g6.4xlarge --peft --batch-size 4
python launch_training_job.py --model groot --instance-type ml.g6.2xlarge --batch-size 4 \
       --extra-args "--policy.tune_diffusion_model=false"
# multi-GPU (DDP) and multi-node with EFA (needs VPC: SG allowing all traffic to itself)
python launch_training_job.py --model act --instance-type ml.g5.12xlarge
python launch_training_job.py --model act --instance-type ml.g6.12xlarge --instance-count 2 \
       --subnets subnet-a,subnet-b,subnet-c --security-group-ids sg-xxxx
# instance list: starts on the first type with capacity (not combinable with --spot)
python launch_training_job.py --model groot --instance-type ml.g7e.2xlarge,ml.g7e.4xlarge,ml.g6e.2xlarge --batch-size 16
# managed Spot; an interrupted job resumes from the last checkpoint in S3
python launch_training_job.py --model act --instance-type ml.g7e.4xlarge --spot
# dataset from S3 instead of the Hub; resume = same --job-name again
python launch_training_job.py --model act --dataset-s3-uri s3://bucket/prefix/so101_teleop_vials_rack_left/
python launch_training_job.py --model act --job-name my-run --steps 20000   # ... later, same name + more steps resumes
```

`--dry-run` prints the full configuration. `--steps`, `--save-freq`, `--max-runtime`, `--keep-alive` (warm pool),
`--dataset-repo-id` (any of the four `sreetz-nv/so101_teleop_vials_rack_left*` datasets) are the other knobs.

## Monitoring

Every job publishes `loss`, `grad_norm`, `lr`, `update_s` and `dataloading_s` as CloudWatch metrics (namespace
`/aws/sagemaker/TrainingJobs`, dimension `TrainingJobName`), via `MetricDefinitions` regexes applied to LeRobot's
`step:… loss:… grdn:… lr:… updt_s:… data_s:…` log line. They are charted on the training job's console page;
`DescribeTrainingJob` returns the final values in `FinalMetricDataList`. GPU/CPU/memory/disk utilisation are published by
SageMaker under the same namespace (dimension `Host`). Raw logs: log group `/aws/sagemaker/TrainingJobs`,
one stream per instance. `results.py <name-prefix>` summarises status, wall time and peak VRAM across jobs.

## How the dataset gets into the job

Two ways; both end in the same `lerobot-train` flags.

1. **From the Hugging Face Hub (default).** `--dataset-repo-id <org>/<name>` (default
   `sreetz-nv/so101_teleop_vials_rack_left`) is passed to the container as `DATASET_REPO_ID`; `train.sh` runs
   `lerobot-train --dataset.repo_id=$DATASET_REPO_ID` and LeRobot downloads the dataset into `HF_LEROBOT_HOME`
   (`/opt/ml/cache/huggingface/lerobot` on the job's EBS volume). ~400 MB for the 75-episode dataset, a few
   seconds; every job downloads afresh. Needs outbound internet (default) or a Hub proxy in a private VPC.
2. **From S3 as a SageMaker input channel.** `--dataset-s3-uri s3://bucket/prefix/` creates
   `InputData(channel_name="dataset")`; SageMaker copies the prefix to `/opt/ml/input/data/dataset` before the
   container starts; `train.sh` sees that directory and adds `--dataset.root=/opt/ml/input/data/dataset`
   (`--dataset.repo_id` is still passed, LeRobot uses it as the dataset name). No Hub access at runtime. Stage the
   dataset once with `hf download <repo> --repo-type dataset --local-dir d && aws s3 sync d s3://bucket/prefix/`.
   For datasets in the tens of GB use FastFile mode or an FSx for Lustre `FileSystemDataSource` instead.

Datasets without a `v3.0` codebase tag on the Hub (e.g. `..._sim_and_real`) need
`--extra-args='--dataset.revision=main'`.

## What `train.sh` does

1. Topology: SageMaker `resourceconfig.json` or Kubeflow `MASTER_ADDR/RANK/WORLD_SIZE`, GPUs via `nvidia-smi`.
2. HF token: plain `HF_TOKEN` env var from `--hf-token` (logged as present/absent only).
3. Dataset: `DATASET_ROOT` (default `/opt/ml/input/data/dataset`, the S3 channel) if it holds `meta/info.json`, else Hub `repo_id` (see above).
4. Resume: recreates `checkpoints/last` from the highest numbered step (S3 sync drops symlinks) and passes
   `--resume=true --config_path=…`; CLI flags after it (e.g. `--steps`) still override.
5. Launch: `lerobot-train` directly on 1 GPU; `accelerate launch --multi_gpu` (DDP) otherwise; `DISTRIBUTED=fsdp`
   is opt-in and needs `FSDP_WRAP_CLS`.
6. `PEFT=1`: generic `--peft.method_type=LORA …` (pi05). Not available for `groot` or `act`.
7. Logs `=== RESULT …` and `=== PEAK_VRAM gpu=N used_mib=…` (nvidia-smi sampler) for `results.py`.
8. Copies the last `pretrained_model/` to `/opt/ml/model` (skipping SageMaker uploader temp files).

## Adding another LeRobot policy

ACT, π₀.₅ and GR00T N1.7 are the validated presets. To train another policy that LeRobot 0.6.1 supports
(for example SmolVLA), add `src/presets/<name>.env` with its `--policy.*` flags in `POLICY_ARGS_STR`, a default
`BATCH_SIZE`, and `PEFT_SUPPORTED=0` if it has no LoRA path, then launch with `--model <name>`. If the policy needs
a LeRobot extra that the image does not install (default `LEROBOT_EXTRAS="groot,pi,peft,training"`), add it to that build argument in the Dockerfile.
Training Jobs pick up a new preset at launch; HyperPod runs `train.sh` from the image, so rebuild and push it first.

## Gotchas

- DLC base: do **not** use the torch 2.9 DLC for GR00T (bf16 Conv3d regression, ~85 s/step); use 2.10+. Remove
  torchcodec (static Python in the DLC, PyAV is used) and the DLC's CUDA-12 flash-attn.
- SDK v3 script mode needs `pip3` in the image (`uv venv --seed`).
- GR00T's backbone `nvidia/Cosmos-Reason2-2B` is gated: `--hf-token` is needed for groot too, not just pi05.
- π₀.₅ with `--policy.type=pi05 --policy.pretrained_path`: do not pass `--rename_map` (features come from the
  dataset); use `--policy.empty_cameras=1`.
- LoRA: pass `--peft.*` only; `--policy.use_peft=true` means "load an adapter".
- GR00T has no LoRA in LeRobot 0.6.1; on 24 GB use `--policy.tune_diffusion_model=false` (projector only).
- NCCL 2.28 (torch cu130) loads `libnccl-net.so`; symlink the aws-ofi plugin to that name or you silently get
  Socket instead of EFA.
- `.xlarge` sizes cap `VolumeSizeInGB` at 250.
- `CreateTrainingJob` must be able to assume the role you pass (`sagemaker.amazonaws.com` in its trust policy).
- Capacity for the larger GPU instance sizes varies by Region. Pass a comma-separated `--instance-type` list
  or use Training Plans.

## LeRobot 0.6.1 vs `main`

0.6.1 (latest release) has no `--parallelism.*`, `--accelerator.*`, `--checkpoint_format` or gradient
accumulation flags; those are on `main`. Multi-GPU is `accelerate launch`. Re-check when 0.7 ships.

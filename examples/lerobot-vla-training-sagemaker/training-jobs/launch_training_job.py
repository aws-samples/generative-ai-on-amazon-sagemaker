#!/usr/bin/env python3
"""Launch a LeRobot VLA fine-tuning job on Amazon SageMaker Training Jobs (SDK v3).

Examples
--------
# GR00T N1.7 on a single 96 GB GPU (default)
python launch_training_job.py --model groot

# pi0.5 on the same instance; gated PaliGemma needs a Hugging Face token
python launch_training_job.py --model pi05 --hf-token hf_xxx      # or export HF_TOKEN=hf_xxx

# Scale out: 8 GPUs on one node, then 2 nodes (FSDP inside node, replicate across nodes)
python launch_training_job.py --model groot --instance-type ml.g7e.48xlarge
python launch_training_job.py --model groot --instance-type ml.p5.48xlarge --instance-count 2

# Instance list: SageMaker starts the job on the first type in the list that has capacity
python launch_training_job.py --model act --instance-type ml.g7e.2xlarge,ml.g6e.2xlarge,ml.g6.2xlarge

# Short smoke test
python launch_training_job.py --model groot --steps 200 --max-runtime 1800

Requires: pip install "sagemaker>=3.22" boto3
"""
from __future__ import annotations

import argparse
import os
import pathlib
import time

import boto3
from sagemaker.core.helper.session_helper import Session
from sagemaker.train import ModelTrainer
from sagemaker.core.shapes import InstancePreference, MetricDefinition
from sagemaker.core.training.configs import (
    CheckpointConfig,
    Compute,
    InputData,
    Networking,
    OutputDataConfig,
    SourceCode,
    StoppingCondition,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# LeRobot logs one line every --log_freq steps, e.g.
#   step:8K smpl:130K ep:534 epch:7.13 loss:0.024 grdn:0.611 lr:1.8e-05 updt_s:1.898 data_s:0.004
# SageMaker applies these regexes to the CloudWatch log stream and publishes the captures as metrics
# (namespace /aws/sagemaker/TrainingJobs, dimension TrainingJobName), charted on the job's console page.
METRIC_DEFINITIONS = [
    MetricDefinition(name="loss", regex=r"loss:([0-9.eE+-]+)"),
    MetricDefinition(name="grad_norm", regex=r"grdn:([0-9.eE+-]+)"),
    MetricDefinition(name="lr", regex=r"lr:([0-9.eE+-]+)"),
    MetricDefinition(name="update_s", regex=r"updt_s:([0-9.eE+-]+)"),
    MetricDefinition(name="dataloading_s", regex=r"data_s:([0-9.eE+-]+)"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    presets = sorted(f.stem for f in pathlib.Path(HERE, "src", "presets").glob("*.env"))
    p.add_argument("--model", choices=presets, default="groot",
                   help="Policy preset from src/presets/<model>.env")
    p.add_argument("--image-uri", default=os.environ.get("TRAINING_IMAGE_URI"),
                   help="ECR URI of the image built from docker/Dockerfile (or set TRAINING_IMAGE_URI)")
    p.add_argument("--role", default=os.environ.get("SAGEMAKER_ROLE_ARN"),
                   help="SageMaker execution role ARN (or set SAGEMAKER_ROLE_ARN)")
    p.add_argument("--bucket", default=None, help="S3 bucket for outputs/checkpoints (default: SDK default bucket)")
    p.add_argument("--instance-type", default="ml.g7e.8xlarge",
                   help="One type, or a comma-separated list in priority order (instance preferences). "
                        "GPUs per instance are detected in the container")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--volume-size", type=int, default=300, help="EBS volume in GB (dataset cache + checkpoints)")
    p.add_argument("--max-runtime", type=int, default=12 * 3600, help="Hard stop in seconds")
    p.add_argument("--spot", action="store_true",
                   help="Managed Spot training; interrupted jobs resume from the last checkpoint in S3")
    p.add_argument("--max-wait", type=int, default=None,
                   help="With --spot: max seconds including waiting for Spot capacity (default: 2 x --max-runtime)")
    p.add_argument("--max-pending", type=int, default=None,
                   help="How long to wait for capacity before failing (seconds, 1800..28 days). Default: SageMaker default")
    p.add_argument("--keep-alive", type=int, default=0,
                   help="Warm pool seconds (e.g. 1800) to make repeated launches start in seconds")
    p.add_argument("--dataset-repo-id", default="sreetz-nv/so101_teleop_vials_rack_left")
    p.add_argument("--dataset-s3-uri", default=None,
                   help="Optional S3 prefix holding a LeRobot v3 dataset; mounted as channel 'dataset'. "
                        "If omitted the job streams the dataset from the Hugging Face Hub.")
    p.add_argument("--input-mode", choices=["File", "FastFile"], default="File",
                   help="S3 channel input mode. FastFile streams objects on first access (good for large video datasets)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=None, help="Per-GPU micro batch (preset default if omitted)")
    p.add_argument("--peft", action="store_true",
                   help="LoRA fine-tune (pi05) instead of full fine-tune; for 24-48 GB GPUs")
    p.add_argument("--distributed", choices=["ddp", "fsdp"], default="ddp",
                   help="Multi-GPU mode via accelerate (LeRobot 0.6.1). fsdp needs FSDP_WRAP_CLS in the preset")
    p.add_argument("--save-freq", type=int, default=2500)
    p.add_argument("--num-workers", type=int, default=None, help="Dataloader workers per process (default: vCPUs per GPU, max 8)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="Hugging Face token (or set HF_TOKEN). Needed for GR00T (Cosmos-Reason2-2B gate) and pi0.5 "
                        "(PaliGemma gate). Passed to the job as the HF_TOKEN environment variable.")
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging (needs WANDB_API_KEY in the environment)")
    p.add_argument("--subnets", default=None,
                   help="Comma-separated subnet IDs. Required for EFA on multi-node p5/g7e.48xlarge jobs; "
                        "all subnets should be in AZs that offer the instance type")
    p.add_argument("--security-group-ids", default=None,
                   help="Comma-separated SG IDs; the SG must allow all inbound/outbound traffic to itself (EFA)")
    p.add_argument("--job-name", default=None, help="Checkpoint prefix; reuse the same name to resume a run")
    p.add_argument("--extra-args", default="", help="Extra lerobot-train flags appended verbatim. Use the = form because the value starts with '--': "
                        "--extra-args='--policy.tune_diffusion_model=false --dataset.revision=main'")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="Extra environment variable(s) for the container, e.g. --env OMP_NUM_THREADS=2")
    p.add_argument("--no-wait", action="store_true", help="Return right after submission")
    p.add_argument("--dry-run", action="store_true", help="Print the configuration and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image_uri or not args.role:
        raise SystemExit("--image-uri and --role are required (or TRAINING_IMAGE_URI / SAGEMAKER_ROLE_ARN)")

    session = Session(boto_session=boto3.Session())
    bucket = args.bucket or session.default_bucket()
    instance_types = [t.strip() for t in args.instance_type.split(",") if t.strip()]
    if len(instance_types) > 1 and args.spot:
        raise SystemExit("--spot cannot be combined with an instance type list")
    if args.spot and args.max_wait is not None and args.max_wait <= args.max_runtime:
        raise SystemExit("--max-wait must be larger than --max-runtime")
    job_name = args.job_name or f"lerobot-{args.model}{'-lora' if args.peft else ''}-{instance_types[0].replace('ml.', '').replace('.', '-')}-x{args.instance_count}"
    run_id = f"{job_name}-{time.strftime('%Y%m%d-%H%M%S')}"

    # Everything under src/ (train.sh + presets/) is synced into /opt/ml/code at job start.
    source_code = SourceCode(
        source_dir=os.path.join(HERE, "src"),
        command=f"bash train.sh --model {args.model} {args.extra_args}".strip(),
    )

    # g5/g6/g6e/g7e sizes with fixed local NVMe cap VolumeSizeInGB (xlarge = 250 GB); clamp to avoid a
    # ValidationException. Other sizes have >= 450 GB.
    volume = min(args.volume_size, 250) if any(t.endswith(".xlarge") for t in instance_types) else args.volume_size
    compute = Compute(
        instance_type=instance_types[0] if len(instance_types) == 1 else None,
        instance_preferences=[InstancePreference(instance_type=t) for t in instance_types] if len(instance_types) > 1 else None,
        instance_count=args.instance_count,
        volume_size_in_gb=volume,
        keep_alive_period_in_seconds=args.keep_alive or None,
        enable_managed_spot_training=args.spot or None,
    )

    environment = {
        "JOB_NAME": job_name,
        "DATASET_REPO_ID": args.dataset_repo_id,
        "STEPS": str(args.steps),
        "SAVE_FREQ": str(args.save_freq),
        "WANDB_ENABLE": "true" if args.wandb else "false",
    }
    if args.batch_size is not None:
        environment["BATCH_SIZE"] = str(args.batch_size)
    if args.peft:
        environment["PEFT"] = "1"
    environment["DISTRIBUTED"] = args.distributed
    if args.num_workers:
        environment["NUM_WORKERS"] = str(args.num_workers)
    if args.hf_token:
        environment["HF_TOKEN"] = args.hf_token
    for kv in args.env:
        k, _, v = kv.partition("=")
        environment[k] = v

    # Checkpoints are synced to S3 continuously; relaunch with the same --job-name to resume.
    checkpoint_config = CheckpointConfig(
        s3_uri=f"s3://{bucket}/lerobot-vla/checkpoints/{job_name}",
        local_path="/opt/ml/checkpoints",
    )
    output_config = OutputDataConfig(s3_output_path=f"s3://{bucket}/lerobot-vla/output")
    stopping = StoppingCondition(max_runtime_in_seconds=args.max_runtime,
                                 max_wait_time_in_seconds=(args.max_wait or 2 * args.max_runtime) if args.spot else None,
                                 max_pending_time_in_seconds=args.max_pending)

    networking = None
    if args.subnets or args.security_group_ids:
        if not (args.subnets and args.security_group_ids):
            raise SystemExit("--subnets and --security-group-ids must be given together")
        networking = Networking(
            subnets=args.subnets.split(","),
            security_group_ids=args.security_group_ids.split(","),
        )
    elif args.instance_count > 1:
        print("WARNING: multi-node without --subnets/--security-group-ids: NCCL will use TCP, not EFA.")

    input_data = None
    if args.dataset_s3_uri:
        input_data = [InputData(channel_name="dataset", data_source=args.dataset_s3_uri)]

    if args.dry_run:
        print(f"job:          {run_id}")
        print(f"image:        {args.image_uri}")
        print(f"compute:      {args.instance_count} x {args.instance_type}, {args.volume_size} GB{', Spot' if args.spot else ''}")
        print(f"command:      {source_code.command}")
        print(f"checkpoints:  {checkpoint_config.s3_uri}")
        print(f"output:       {output_config.s3_output_path}")
        print(f"input:        {args.dataset_s3_uri or 'Hugging Face Hub (' + args.dataset_repo_id + ')'}")
        print(f"networking:   {networking.model_dump(exclude_none=True) if networking else 'SageMaker-managed (no EFA)'}")
        print(f"environment:  { {k: ('***' if k == 'HF_TOKEN' else v) for k, v in environment.items()} }")
        return

    trainer = ModelTrainer(
        training_image=args.image_uri,
        source_code=source_code,
        compute=compute,
        networking=networking,
        environment=environment,
        checkpoint_config=checkpoint_config,
        output_data_config=output_config,
        stopping_condition=stopping,
        role=args.role,
        base_job_name=job_name,
        training_input_mode=args.input_mode,
        sagemaker_session=session,
    ).with_metric_definitions(METRIC_DEFINITIONS)

    print(f"Submitting {run_id} on {args.instance_count} x {args.instance_type} ...")
    trainer.train(input_data_config=input_data, wait=not args.no_wait)

    if args.no_wait:
        print("Monitor with:")
        print(f"  aws sagemaker describe-training-job --training-job-name <name starting with {job_name}> "
              "--query '{Status: TrainingJobStatus, Secondary: SecondaryStatus}'")
        print("Logs: CloudWatch log group /aws/sagemaker/TrainingJobs")


if __name__ == "__main__":
    main()

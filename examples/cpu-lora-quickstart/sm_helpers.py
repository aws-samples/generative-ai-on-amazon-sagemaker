"""Notebook-side helpers shared by 02-04 (01 spells the same steps out inline): session setup, a CPU training job
in one of the three capacity modes, waiting for jobs, and reading their metrics."""
import json
import time

import boto3
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.shapes import InstancePreference
from sagemaker.core.training.configs import Compute, OutputDataConfig, SourceCode, StoppingCondition
from sagemaker.train import ModelTrainer

INSTANCE_TYPE = "ml.c7i.4xlarge"
FALLBACK_TYPES = ["ml.c7i.4xlarge", "ml.m7i.4xlarge", "ml.c6i.4xlarge", "ml.m6i.4xlarge", "ml.m5.4xlarge"]
IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.8-cpu-py312"


class Lab:
    def __init__(self):
        self.sess = Session()
        self.region = self.sess.boto_region_name
        try:
            self.role = get_execution_role()
        except ValueError:
            self.role = boto3.client("iam").get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]
        self.bucket = self.sess.default_bucket()
        self.prefix = "/".join(p for p in (self.sess.default_bucket_prefix, "cpu-lora-quickstart") if p)
        self.sm = boto3.client("sagemaker", region_name=self.region)
        self.s3 = boto3.client("s3", region_name=self.region)
        print(f"role:   {self.role}\nregion: {self.region}\noutput: s3://{self.bucket}/{self.prefix}/")

    def fallback_types(self):
        """FALLBACK_TYPES minus types without a training quota (SageMaker rejects lists containing one)."""
        try:
            sq = boto3.client("service-quotas", region_name=self.region)
            quotas = {q["QuotaName"]: q["Value"] for page in sq.get_paginator("list_service_quotas")
                      .paginate(ServiceCode="sagemaker") for q in page["Quotas"]}
            return [t for t in FALLBACK_TYPES if quotas.get(f"{t} for training job usage", 0) >= 1] or FALLBACK_TYPES
        except Exception as e:
            print(f"quota check skipped: {e}")
            return FALLBACK_TYPES

    def submit(self, name, entry_script, hyperparameters, capacity="fallback", requirements="requirements.txt"):
        """Starts a CPU training job running scripts/<entry_script>; capacity: on-demand | spot | fallback."""
        if capacity == "fallback":
            compute = Compute(instance_preferences=[InstancePreference(instance_type=t) for t in self.fallback_types()],
                              instance_count=1, volume_size_in_gb=30)
        else:
            compute = Compute(instance_type=INSTANCE_TYPE, instance_count=1, volume_size_in_gb=30,
                              enable_managed_spot_training=capacity == "spot")
        trainer = ModelTrainer(
            training_image=IMAGE.format(region=self.region),
            source_code=SourceCode(source_dir="scripts", requirements=requirements, entry_script=entry_script),
            hyperparameters=hyperparameters,
            compute=compute,
            stopping_condition=StoppingCondition(max_runtime_in_seconds=3600,
                                                 max_wait_time_in_seconds=7200 if capacity == "spot" else None),
            output_data_config=OutputDataConfig(s3_output_path=f"s3://{self.bucket}/{self.prefix}/jobs",
                                                compression_type="NONE"),
            base_job_name=f"cpu-lora-{name}",
            role=self.role,
            sagemaker_session=self.sess,
        )
        trainer.train(wait=False)
        job = trainer._latest_training_job.training_job_name
        print(f"submitted: {job} ({capacity})")
        return job

    def latest(self, name):
        """Most recent job started with submit(name, ...): use after a kernel restart."""
        prefix = f"cpu-lora-{name}-"
        jobs = self.sm.list_training_jobs(NameContains=prefix, SortBy="CreationTime", SortOrder="Descending",
                                          MaxResults=20)["TrainingJobSummaries"]
        # "dpo" must not match "dpo-sft-...": the name continues with the job timestamp
        return next(j["TrainingJobName"] for j in jobs if j["TrainingJobName"][len(prefix):][:1].isdigit())

    def wait(self, *jobs, poll=15):
        """Prints status changes until all jobs finish; returns their descriptions. Safe to interrupt and re-run."""
        last = {}
        while True:
            desc = {j: self.sm.describe_training_job(TrainingJobName=j) for j in jobs}
            for j, d in desc.items():
                line = f"{d['TrainingJobStatus']}/{d['SecondaryStatus']}"
                if last.get(j) != line:
                    print(time.strftime("%H:%M:%S"), j, line, flush=True)
                    last[j] = line
            if all(d["TrainingJobStatus"] in ("Completed", "Failed", "Stopped") for d in desc.values()):
                break
            time.sleep(poll)
        for j, d in desc.items():
            if d["TrainingJobStatus"] != "Completed":
                print(f"{j}: {d.get('FailureReason', d['TrainingJobStatus'])}")
            else:
                rc = d["ResourceConfig"]
                print(f"{j}: {rc.get('SelectedInstanceType') or rc.get('InstanceType')}, "
                      f"{(d['TrainingEndTime'] - d['CreationTime']).total_seconds() / 60:.1f} min end to end, "
                      f"billed {d['BillableTimeInSeconds']} s")
        return desc

    def metrics(self, job):
        d = self.sm.describe_training_job(TrainingJobName=job)
        b, k = (d["ModelArtifacts"]["S3ModelArtifacts"].rstrip("/") + "/")[5:].split("/", 1)
        return json.loads(self.s3.get_object(Bucket=b, Key=k + "metrics.json")["Body"].read())

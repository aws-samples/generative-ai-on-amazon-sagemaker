"""Notebook-side helpers. MLflow setup (resolve_mlflow, mlflow_environment, mlflow_url) is used by all labs; `Lab`
(session, a CPU training job in one of the three capacity modes, waiting, metrics) by 02-04, while 01 spells the
same steps out inline."""
import json
import time

import boto3
from botocore.config import Config
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.shapes import InstancePreference
from sagemaker.core.training.configs import Compute, OutputDataConfig, SourceCode, StoppingCondition
from sagemaker.train import ModelTrainer

INSTANCE_TYPE = "ml.c7i.4xlarge"
SPOT_INSTANCE_TYPE = "ml.m5.4xlarge"  # Spot capacity differs by type (EC2 Spot placement score: see 01-sft.ipynb)
FALLBACK_TYPES = ["ml.c7i.4xlarge", "ml.m7i.4xlarge", "ml.c6i.4xlarge", "ml.m6i.4xlarge", "ml.m5.4xlarge"]
IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.8-cpu-py312"
MLFLOW_EXPERIMENT = "cpu-fine-tuning-techniques"


def resolve_mlflow(sm_client, session, arn=""):
    """MLflow App ARN to log to: `arn` if set, else the account's app (created if there is none: several minutes,
    first time only). None turns MLflow off. Never blocks the lab: on any error it prints why and returns None."""
    if arn is None:
        return None
    from workshop_mlflow import resolve_app

    try:
        return resolve_app(sm_client, arn, session=session, native=True, app_name="serverless-customization-workshop")
    except Exception as e:
        print(f"MLflow off: {e}")
        return None


def mlflow_environment(arn):
    """Job environment variables that make the training scripts log to MLflow (the URI must be the App ARN)."""
    return {"MLFLOW_TRACKING_URI": arn, "MLFLOW_EXPERIMENT_NAME": MLFLOW_EXPERIMENT} if arn else {}


def mlflow_url(sm_client, arn):
    """Link to the MLflow UI, valid for a few minutes."""
    return sm_client.create_presigned_mlflow_app_url(Arn=arn)["AuthorizedUrl"] if arn else None


class Lab:
    def __init__(self, mlflow_app_arn=""):
        self.sess = Session()
        self.region = self.sess.boto_region_name
        try:
            self.role = get_execution_role()
        except ValueError:
            self.role = boto3.client("iam").get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]
        self.bucket = self.sess.default_bucket()
        self.prefix = "/".join(p for p in (self.sess.default_bucket_prefix, "cpu-fine-tuning-techniques") if p)
        self.sm = boto3.client("sagemaker", region_name=self.region)
        self.s3 = boto3.client("s3", region_name=self.region)
        self.mlflow_arn = resolve_mlflow(self.sm, self.sess, mlflow_app_arn)
        print(f"role:   {self.role}\nregion: {self.region}\noutput: s3://{self.bucket}/{self.prefix}/\n"
              f"MLflow: {self.mlflow_arn or 'off'}")

    def fallback_types(self):
        """FALLBACK_TYPES minus types without a training quota (SageMaker rejects lists containing one)."""
        try:
            # adaptive retries: notebooks started together can hit the Service Quotas rate limit
            sq = boto3.client("service-quotas", region_name=self.region,
                              config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))
            # applied values override defaults; a quota still at its default is only in the defaults list
            quotas = {}
            for op in ("list_aws_default_service_quotas", "list_service_quotas"):
                for page in sq.get_paginator(op).paginate(ServiceCode="sagemaker"):
                    quotas.update({q["QuotaName"]: q["Value"] for q in page["Quotas"]})
            return [t for t in FALLBACK_TYPES if quotas.get(f"{t} for training job usage", 0) >= 1] or FALLBACK_TYPES
        except Exception as e:
            print(f"quota check skipped: {e}")
            return FALLBACK_TYPES

    def submit(self, name, entry_script, hyperparameters, capacity="fallback", requirements="requirements.txt",
               instance_count=1, distributed=None):
        """Starts a CPU training job running scripts/<entry_script>; capacity: on-demand | spot | fallback.
        instance_count > 1 with distributed=Torchrun() runs one process per instance (notebook 05)."""
        if capacity == "fallback":
            compute = Compute(instance_preferences=[InstancePreference(instance_type=t) for t in self.fallback_types()],
                              instance_count=instance_count, volume_size_in_gb=30)
        else:
            compute = Compute(instance_type=SPOT_INSTANCE_TYPE if capacity == "spot" else INSTANCE_TYPE,
                              instance_count=instance_count, volume_size_in_gb=30, enable_managed_spot_training=capacity == "spot")
        trainer = ModelTrainer(
            training_image=IMAGE.format(region=self.region),
            source_code=SourceCode(source_dir="scripts", requirements=requirements, entry_script=entry_script),
            hyperparameters=hyperparameters,
            environment=mlflow_environment(self.mlflow_arn),
            compute=compute,
            distributed=distributed,
            stopping_condition=StoppingCondition(max_runtime_in_seconds=3600,
                                                 max_wait_time_in_seconds=7200 if capacity == "spot" else None),
            output_data_config=OutputDataConfig(s3_output_path=f"s3://{self.bucket}/{self.prefix}/jobs",
                                                compression_type="NONE"),
            base_job_name=f"cpu-ft-{name}",
            role=self.role,
            sagemaker_session=self.sess,
        )
        trainer.train(wait=False)
        job = trainer._latest_training_job.training_job_name
        print(f"submitted: {job} ({capacity})")
        return job

    def latest(self, name):
        """Most recent job started with submit(name, ...): use after a kernel restart."""
        prefix = f"cpu-ft-{name}-"
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

    def metrics(self, job, filename="metrics.json"):
        """A JSON file the job wrote to /opt/ml/model (metrics.json for 02-04, training_metrics.json for 05)."""
        d = self.sm.describe_training_job(TrainingJobName=job)
        b, k = (d["ModelArtifacts"]["S3ModelArtifacts"].rstrip("/") + "/")[5:].split("/", 1)
        return json.loads(self.s3.get_object(Bucket=b, Key=k + filename)["Body"].read())

    def mlflow_url(self):
        return mlflow_url(self.sm, self.mlflow_arn)

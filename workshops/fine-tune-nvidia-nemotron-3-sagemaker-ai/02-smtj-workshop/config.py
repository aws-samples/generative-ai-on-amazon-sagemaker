import re

# Base model for this lab: a Hugging Face model id, pulled from the Hub inside the
# Training job.
BASE_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"

# A Hugging Face id is not a legal SageMaker resource name: every job, model, endpoint and
# endpoint config has to match
#
#     ([a-zA-Z0-9]([a-zA-Z0-9-]){0,62})(?<!-)
#
# so the `nvidia/` org prefix alone makes `BASE_MODEL_ID` unusable - `ValidationException:
# Member must satisfy regular expression pattern`. Drop the org and collapse anything that
# is not alphanumeric into a hyphen, once, here, so every notebook derives the same names.
MODEL_SLUG = re.sub(r"[^a-zA-Z0-9]+", "-", BASE_MODEL_ID.split("/")[-1]).strip("-")

# Fixed dataset / resource names used across the notebooks
DATASET_PREFIX = "contractnli-nda-review"

# Where notebook 1 writes the JSONL splits and notebook 2 reads them from, as Training job
# channels.
DATA_PREFIX = f"datasets/{DATASET_PREFIX}"

# Base name for the training job. SageMaker appends a timestamp, so the job that runs is
# `<TRAIN_JOB_PREFIX>-<timestamp>` and its artifacts land under
# `s3://<bucket>/[<prefix>/]<TRAIN_JOB_PREFIX>/<full-job-name>/output/model.tar.gz`.
#
# Derived once, here, because notebook 2 uses it as `base_job_name` and notebook 3 uses it
# to find the last completed job.
TRAIN_JOB_PREFIX = f"train-{MODEL_SLUG}-sft"

"""Runtime configuration for the Octank Gift Concierge workshop.

Everything is resolved from the current session so the same code runs in us-east-1 or us-west-2
with no edits: region from the boto3 session, bucket = SageMaker default bucket for that region,
S3 prefix = "gift-concierge" (override with OCTANK_S3_PREFIX).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]          # repository root
DATA_DIR = REPO_ROOT / "data"                            # seed data shipped in the repo
WORK_DIR = REPO_ROOT / "work"                            # local scratch (git-ignored)
# Version-controlled runs: their LLM caches and state.json live here (derived files stay in work/).
TRACKED_RUNS = {"instructor": REPO_ROOT / "instructor"}
POLICY_DIR = Path(__file__).resolve().parent / "policy"
JUDGE_DIR = REPO_ROOT / "judge"

# --- Models -------------------------------------------------------------------------------------
# Trained / served model (SageMaker Hub id).
BASE_MODEL_ID = os.environ.get("OCTANK_BASE_MODEL", "huggingface-vlm-qwen3-5-4b")

# Bedrock models. Everything that produces training data or training signal is open-weight and
# available in both us-east-1 and us-west-2. Claude is used only as the evaluation judge.
GENERATOR_MODEL = "openai.gpt-oss-120b-1:0"     # catalog, requests, gap-fill, teacher, gold drafts
LABELER_MODEL = "openai.gpt-oss-20b-1:0"        # auto-labeling
SAFETY_MODEL = "openai.gpt-oss-safeguard-20b"   # toxicity filter (policy in policy/toxicity.md)
TASTE_MODEL = "openai.gpt-oss-20b-1:0"          # LLM "taste" inside the hybrid RLVR reward Lambda
JUDGE_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"  # LLM-as-a-Judge evaluation (scores only)

# Execution role for the reward Lambda. Unset: octank.reward creates octank-hybrid-reward-lambda-role
# (needs iam:CreateRole/PutRolePolicy/AttachRolePolicy). Set it to a pre-provisioned role (e.g. from
# Workshop Studio) trusted by lambda.amazonaws.com with Bedrock Converse on the taste models.
REWARD_ROLE_ARN = os.environ.get("OCTANK_REWARD_ROLE_ARN") or None

# --- Data -----------------------------------------------------------------------------------------
TEACHER_MAX_THINKING_CHARS = 4500   # teacher reasoning kept only if <= ~1,100 tokens
SFT_MAX_COMPLETION_TOKENS = 2048    # SFT targets longer than this are dropped
RL_BATCH = 128                      # RLVR global_batch_size (prompts); steps/epoch = floor(train / 128)

# --- Training: SFT -> RLVR, identical for instructor and attendee (only the data size differs) ---------
LORA_RANK, LORA_ALPHA = 32, 64
SFT_HP = {"max_epochs": 2, "dataset_max_len": 6144, "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA}
# KL off: with an SFT starting point the KL reference pulled the policy back toward the base model.
# train_val_split_ratio 0.9 (1.0 leaves an empty validation set and the trainer fails).
RLVR_HP = {"learning_rate": 2e-5, "max_epochs": 1, "rollout_temperature": 0.6, "rollout_n": 8,
           "max_response_length": 1536, "use_kl_loss": False, "train_val_split_ratio": 0.9,
           "global_batch_size": RL_BATCH,     # = the recipe default (128); set explicitly
           "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA}

# --- Evaluation -----------------------------------------------------------------------------------
# Evaluate the way we serve: Qwen thinking-mode sampling (greedy decoding makes the model loop).
EVAL_MAX_NEW_TOKENS = 4096          # 40-candidate prompts (~2.5K tokens)
EVAL_MAX_NEW_TOKENS_80 = 3072       # 80-candidate prompts (~4.7K tokens) must fit MAX_MODEL_LEN
EVAL_TEMPERATURE, EVAL_TOP_P = 0.6, 0.95

# --- Serving --------------------------------------------------------------------------------------
INSTANCE_TYPE = "ml.g7.2xlarge"
MAX_MODEL_LEN = 8192                # prompt + answer


@dataclass
class Settings:
    region: str
    account: str
    bucket: str
    prefix: str
    role_arn: str
    run_name: str = "instructor"               # "instructor" or an attendee id
    extra: dict = field(default_factory=dict)

    @property
    def s3_root(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}/{self.run_name}"

    def s3(self, *parts: str) -> str:
        return "/".join([self.s3_root, *parts])

    @property
    def tracked_dir(self) -> Path | None:
        """Version-controlled folder for this run's caches + state (instructor), else None."""
        return TRACKED_RUNS.get(self.run_name)

    def local(self, *parts: str) -> Path:
        p = WORK_DIR / self.run_name
        for part in parts:
            p = p / part
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _resolve_role(session) -> str:
    """Execution role in Studio; falls back to the caller identity's role ARN."""
    try:
        from sagemaker.core.helper.session_helper import get_execution_role
        return get_execution_role()
    except Exception:
        arn = session.client("sts").get_caller_identity()["Arn"]
        if ":assumed-role/" in arn:
            acct = arn.split(":")[4]
            role = arn.split("/")[1]
            return f"arn:aws:iam::{acct}:role/{role}"
        return arn


@lru_cache(maxsize=4)
def get_settings(run_name: str = "instructor") -> Settings:
    import boto3

    os.environ.setdefault("AWS_SDK_UA_APP_ID", "AWSSkill-SageMaker")
    session = boto3.Session()
    region = session.region_name or os.environ.get("AWS_REGION") or "us-west-2"
    account = session.client("sts").get_caller_identity()["Account"]
    bucket = os.environ.get("OCTANK_BUCKET") or f"sagemaker-{region}-{account}"
    prefix = os.environ.get("OCTANK_S3_PREFIX", "gift-concierge")
    return Settings(region=region, account=account, bucket=bucket, prefix=prefix,
                    role_arn=_resolve_role(session), run_name=run_name)


def judge_profile_id(region: str) -> str:
    """Claude on Bedrock is invoked via a `us.` inference profile."""
    return f"us.{JUDGE_MODEL}"

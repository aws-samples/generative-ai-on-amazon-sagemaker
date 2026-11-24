"""Deploy the hybrid RLVR reward Lambda (judge/reward_lambda.py) - identical for instructor and
attendees: same code, runtime, timeout, memory, environment and Bedrock permissions.

Why not the SDK's Evaluator.create(source="lambda_function.py")? It creates the function with
Lambda defaults (3 s timeout, 128 MB) and the caller's role, which can't make the parallel
gpt-oss-20b taste calls. So we create the function ourselves and register it by ARN.

    from octank import reward
    arn = reward.deploy_hybrid_lambda(region)               # create or update, returns the ARN
    reward.verify_matches(arn, region)                      # deployed code == judge/ file

Execution role: `role_arn`, else config.REWARD_ROLE_ARN (env OCTANK_REWARD_ROLE_ARN), else
ensure_role() creates one (needs IAM create permissions). Deploying always needs
lambda:CreateFunction/UpdateFunction* and iam:PassRole on that role.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from . import config
from .config import JUDGE_DIR

SOURCE = JUDGE_DIR / "reward_lambda.py"
FUNCTION_NAME = "SageMaker-octank-hybrid-reward"     # prefix "SageMaker-" matches the SDK convention
ROLE_NAME = "octank-hybrid-reward-lambda-role"
RUNTIME, TIMEOUT, MEMORY = "python3.12", 300, 1024
ENV = {"TASTE_MODEL": config.TASTE_MODEL, "TASTE_WEIGHT": "0.30",
       "TASTE_WORKERS": "32", "TASTE_BUDGET_S": "120"}
TASTE_MODELS = ["openai.gpt-oss-20b-1:0", "openai.gpt-oss-120b-1:0", "qwen.qwen3-32b-v1:0"]


def source_sha256() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def _zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(SOURCE, "lambda_function.py")
    return buf.getvalue()


def ensure_role(role_name: str = ROLE_NAME) -> str:
    """Lambda execution role: basic logging + Bedrock Converse on the taste models only."""
    try:
        return _ensure_role(role_name)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("AccessDenied", "AccessDeniedException", "UnauthorizedOperation"):
            raise PermissionError(
                f"Can't create the reward Lambda's IAM role ({e.response['Error'].get('Message', e)}). "
                "Use a pre-provisioned role instead: set OCTANK_REWARD_ROLE_ARN (or pass role_arn=) to a "
                "role trusted by lambda.amazonaws.com with AWSLambdaBasicExecutionRole and "
                f"bedrock:InvokeModel/Converse on {', '.join(TASTE_MODELS)}.") from e
        raise


def _ensure_role(role_name: str) -> str:
    iam = boto3.client("iam")
    try:
        return iam.get_role(RoleName=role_name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass
    arn = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps({
        "Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
                                                "Action": "sts:AssumeRole"}]}))["Role"]["Arn"]
    iam.attach_role_policy(RoleName=role_name,
                           PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    iam.put_role_policy(RoleName=role_name, PolicyName="bedrock-taste", PolicyDocument=json.dumps({
        "Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": ["bedrock:InvokeModel", "bedrock:Converse"],
                                                "Resource": [f"arn:aws:bedrock:*::foundation-model/{m}" for m in TASTE_MODELS]}]}))
    time.sleep(12)                                    # IAM propagation before Lambda can assume it
    return arn


def deploy_hybrid_lambda(region: str, function_name: str = FUNCTION_NAME, role_arn: str | None = None) -> str:
    """Create or update the reward Lambda with the reference configuration. Returns its ARN."""
    lam = boto3.client("lambda", region_name=region)
    role_arn = role_arn or config.REWARD_ROLE_ARN or ensure_role()
    cfg = dict(Runtime=RUNTIME, Handler="lambda_function.lambda_handler", Role=role_arn,
               Timeout=TIMEOUT, MemorySize=MEMORY, Environment={"Variables": ENV})
    try:
        lam.get_function(FunctionName=function_name)
        lam.update_function_code(FunctionName=function_name, ZipFile=_zip())
        lam.get_waiter("function_updated_v2").wait(FunctionName=function_name)
        lam.update_function_configuration(FunctionName=function_name, **cfg)
    except lam.exceptions.ResourceNotFoundException:
        lam.create_function(FunctionName=function_name, Code={"ZipFile": _zip()}, **cfg)
    lam.get_waiter("function_active_v2").wait(FunctionName=function_name)
    lam.get_waiter("function_updated_v2").wait(FunctionName=function_name)
    return lam.get_function_configuration(FunctionName=function_name)["FunctionArn"]


def deployed_config(arn: str, region: str) -> dict:
    """Code hash + configuration of a deployed reward Lambda (for comparing two deployments)."""
    import urllib.request
    lam = boto3.client("lambda", region_name=region)
    f = lam.get_function(FunctionName=arn)
    code = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(f["Code"]["Location"]).read())).read("lambda_function.py")
    c = f["Configuration"]
    return {"code_sha256": hashlib.sha256(code).hexdigest(), "runtime": c["Runtime"], "handler": c["Handler"],
            "timeout": c["Timeout"], "memory": c["MemorySize"], "env": c.get("Environment", {}).get("Variables", {})}


def verify_matches(arn: str, region: str) -> dict:
    """Raise unless the deployed function is the judge/ source with the reference configuration."""
    d = deployed_config(arn, region)
    want = {"code_sha256": source_sha256(), "runtime": RUNTIME, "handler": "lambda_function.lambda_handler",
            "timeout": TIMEOUT, "memory": MEMORY, "env": ENV}
    diff = {k: (d[k], v) for k, v in want.items() if d[k] != v}
    if diff:
        raise RuntimeError(f"reward Lambda differs from the reference: {diff}")
    return d

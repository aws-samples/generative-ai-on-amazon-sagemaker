"""Real-time endpoint with sagemaker-core resources (no ModelBuilder).

Topology (one ml.g7.2xlarge, inference-component mode):
  Endpoint  <- EndpointConfig(execution_role_arn=..., variant without model_name)
    |- base IC     : Qwen base model in the LMI (vLLM) container, LoRA enabled
    |- adapter IC  : instructor SFT             (from the repo's adapters/, base_inference_component_name = base IC)
    |- adapter IC  : instructor SFT -> RLVR
    |- adapter IC  : your SFT / SFT -> RLVR     (optional, when your jobs finish; up to 4 adapters resident)

The container image + environment come from the base model's LoRA recipe hosting config in the
SageMaker Hub (published for larger instances); we keep everything and only resize it for a
single 24 GB A10G (tensor parallel 1, bounded context, few resident adapters).
"""
from __future__ import annotations

import json
import time

import boto3
from botocore.exceptions import ClientError

from .config import BASE_MODEL_ID, INSTANCE_TYPE, MAX_MODEL_LEN, REPO_ROOT

HUB = "SageMakerPublicHub"


def hub_hosting(region: str, model_id: str = BASE_MODEL_ID, technique: str = "SFT") -> dict:
    """Image, env and base artifact URI for LoRA serving, from the Hub recipe hosting config."""
    sm = boto3.client("sagemaker", region_name=region)
    doc = json.loads(sm.describe_hub_content(HubName=HUB, HubContentType="Model",
                                             HubContentName=model_id)["HubContentDocument"])
    recipe = next(r for r in doc["RecipeCollection"]
                  if r.get("CustomizationTechnique") == technique and r.get("Peft") == "LORA")
    hc = next((h for h in recipe["HostingConfigs"] if h.get("Profile") == "Default"),
              recipe["HostingConfigs"][0])
    return {"image": hc["EcrAddress"], "env": dict(hc.get("Environment") or {}),
            "artifact_uri": doc["HostingArtifactUri"], "recipe": recipe["Name"],
            "published_instance": hc["InstanceType"]}


def g5_env(env: dict) -> dict:
    """Resize the published hosting env for one GPU (A10G 24 GB / H100 80 GB).

    - OPTION_ENFORCE_EAGER=false: the published config runs eager (no CUDA graphs); measured
      ~12 tok/s per stream on a p5.4xlarge that way - far too slow for thinking responses.
    - OPTION_MAX_LORA_RANK = our LoRA rank (32) instead of 128 (less LoRA kernel overhead).
    - max model len 8192 = prompt (~2.5K at 40 candidates, ~4.7K at 80) + response budget."""
    from .config import LORA_RANK
    env = dict(env)
    env.update({
        "OPTION_ENFORCE_EAGER": "false",
        "OPTION_MAX_LORA_RANK": str(LORA_RANK),
        "OPTION_TENSOR_PARALLEL_DEGREE": "1",
        "OPTION_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
        "OPTION_MAX_ROLLING_BATCH_SIZE": "8",
        "OPTION_GPU_MEMORY_UTILIZATION": "0.90",
        "OPTION_MAX_LORAS": "4",          # up to 4 adapters resident on the GPU
        "OPTION_MAX_CPU_LORAS": "8",
        "SAGEMAKER_MAX_NUMBER_OF_ADAPTERS_IN_MEMORY": "8",
    })
    return env


def _exists(fn, **kw) -> bool:
    try:
        fn(**kw)
        return True
    except ClientError:
        return False


def create_endpoint(name: str, role_arn: str, region: str, wait: bool = True,
                    instance_type: str = INSTANCE_TYPE, ml_reservation_arn: str | None = None,
                    instance_pools: list[str] | None = None) -> dict:
    """EndpointConfig + Endpoint in inference-component mode (no model yet; see add_model).
    Idempotent: existing resources are kept.

    ml_reservation_arn: optional training plan / reserved capacity (target resource "endpoint")
    to place the instance on reserved capacity instead of on-demand."""
    from sagemaker.core.resources import Endpoint, EndpointConfig
    from sagemaker.core.shapes import ProductionVariant, ProductionVariantCapacityReservationConfig

    sm = boto3.client("sagemaker", region_name=region)
    if not _exists(sm.describe_endpoint_config, EndpointConfigName=name):
        pv = dict(variant_name="AllTraffic", initial_instance_count=1,
                  container_startup_health_check_timeout_in_seconds=1800,
                  model_data_download_timeout_in_seconds=1800)
        if instance_pools:          # priority-ordered fallback: first type with capacity wins
            from sagemaker.core.shapes import InstancePool
            pv["instance_pools"] = [InstancePool(instance_type=t, priority=i + 1)
                                    for i, t in enumerate(instance_pools)]
        else:
            pv["instance_type"] = instance_type
        if ml_reservation_arn:
            pv["capacity_reservation_config"] = ProductionVariantCapacityReservationConfig(
                capacity_reservation_preference="capacity-reservations-only",
                ml_reservation_arn=ml_reservation_arn)
        EndpointConfig.create(endpoint_config_name=name, execution_role_arn=role_arn,
                              region=region, production_variants=[ProductionVariant(**pv)])
    if not _exists(sm.describe_endpoint, EndpointName=name):
        Endpoint.create(endpoint_name=name, endpoint_config_name=name, region=region)
    if wait:
        Endpoint.get(endpoint_name=name, region=region).wait_for_status(
            target_status="InService", poll=30, timeout=3600)
    return {"endpoint": name}


def add_model(endpoint: str, role_arn: str, region: str, cpu_cores: int = 2, memory_mb: int = 8192,
              model_tag: str = "", wait: bool = True) -> dict:
    """The base model as an inference component on `endpoint` (the counterpart of add_adapter).
    Creates the SageMaker Model from the Hub's hosting recipe if needed. Idempotent.

    cpu_cores / memory_mb are this component's reservation on the instance; adapters share it."""
    from sagemaker.core.resources import InferenceComponent, Model
    from sagemaker.core.shapes import (ContainerDefinition,
                                       InferenceComponentComputeResourceRequirements,
                                       InferenceComponentRuntimeConfig,
                                       InferenceComponentSpecification, ModelAccessConfig,
                                       ModelDataSource, S3ModelDataSource)

    h = hub_hosting(region)
    model_name = f"{endpoint}-base-{model_tag}" if model_tag else f"{endpoint}-base"
    base_ic = f"{endpoint}-base-ic"
    sm = boto3.client("sagemaker", region_name=region)

    if not _exists(sm.describe_model, ModelName=model_name):
        Model.create(
            model_name=model_name, execution_role_arn=role_arn, region=region,
            primary_container=ContainerDefinition(
                image=h["image"], environment=g5_env(h["env"]),
                model_data_source=ModelDataSource(s3_data_source=S3ModelDataSource(
                    s3_uri=h["artifact_uri"], s3_data_type="S3Prefix", compression_type="None",
                    model_access_config=ModelAccessConfig(accept_eula=False)))))
    if not _exists(sm.describe_inference_component, InferenceComponentName=base_ic):
        InferenceComponent.create(
            inference_component_name=base_ic, endpoint_name=endpoint, variant_name="AllTraffic",
            region=region,
            specification=InferenceComponentSpecification(
                model_name=model_name,
                compute_resource_requirements=InferenceComponentComputeResourceRequirements(
                    number_of_accelerator_devices_required=1, number_of_cpu_cores_required=cpu_cores,
                    min_memory_required_in_mb=memory_mb)),
            runtime_config=InferenceComponentRuntimeConfig(copy_count=1))
    if wait:
        wait_ic(base_ic, region)
    return {"base_ic": base_ic, "image": h["image"], "recipe": h["recipe"]}


def add_adapter(endpoint: str, base_ic: str, ic_name: str, adapter_s3_uri: str, region: str,
                wait: bool = True) -> str:
    """Attach a LoRA adapter (S3 prefix with adapter_config.json + weights) as its own IC."""
    from sagemaker.core.resources import InferenceComponent
    from sagemaker.core.shapes import (InferenceComponentContainerSpecification,
                                       InferenceComponentSpecification)

    InferenceComponent.create(
        inference_component_name=ic_name, endpoint_name=endpoint, region=region,
        specification=InferenceComponentSpecification(
            base_inference_component_name=base_ic,
            container=InferenceComponentContainerSpecification(artifact_url=adapter_s3_uri)))
    if wait:
        wait_ic(ic_name, region)
    return ic_name


# The instructor's adapters shipped in the repo (bf16): app/state key -> folder.
INSTRUCTOR_ADAPTERS = {"instructor_sft": "v2-sft", "instructor_sft_rlvr": "v2-sft-rlvr"}


def upload_adapter(folder: str, s3_prefix: str, region: str) -> str:
    """Copy a repo adapter (adapters/<folder>, same layout as a training job's checkpoints/hf/) to S3
    for an inference component. Idempotent: files already there with the same size are skipped. Returns the S3 prefix (with a trailing slash)."""
    src = REPO_ROOT / "adapters" / folder
    bucket, key = s3_prefix.removeprefix("s3://").rstrip("/").split("/", 1)
    s3 = boto3.client("s3", region_name=region)
    for f in sorted(p.name for p in src.iterdir() if p.is_file() and p.name != "README.md"):
        try:
            if s3.head_object(Bucket=bucket, Key=f"{key}/{f}")["ContentLength"] == (src / f).stat().st_size:
                continue
        except s3.exceptions.ClientError:
            pass
        s3.upload_file(str(src / f), bucket, f"{key}/{f}")
    return f"s3://{bucket}/{key}/"


def adapter_uri_from_job(training_job_name: str, region: str) -> str:
    from sagemaker.core.resources import TrainingJob
    job = TrainingJob.get(training_job_name=training_job_name, region=region)
    return f"{job.model_artifacts.s3_model_artifacts}/checkpoints/hf/"


def wait_ic(ic_name: str, region: str, timeout: int = 3600):
    sm = boto3.client("sagemaker", region_name=region)
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = sm.describe_inference_component(InferenceComponentName=ic_name)
        s = d["InferenceComponentStatus"]
        if s == "InService":
            return d
        if s == "Failed":
            raise RuntimeError(f"{ic_name} failed: {d.get('FailureReason')}")
        time.sleep(20)
    raise TimeoutError(ic_name)


class Cancelled(Exception):
    """Raised by chat() when its `cancel` event is set; the stream is closed first."""


def chat(endpoint: str, ic_name: str, prompt: str, region: str, max_tokens: int = 4096,
         temperature: float = 0.6, top_p: float = 0.95, on_token=None, cancel=None) -> dict:
    """OpenAI-style chat to an inference component, STREAMED.

    Thinking responses can take > 60 s, the limit for a non-streaming InvokeEndpoint call, so we
    use InvokeEndpointWithResponseStream. `on_token(kind, text)` (kind = "thinking"|"answer") is
    called as tokens arrive (used by the app). Returns {text, thinking, latency_s, usage}.
    `cancel` (threading.Event): when set, the stream is closed and Cancelled is raised. Closing the
    stream stops generation on the endpoint immediately (measured), which matters because the
    endpoint serves streaming requests one at a time: an abandoned stream would block the next."""
    from botocore.config import Config
    rt = boto3.client("sagemaker-runtime", region_name=region,
                      config=Config(read_timeout=600, retries={"max_attempts": 2}))
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.time()
    r = rt.invoke_endpoint_with_response_stream(
        EndpointName=endpoint, InferenceComponentName=ic_name,
        ContentType="application/json", Body=json.dumps(body))
    buf, content, thinking, usage = "", [], [], {}
    try:
        _consume = _stream_events(r["Body"], cancel)
    except Exception:
        r["Body"].close()
        raise
    for ev in _consume:
        buf += ev.get("PayloadPart", {}).get("Bytes", b"").decode("utf-8", "ignore")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage") or usage
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or ch.get("message") or {}
                rc = d.get("reasoning_content") or d.get("reasoning")
                if rc:
                    thinking.append(rc)
                    if on_token:
                        on_token("thinking", rc)
                if d.get("content"):
                    content.append(d["content"])
                    if on_token:
                        on_token("answer", d["content"])
    th, ans = "".join(thinking), "".join(content)
    if not th:                                # cut off before </think>: reasoning arrives as content
        from .checks import split_thinking
        th, ans = split_thinking(ans)
    text = f"<think>\n{th}\n</think>\n\n{ans}" if th else ans
    return {"text": text, "thinking": th, "latency_s": round(time.time() - t0, 2), "usage": usage}


def _stream_events(body, cancel):
    """Yield stream events; close the stream and raise Cancelled as soon as `cancel` is set, and
    close it on any error so the endpoint never keeps generating for nobody."""
    try:
        for ev in body:
            if cancel is not None and cancel.is_set():
                raise Cancelled("cancelled by a newer run")
            yield ev
    except BaseException:
        body.close()
        raise


def delete_components(endpoint: str, region: str, adapters_only: bool = False):
    """Remove inference components (adapters first, then base) but KEEP the endpoint/instance."""
    sm = boto3.client("sagemaker", region_name=region)
    ics = sm.list_inference_components(EndpointNameEquals=endpoint)["InferenceComponents"]
    adapters = [i for i in ics if not i["InferenceComponentName"].endswith("-base-ic")]
    for ic in adapters + ([] if adapters_only else [i for i in ics if i not in adapters]):
        sm.delete_inference_component(InferenceComponentName=ic["InferenceComponentName"])
        while True:
            try:
                sm.describe_inference_component(InferenceComponentName=ic["InferenceComponentName"])
                time.sleep(10)
            except sm.exceptions.ClientError:
                break


def delete_all(endpoint: str, region: str, confirm_release_endpoint: bool = False):
    """Delete ICs AND release the endpoint hardware. Requires confirm_release_endpoint=True."""
    if not confirm_release_endpoint:
        raise RuntimeError("delete_all releases the endpoint instance; pass "
                           "confirm_release_endpoint=True (or use delete_components)")
    sm = boto3.client("sagemaker", region_name=region)
    ics = sm.list_inference_components(EndpointNameEquals=endpoint)["InferenceComponents"]
    adapters = [i for i in ics if not i["InferenceComponentName"].endswith("-base-ic")]
    for ic in adapters + [i for i in ics if i not in adapters]:
        sm.delete_inference_component(InferenceComponentName=ic["InferenceComponentName"])
        while True:
            try:
                sm.describe_inference_component(InferenceComponentName=ic["InferenceComponentName"])
                time.sleep(10)
            except sm.exceptions.ClientError:
                break
    for fn, kw in [(sm.delete_endpoint, {"EndpointName": endpoint}),
                   (sm.delete_endpoint_config, {"EndpointConfigName": endpoint}),
                   (sm.delete_model, {"ModelName": f"{endpoint}-base"})]:
        try:
            fn(**kw)
        except sm.exceptions.ClientError as e:
            print("skip:", e)

"""Fast evaluation: managed generation (LLMAsJudgeEvaluator inference stage only) + own scoring.

    execs = launch(S, {"label": model_package_arn}, cases, tag)    # one pipeline per model
    stop_after_inference(S, execs)                                  # stops before the slow judge
    df = score(S, execs, cases)                                     # checks + tokens + fastjudge

`cases`: list of {request, candidates, reference, track, difficulty} (see benchmarks.py). The managed judge stage is
skipped because it is slow (~65 min per 1,000 calls) and never sees our ground truth ('dummy').
"""
from __future__ import annotations

import ast
import json
import re
import time

import boto3
import pandas as pd

from . import checks, config, difficulty, evalpatch, fastjudge, formats, metrics, tokens
from .prompts import build_prompt

METRIC_NAMES = [m["name"] for m in fastjudge.METRICS]


# ---- S3 outputs of the managed evaluation pipelines --------------------------------------------
_EXEC_RE = re.compile(r"/execution/([a-z0-9]+)$")


def _exec_id(arn: str) -> str:
    return _EXEC_RE.search(arn).group(1)


def _list(s3, bucket: str, prefix: str):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            yield o["Key"]


def _find_outputs(S, eval_arn: str):
    """{'custom'|'base': {'judge': key, 'inference': key}} for one evaluation execution."""
    s3 = boto3.client("s3", region_name=S.region)
    eid = _exec_id(eval_arn)
    out: dict[str, dict] = {}
    for key in _list(s3, S.bucket, f"{S.prefix}/{S.run_name}/"):
        if eid not in key:
            continue
        who = "base" if ("BaseInference" in key or "EvaluateBase" in key or "base-llmaj" in key) \
            else "custom"
        if key.endswith("bedrock_llm_judge_results.json"):
            out.setdefault(who, {})["judge"] = key
        elif key.endswith("inference_output.jsonl"):
            out.setdefault(who, {})["inference"] = key
    return out


def _read(S, key: str) -> bytes:
    return boto3.client("s3", region_name=S.region).get_object(Bucket=S.bucket, Key=key)["Body"].read()


def _response(rec) -> str:
    t = rec["modelResponses"][0]["response"]
    try:
        v = ast.literal_eval(t)
        t = v[0] if isinstance(v, list) and v else t
    except Exception:  # noqa: BLE001
        pass
    return t


def launch(S, models: dict[str, str], cases: list[dict], tag: str, log=print,
           max_new_tokens: int | None = None, base_with: str | None = None) -> dict[str, str]:
    """base_with: label of the model whose pipeline also generates base-model answers (scored as
    model "base"); the base model is evaluated once, in that pipeline."""
    from sagemaker.train.evaluate import LLMAsJudgeEvaluator
    rows = [{"query": build_prompt(c["request"], c["candidates"]),
             "response": json.dumps({"reference_answer": c.get("reference"),
                                     "constraints": json.loads(formats.constraints_json(c["request"], c["candidates"]))})}
            for c in cases]
    local = S.local(f"{tag}_eval.jsonl")
    formats.write_jsonl(rows, local)
    uri = S.s3("datasets", f"{tag}_eval.jsonl")
    b, k = uri[5:].split("/", 1)
    boto3.client("s3", region_name=S.region).upload_file(str(local), b, k)
    if max_new_tokens:
        evalpatch.apply(max_new_tokens, config.EVAL_TEMPERATURE, config.EVAL_TOP_P)
    else:
        evalpatch.apply_default()
    execs = {}
    for name, pkg in models.items():                  # construct sequentially
        ex = LLMAsJudgeEvaluator(model=pkg, evaluator_model=config.JUDGE_MODEL, dataset=uri,
                                 s3_output_path=S.s3("eval-output", tag, name) + "/",
                                 evaluate_base_model=(name == base_with),
                                 region=S.region, mlflow_experiment_name=f"octank-{tag}",
                                 custom_metrics=metrics.eval_metrics_json()).evaluate()
        execs[name] = ex.arn
        log(f"launched {tag}/{name}: {ex.arn}")
        time.sleep(5)
    return execs


def stop_after_inference(S, execs: dict[str, str], log=print, poll=60):
    sm = boto3.client("sagemaker", region_name=S.region)
    done = set()
    while len(done) < len(execs):
        for name, arn in execs.items():
            if name in done:
                continue
            steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=arn)["PipelineExecutionSteps"]
            inf = next((s for s in steps if s["StepName"] == "EvaluateCustomInferenceModel"), None)
            base = next((s for s in steps if s["StepName"] == "EvaluateBaseInferenceModel"), None)
            if base and base["StepStatus"] in ("Failed", "Stopped"):
                raise RuntimeError(f"{name} base inference {base['StepStatus']}: {base.get('FailureReason')}")
            if inf and inf["StepStatus"] == "Succeeded" and (base is None or base["StepStatus"] == "Succeeded"):
                try:
                    sm.stop_pipeline_execution(PipelineExecutionArn=arn)
                except Exception as e:  # noqa: BLE001
                    log(f"stop failed (continuing): {e}"[:160])
                done.add(name)
                log(f"inference done -> stopped {name}")
            elif inf and inf["StepStatus"] in ("Failed", "Stopped"):
                raise RuntimeError(f"{name} inference {inf['StepStatus']}: {inf.get('FailureReason')}")
        if len(done) < len(execs):
            time.sleep(poll)


def score(S, execs: dict[str, str], cases: list[dict], log=print, workers: int = 16,
          skip_invalid: bool = False) -> pd.DataFrame:
    """skip_invalid: answers with no parsable JSON (cut off) get 0 on every judge metric without a
    judge call - the app's gate; saves most of the base model's judge calls."""
    fp = {tuple(x["id"] for x in c["candidates"]): c for c in cases}
    rows = []
    for name, arn in execs.items():
        found = _find_outputs(S, arn)
        for who in ("custom", "base"):
            if who not in found or "inference" not in found[who]:
                continue
            label = name if who == "custom" else "base"
            recs = [json.loads(l) for l in _read(S, found[who]["inference"]).decode().splitlines() if l]
            texts = [_response(r) for r in recs]
            ntok = tokens.count(texts, S.region)
            for r, t, n in zip(recs, texts, ntok):
                c = fp.get(tuple(re.findall(r"\[(OCT-\d+)\]", r["prompt"])))
                if c is None:
                    continue
                ck = checks.check_text(t, c["request"], c["candidates"])
                rows.append({"model": label, "track": c["track"], "case": c["request"]["request_id"],
                             "difficulty": c["difficulty"], "hidden_need": ",".join(c["request"].get("hidden_avoid") or []),
                             "json_valid": ck["json_valid"], "hard_pass": ck["hard_pass"],
                             "within_budget": ck["within_budget"], "no_avoid": ck["no_avoid"],
                             "item_count_ok": ck["item_count_ok"], "cut_off": not ck["json_valid"], "tokens": n,
                             "_prompt": build_prompt(c["request"], c["candidates"]), "_response": t,
                             "_gt": formats.constraints_json(c["request"], c["candidates"])})
    todo = [i for i, r in enumerate(rows) if r["json_valid"] or not skip_invalid]
    log(f"judging {len(todo)} of {len(rows)} answers")
    jt = fastjudge.judge_many([{"prompt": rows[i]["_prompt"], "response": rows[i]["_response"],
                                "ground_truth": rows[i]["_gt"]} for i in todo], S.region, workers=workers)
    zero = {m: 0.0 for m in METRIC_NAMES}
    j = [zero] * len(rows)
    for i, x in zip(todo, jt):
        j[i] = x
    df = pd.DataFrame([{**r, **x} for r, x in zip(rows, j)])
    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["track", "model"]).agg(
        n=("case", "size"), hard_pass=("hard_pass", "mean"), no_avoid=("no_avoid", "mean"),
        cut_off=("cut_off", "mean"), tokens=("tokens", "mean"), **{m: (m, "mean") for m in METRIC_NAMES})

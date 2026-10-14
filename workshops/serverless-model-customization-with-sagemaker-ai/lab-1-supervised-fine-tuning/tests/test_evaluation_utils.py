"""Offline behavior checks for the evaluation lesson; no AWS calls are made."""

import ast
import contextlib
import copy
import io
import json
import pathlib
import re
import sys
import types
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

LAB = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
import contractnli as C
import evaluation_utils as E

LABELS = {
    "nda-1": {"hypothesis": "Keep information secret.", "short_description": "secrecy"},
    "nda-2": {"hypothesis": "Information is public.", "short_description": "public"},
}
DOC = {
    "text": "Keep it secret. Do not disclose.",
    "spans": [[0, 15], [16, 32]],
    "annotation_sets": [{"annotations": {
        "nda-1": {"choice": "Entailment", "spans": [0]},
        "nda-2": {"choice": "Contradiction", "spans": [1]},
    }}],
}
DOCUMENTS = [copy.deepcopy(DOC) for _ in range(3)]
BASE = {"label_correct": .5, "evidence_f1": .4, "contradiction_correct": .5, "json_valid": 1.}
TUNED = {"label_correct": .8, "evidence_f1": .75, "contradiction_correct": 1., "json_valid": 1.}
FRONTIER_PREDICTION = {
    "nda-1": {"label": "NotMentioned", "evidence": []},
    "nda-2": {"label": "Contradiction", "evidence": [1]},
}


def artifact_bytes():
    artifacts = {}
    prompt = repr([{"role": "system", "content": "CHECKLIST: nda-1, nda-2"},
                   {"role": "user", "content": "contract"}])
    for tag, scores in (("base", BASE), ("tuned", TUNED)):
        artifacts[f"{tag}_scored.json"] = json.dumps(scores).encode()
        frame = pd.DataFrame({
            "full_prompt": [prompt] * 3,
            "metrics": [repr({"custom_lambda_metric": scores})] * 3,
        })
        out = io.BytesIO()
        frame.to_parquet(out, index=False)
        artifacts[f"{tag}_details.parquet"] = out.getvalue()
    for name in ("base_judge.jsonl", "custom_judge.jsonl"):
        artifacts[name] = "\n".join(json.dumps({
            "automatedEvaluationResult": {"scores": [
                {"metricName": metric["customMetricDefinition"]["name"], "result": score}
                for metric in E.build_judge_metrics()
            ]}}) for score in (.25, None, .75)).encode()
    rows = [row for doc in DOCUMENTS for row in E.score_contract(doc, FRONTIER_PREDICTION, LABELS)]
    artifacts["frontier_metrics.json"] = json.dumps(E.metrics(rows)).encode()
    artifacts["frontier_rows.json"] = json.dumps(rows, default=E._jsonable).encode()
    return artifacts


class FakeS3:
    def __init__(self):
        self.assets = artifact_bytes()
        self.objects = {"datasets/contractnli-nda-review/test/dataset.jsonl":
                        b'{"query":"a","response":"gold"}\n' * 3}
        self.downloads = []
        for step, tag in (("EvaluateBaseModel", "base"), ("EvaluateCustomModel", "tuned")):
            prefix = f"contractnli-scorer-eval/scorer-run/{step}"
            self.objects[f"{prefix}/details/details.parquet"] = self.assets[f"{tag}_details.parquet"]
            scores = json.loads(self.assets[f"{tag}_scored.json"])
            self.objects[f"{prefix}/results_1.json"] = json.dumps({"results": {"custom": scores}}).encode()
        for job, filename in E.LOCAL_JUDGE.items():
            self.objects[f"contractnli-judge-eval/judge-run/{job}/model_output.jsonl"] = self.assets[filename]

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return types.SimpleNamespace(paginate=lambda **kw: [{
            "Contents": [{"Key": key, "LastModified": datetime(2026, 10, 1, tzinfo=timezone.utc)}
                         for key in self.objects if key.startswith(kw["Prefix"])]
        }])

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def download_file(self, bucket, key, target):
        self.downloads.append((bucket, key))
        pathlib.Path(target).write_bytes(self.assets[key.rsplit("/", 1)[-1]])


class FakeSM:
    def __init__(self):
        self.polls = []

    def list_model_packages(self, **kwargs):
        return {"ModelPackageSummaryList": [{"ModelPackageArn": "arn:model-package/test/1"}]}

    def describe_pipeline_execution(self, **kwargs):
        self.polls.append(kwargs["PipelineExecutionArn"])
        return {"PipelineExecutionStatus": "Succeeded"}

    def get_paginator(self, operation):
        if operation == "list_pipelines":
            return types.SimpleNamespace(paginate=lambda: [{"PipelineSummaries": [
                {"PipelineName": "CustomScorerEvaluation-old"},
                {"PipelineName": "LLMAJEvaluation-old"},
            ]}])
        assert operation == "list_pipeline_executions"
        return types.SimpleNamespace(paginate=lambda **kwargs: [{"PipelineExecutionSummaries": [
            {"PipelineExecutionStatus": "Succeeded", "PipelineExecutionArn": "arn:execution/old",
             "StartTime": datetime(2026, 1, 1)},
            {"PipelineExecutionStatus": "Succeeded", "PipelineExecutionArn": "arn:execution/latest",
             "StartTime": datetime(2026, 10, 1)},
        ]}])


def run_notebook(monkeypatch, directory, live, notebook=LAB / "3-evaluation.ipynb", region="us-west-2"):
    """Execute every cell against controlled AWS clients, in either supported mode."""
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(directory)
    monkeypatch.setattr(C, "ensure_dataset", lambda *args: None)
    monkeypatch.setattr(C, "load", lambda split: (copy.deepcopy(DOCUMENTS), copy.deepcopy(LABELS)))
    monkeypatch.setattr(plt, "show", lambda: None)
    plt.close("all")
    s3, sm = FakeS3(), FakeSM()
    submitted, explanations, converse = [], [], []

    def call_converse(**kwargs):
        converse.append(kwargs)
        return {"output": {"message": {"content": [{"text": json.dumps(FRONTIER_PREDICTION)}]}}}

    def client(name, **kwargs):
        return {"s3": s3, "sagemaker": sm,
                "bedrock-runtime": types.SimpleNamespace(converse=call_converse)}[name]

    session = types.SimpleNamespace(boto_region_name=region, default_bucket=lambda: "workshop")
    def register(name, **attrs):
        module = types.ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    register("boto3", client=client, Session=lambda: types.SimpleNamespace(region_name=region))
    for name in ("sagemaker", "sagemaker.core", "sagemaker.core.helper",
                 "sagemaker.ai_registry", "sagemaker.train"):
        register(name)
    register("sagemaker.core.helper.session_helper", Session=lambda: session,
             get_execution_role=lambda: "arn:aws:iam::123456789012:role/workshop")

    class Asset:
        @staticmethod
        def get(name):
            return types.SimpleNamespace(name=name, arn=f"arn:asset/{name}")
        @staticmethod
        def create(**kwargs):
            return types.SimpleNamespace(name=kwargs["name"], arn="arn:asset/new", refresh=lambda: None)

    register("sagemaker.ai_registry.air_constants", REWARD_FUNCTION="REWARD_FUNCTION")
    register("sagemaker.ai_registry.dataset", DataSet=Asset)
    register("sagemaker.ai_registry.evaluator", Evaluator=Asset)
    register("config", BASE_MODEL_ID="huggingface-reasoning-qwen3-4b",
             DATASET_PREFIX="contractnli-nda-review",
             resolve_mlflow_app=lambda *args, **kwargs: "arn:mlflow-app/workshop")

    class ManagedEvaluation:
        kind = ""
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.hyperparameters = types.SimpleNamespace(temperature=0)
            self.hyperparameters.to_dict = lambda: {
                k: v for k, v in vars(self.hyperparameters).items() if k != "to_dict"}
        def evaluate(self):
            submitted.append((self.kind, self.kwargs))
            return types.SimpleNamespace(
                arn=f"arn:execution/{self.kind}-run",
                show_results=lambda **kwargs: explanations.append(kwargs))
    class Scorer(ManagedEvaluation):
        kind = "scorer"
    class Judge(ManagedEvaluation):
        kind = "judge"
    register("sagemaker.train.evaluate", CustomScorerEvaluator=Scorer, LLMAsJudgeEvaluator=Judge)

    ns = {"__name__": "__main__"}
    notebook_data = json.loads(pathlib.Path(notebook).read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        for index, cell in enumerate(notebook_data["cells"]):
            if cell["cell_type"] != "code":
                continue
            code = re.sub(r"RUN_EVALUATIONS = (?:False|True)", f"RUN_EVALUATIONS = {live}", "".join(cell["source"]))
            code = "\n".join(line for line in code.splitlines() if not line.startswith("%"))
            exec(compile(code, f"{notebook}:cell-{index}", "exec"), ns)
    snapshots = {p.stem: json.loads(p.read_text()) for p in (directory / "eval_result").glob("*.json")}
    figures = len(plt.get_fignums())
    plt.close("all")
    return {"snapshots": snapshots, "submitted": submitted, "explanations": explanations,
            "converse": converse, "downloads": s3.downloads, "figures": figures, "polls": sm.polls}


@pytest.mark.parametrize("live", [False, True])
def test_complete_notebook_modes(monkeypatch, tmp_path, live):
    result = run_notebook(monkeypatch, tmp_path, live)
    assert result["snapshots"]["scorer_base"] == BASE
    assert result["snapshots"]["scorer_tuned"] == TUNED
    assert result["snapshots"]["summary"]["eval_contracts"] == 3
    assert set(result["snapshots"]) >= {"scorer_base", "scorer_tuned", "summary", "judge_base", "judge_tuned"}
    assert result["figures"] == 5
    assert any(row["score"] is None for row in result["snapshots"]["judge_base"])
    if live:
        assert [kind for kind, _ in result["submitted"]] == ["scorer", "judge"]
        assert all(kwargs["evaluate_base_model"] for _, kwargs in result["submitted"])
        assert len(result["converse"]) == 3
        assert result["explanations"] == [{"limit": 5, "offset": 0, "show_explanations": True}]
        assert not result["downloads"]
        assert len(result["snapshots"]["frontier_rows"]) == 6
    else:
        assert not result["submitted"] and not result["converse"]
        assert len(result["downloads"]) == 8
        assert all("pdx" in bucket for bucket, _ in result["downloads"])


def test_parser_and_local_metric_definitions():
    answer = {"nda-1": {"label": "Entailment", "evidence": ["0"]},
              "nda-2": {"label": "Contradiction", "evidence": [1]}}
    assert E.parse_verdicts("<think>ignore</think>```json\n" + json.dumps(answer) + "\n``` trailing") == answer
    assert E.parse_verdicts("{{" + json.dumps(answer)[1:]) == answer
    assert E.parse_verdicts("not JSON") is None
    rows = E.score_contract(DOC, answer, LABELS)
    assert E.metrics(rows) == {"decisions": 2, "accuracy": 100., "contradiction_f1": 100.,
                               "evidence_f1": 100., "parse_failures": 0}
    assert E.metrics(E.score_contract(DOC, None, LABELS), 1)["parse_failures"] == 1
    E.check_scorer(DOC)


def test_bedrock_retries_keep_prompt_and_temperature_fallback(monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(E.time, "sleep", sleeps.append)
    def converse(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if len(calls) == 1:
            raise RuntimeError("temperature not supported")
        if len(calls) == 2:
            raise RuntimeError("ThrottlingException")
        return {"output": {"message": {"content": [{"text": "ok"}]}}}
    client = types.SimpleNamespace(converse=converse)
    assert E.ask_bedrock(DOC, client=client, labels=LABELS) == "ok"
    assert calls[0]["inferenceConfig"]["temperature"] == 0
    assert "temperature" not in calls[-1]["inferenceConfig"]
    assert calls[-1]["system"] == [{"text": C.build_system(LABELS)}]
    assert calls[-1]["messages"][0]["content"] == [{"text": C.build_user(DOC, no_think=False)}]
    assert sleeps == [2.0]


def test_failed_contract_does_not_discard_frontier_pass(monkeypatch):
    def ask(doc, *args, **kwargs):
        if doc is DOCUMENTS[0]:
            raise RuntimeError("unavailable")
        return json.dumps(FRONTIER_PREDICTION)
    monkeypatch.setattr(E, "ask_bedrock", ask)
    metrics, rows = E.evaluate_with_bedrock(DOCUMENTS, "test-model", client=None, labels=LABELS)
    assert len(rows) == 6 and metrics["parse_failures"] == 1


def results():
    return E.EvaluationResults(s3=FakeS3(), sm=FakeSM(), bucket="workshop",
        documents=DOCUMENTS, checklist=LABELS, base_model_id="model", frontier_model="frontier",
        scorer_output="s3://workshop/contractnli-scorer-eval",
        judge_output="s3://workshop/contractnli-judge-eval")


def test_result_selection_restart_and_scorer_failure_warning(capsys):
    r = results()
    assert r.resolve_execution(None, "CustomScorerEvaluation") == "arn:execution/latest"
    assert r.resolve_execution(types.SimpleNamespace(arn="specific"), "ignored") == "specific"
    keys = r.find_s3_keys("contractnli-scorer-eval", "EvaluateBaseModel", suffix=".json",
                         execution_arn="arn:execution/scorer-run")
    assert len(keys) == 1
    payload = json.loads(r.s3.objects[keys[0]])
    payload["results"]["custom"]["byoc_failure_count"] = 2
    r.s3.objects[keys[0]] = json.dumps(payload).encode()
    # Avoid creating snapshots outside pytest's temp directory.
    r.save_json = lambda *args: None
    r.scorer_results("arn:execution/scorer-run")
    assert "2 scorer invocation failure(s)" in capsys.readouterr().out


def test_both_prompt_paths_are_checked():
    r = results()
    key = next(k for k in r.s3.objects if "EvaluateBaseModel/details" in k)
    frame = pd.read_parquet(io.BytesIO(r.s3.objects[key]))
    frame["full_prompt"] = repr([{"role": "user", "content": "missing instructions"}])
    out = io.BytesIO()
    frame.to_parquet(out, index=False)
    r.s3.objects[key] = out.getvalue()
    with pytest.raises(AssertionError, match="EvaluateBaseModel"):
        r.verify_prompts("arn:execution/scorer-run")


def test_judge_prompts_and_subset_preserved():
    definitions = E.build_judge_metrics()
    assert [x["customMetricDefinition"]["name"] for x in definitions] == [
        "EvidenceSupportsVerdict", "CarveOutHandling", "ChecklistCompleteness"]
    assert [len(x["customMetricDefinition"]["ratingScale"]) for x in definitions] == [4, 4, 2]
    assert all("{{prompt}}" in x["customMetricDefinition"]["instructions"] for x in definitions)
    definitions[0]["customMetricDefinition"]["name"] = "mutated"
    assert E.build_judge_metrics()[0]["customMetricDefinition"]["name"] == "EvidenceSupportsVerdict"
    r = results()
    uri = r.write_judge_sample("contractnli-nda-review", 2)
    assert uri.endswith("-judge-2/dataset.jsonl")
    assert len(r.s3.objects[uri.split("workshop/", 1)[1]].splitlines()) == 2

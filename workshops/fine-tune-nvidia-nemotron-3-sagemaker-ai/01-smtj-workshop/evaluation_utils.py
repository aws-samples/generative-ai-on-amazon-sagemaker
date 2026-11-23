"""Supporting implementation for endpoint and Bedrock evaluation.

The notebook retains execution switches, the Bedrock submission payload, the
streaming performance/cost controls, and cleanup. This module preserves endpoint
requests, scoring, baselines, judge inputs/results, plots, and error analysis.
Importing it creates no clients or jobs.
"""
import hashlib
import json
import pathlib
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import contractnli as C
from contractnli_scorer import score_record

BASE_C, FRONTIER_C, TUNED_C = "#b0b0b0", "#ff7f0e", "steelblue"
MAX_NAME = 63


def rname(base, suffix):
    cand = f"{base}{suffix}"
    if len(cand) <= MAX_NAME:
        return cand
    digest = hashlib.sha1(base.encode()).hexdigest()[:6]
    keep = MAX_NAME - len(suffix) - len(digest) - 1
    return f"{base[:keep].rstrip('-')}-{digest}{suffix}"

def summarise(rows):
    names = ("label_correct", "evidence_f1", "contradiction_correct", "json_valid")
    return {n: round(100 * statistics.mean(r[n] for r in rows), 2) for n in names}

def judge_table(scores_by_model):
    """Mean and coverage per metric per model."""
    metrics = sorted({m for s in scores_by_model.values() for m in s})
    width = max(len(m) for m in metrics) + 2
    print(f"{'metric':{width}s}" + "".join(f"{n:>22s}" for n in scores_by_model))
    for metric in metrics:
        row = f"{metric:{width}s}"
        for name in scores_by_model:
            vals = scores_by_model[name].get(metric, [])
            got = [v for v in vals if v is not None]
            mean = sum(got) / len(got) if got else float("nan")
            row += f"{mean:.3f} ({len(got)}/{len(vals)})".rjust(22)
        print(row)

def error_directions(rows, docs, top=6):
    """Which gold -> predicted confusions dominate."""
    pairs = Counter()
    for r, d in zip(rows, docs):
        try:
            pred = json.loads(r["generated"])
        except Exception:                                   # noqa: BLE001
            continue
        g = C.gold_for(d)
        for k, v in g.items():
            if k in pred and isinstance(pred[k], dict):
                got = pred[k].get("label")
                if got and got != v["choice"]:
                    pairs[f"{v['choice']} -> {got}"] += 1
    return pairs.most_common(top)

def gold_json(doc, label_keys):
    """The expert answer, in the shape the model is trained to emit."""
    g = C.gold_for(doc)
    return json.dumps({k: {"label": g[k]["choice"], "evidence": list(g[k]["spans"])}
                       for k in label_keys if k in g})

def check_scorer(doc, label_keys):
    # A reference that does not start with '{' silently scores 0.00 on every content metric.
    _probe = gold_json(doc, label_keys)
    assert _probe.lstrip().startswith("{"), f"gold is not bare JSON: {_probe[:60]!r}"

    _perfect = score_record({"id": "check", "model_response": _probe,
                             "reference_answer": _probe}, 0)
    _flat = {m["name"]: m["value"] for m in _perfect["metrics_list"]}
    assert _flat["label_correct"] == 1.0, f"scoring gold against itself gave {_flat}"
    print("scorer self-check passed: gold scored against itself is 1.0")


def ask_endpoint(doc, *, runtime, endpoint_name, served_model_name, labels, max_tokens=700):
    """One contract to the endpoint, returning the raw generated text."""
    payload = {
        "model": served_model_name,
        "messages": C.build_messages(doc, labels),
        "max_tokens": max_tokens,
        "temperature": 0,
        # the flag that turns reasoning off, forwarded per request exactly as in training
        "chat_template_kwargs": C.CHAT_TEMPLATE_KWARGS,
    }
    r = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                ContentType="application/json",
                                Body=json.dumps(payload))
    return json.loads(r["Body"].read())["choices"][0]["message"]["content"]

def score_all(docs, *, runtime, endpoint_name, served_model_name, labels, workers=8):
    """Score every contract, keeping `workers` requests in flight."""
    def one(i_doc):
        i, d = i_doc
        try:
            return i, ask_endpoint(d, runtime=runtime, endpoint_name=endpoint_name, served_model_name=served_model_name, labels=labels)
        except Exception as e:                              # noqa: BLE001
            print(f"  record {i}: {type(e).__name__}")
            return i, ""

    t0, gens = time.time(), {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        with tqdm(total=len(docs), unit="contract", desc=f"scoring ({workers} workers)") as bar:
            for i, text in ex.map(one, enumerate(docs)):
                gens[i] = text
                bar.update(1)
                bar.set_postfix_str(f"{(time.time() - t0) / len(gens):.1f}s/record")

    rows = []
    for i, d in enumerate(docs):
        scored = score_record({"id": str(i), "model_response": gens[i],
                               "reference_answer": gold_json(d, list(labels))}, i)
        flat = {m["name"]: m["value"] for m in scored["metrics_list"]}
        rows.append({"index": i, "generated": gens[i], **flat})
    print(f"\n{len(rows)} contracts in {(time.time() - t0) / 60:.1f} min")
    return rows

def load_precomputed(name, directory="baselines"):
    with open(pathlib.Path(directory) / f"{name}_metrics.json") as stream:
        return json.load(stream)


def load_rows(filename, directory="baselines"):
    with open(pathlib.Path(directory) / filename) as stream:
        return json.load(stream)


def compare_models(base, frontier, tuned, reference, model_slug, *, live=False):
    rows_out = ((f"{model_slug} (untuned)", base),
                (f"{model_slug} (fine-tuned)", tuned),
                (frontier["model"].split(".")[-1], frontier))
    w = max(len(n) for n, _ in rows_out) + 2
    print(f"{'model':{w}s} {'accuracy':>9s} {'evidence-F1':>12s} {'contra-ok':>10s} {'JSON':>7s}")
    for name, m in rows_out:
        print(f"{name:{w}s} {m['label_correct']:9.1f} {m['evidence_f1']:12.1f} "
              f"{m['contradiction_correct']:10.1f} {m['json_valid']:7.1f}")

    # A reference run of this exact lab, for comparison. Yours will differ by a few tenths.
    ref = reference
    print()
    print(f"{'reference run of this lab':{w}s} {ref['label_correct']:9.1f} "
          f"{ref['evidence_f1']:12.1f} {ref['contradiction_correct']:10.1f} {ref['json_valid']:7.1f}")
    if live:
        print(f"{'your run, difference':{w}s} {tuned['label_correct'] - ref['label_correct']:+9.1f} "
              f"{tuned['evidence_f1'] - ref['evidence_f1']:+12.1f} "
              f"{tuned['contradiction_correct'] - ref['contradiction_correct']:+10.1f} "
              f"{tuned['json_valid'] - ref['json_valid']:+7.1f}")


def plot_comparison(base, frontier, tuned):
    import matplotlib.pyplot as plt
    import numpy as np

    BASE_C, FRONTIER_C, TUNED_C = "#b0b0b0", "#ff7f0e", "steelblue"
    metrics = ["accuracy", "evidence F1", "contradiction correctness"]
    series = [
        ("untuned", BASE_C, [base["label_correct"], base["evidence_f1"],
                             base["contradiction_correct"]]),
        ("fine-tuned", TUNED_C, [tuned["label_correct"], tuned["evidence_f1"],
                                 tuned["contradiction_correct"]]),
        ("Sonnet 5", FRONTIER_C, [frontier["label_correct"], frontier["evidence_f1"],
                                  frontier["contradiction_correct"]]),
    ]

    x = np.arange(len(metrics))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, (label, colour, vals) in enumerate(series):
        bars = ax.bar(x + (k - 1) * width, vals, width, label=label, color=colour)
        ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("%"); ax.set_ylim(0, 100)
    ax.legend(); ax.set_title("ContractNLI, 123 held-out contracts")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.show()


def build_judge_metrics():
    _intro = (
        "Judge ONLY the final JSON object in the response. These models emit a "
        "<think> reasoning block first; ignore it completely, a confident-sounding "
        "deliberation that reaches the wrong label is still wrong. The response may "
        "also arrive wrapped in a stringified list; unwrap it.\n"
    )

    custom_metrics = [
        {"customMetricDefinition": {
            "name": "EvidenceSupportsVerdict",
            "instructions": (
                _intro +
                "The contract is given as numbered spans in the prompt. For each checklist "
                "item the response gives a verdict and cites span numbers. Look up the cited "
                "spans in the prompt and judge whether their text actually justifies the "
                "verdict, ignore whether they match the reference span numbers, since a "
                "different clause may state the same thing.\n"
                "Prompt: {{prompt}}\nResponse: {{prediction}}\nReference: {{ground_truth}}"),
            "ratingScale": [
                {"definition": "Every committed verdict is supported by the clauses it cites",
                 "value": {"floatValue": 3}},
                {"definition": "Most verdicts are supported; a few citations are off",
                 "value": {"floatValue": 2}},
                {"definition": "Several citations do not support their verdict",
                 "value": {"floatValue": 1}},
                {"definition": "Citations are largely unrelated to the verdicts",
                 "value": {"floatValue": 0}},
            ]}},
        {"customMetricDefinition": {
            "name": "CarveOutHandling",
            "instructions": (
                _intro +
                "A checklist item stated absolutely is contradicted by a clause with no "
                "exception for it. Judge whether the response handled exceptions and "
                "carve-outs correctly, rather than defaulting to NotMentioned when the "
                "contract is in fact silent, or to Entailment when a protection is absent.\n"
                "Prompt: {{prompt}}\nResponse: {{prediction}}\nReference: {{ground_truth}}"),
            "ratingScale": [
                {"definition": "Every exception and carve-out is handled correctly",
                 "value": {"floatValue": 3}},
                {"definition": "Most are handled; one or two are missed",
                 "value": {"floatValue": 2}},
                {"definition": "Several are missed, defaulting to NotMentioned or Entailment",
                 "value": {"floatValue": 1}},
                {"definition": "Exceptions are ignored throughout", "value": {"floatValue": 0}},
            ]}},
        {"customMetricDefinition": {
            "name": "ChecklistCompleteness",
            "instructions": (
                _intro +
                "Check that the response is valid JSON containing an entry for every "
                "checklist key listed in the prompt, each with a label and an evidence "
                "list, and no commentary outside the JSON.\n"
                "Prompt: {{prompt}}\nResponse: {{prediction}}"),
            "ratingScale": [
                {"definition": "Complete and well-formed", "value": {"floatValue": 1}},
                {"definition": "Missing entries or malformed", "value": {"floatValue": 0}},
            ]}},
    ]

    return custom_metrics


def build_judge_dataset(rows, name, *, s3, bucket_name, documents, labels, judge_prefix, n=30):
    """Upload `n` answers as precomputed inference responses, returning the S3 URI."""
    records = []
    for r in rows[:n]:
        doc = documents[r["index"]]
        records.append({
            # the judge has to look up cited span numbers, so it gets the user turn verbatim
            "prompt": C.build_messages(doc, labels)[1]["content"],
            "referenceResponse": gold_json(doc, list(labels)),
            "modelResponses": [{"response": r["generated"], "modelIdentifier": name}],
        })
    key = f"{judge_prefix}/input/{name}-{n}.jsonl"
    body = "".join(json.dumps(rec) + "\n" for rec in records)
    s3.put_object(
        Bucket=bucket_name, Key=key, Body=body.encode())
    uri = f"s3://{bucket_name}/{key}"
    print(f"{name}: {len(records)} records -> {uri}")
    return uri

def wait_for_judge(arns, *, bedrock, poll=60):
    """Block until every job leaves InProgress, reporting elapsed minutes as it goes."""
    pending, started = dict(arns), time.time()
    while pending:
        for name, arn in list(pending.items()):
            job = bedrock.get_evaluation_job(jobIdentifier=arn)
            if job["status"] not in ("InProgress", "Scheduled", "Stopping"):
                print(f"[{(time.time() - started) / 60:5.1f} min] {name}: {job['status']}")
                if job["status"] == "Failed":
                    print(f"  {job.get('failureMessages')}")
                pending.pop(name)
        if pending:
            print(f"[{(time.time() - started) / 60:5.1f} min] waiting on {', '.join(pending)}")
            time.sleep(poll)

def read_judge_scores(arn, *, bedrock, s3, bucket_name):
    """Per-metric scores from a finished job, keeping None where the judge declined.

    None is not zero. The judge returns nothing when it cannot read an answer, and counting that
    as zero charges a model twice: once for the metric it failed, once for the zero. Those are
    kept out of the mean and surfaced as coverage instead.
    """
    out = bedrock.get_evaluation_job(jobIdentifier=arn)["outputDataConfig"]["s3Uri"]
    prefix = out.replace(f"s3://{bucket_name}/", "")
    job_id = arn.split("/")[-1]
    scores = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith(".jsonl") or job_id not in obj["Key"]:
                continue
            body = s3.get_object(Bucket=bucket_name, Key=obj["Key"])["Body"].read().decode()
            for line in body.splitlines():
                if not line.strip():
                    continue
                result = json.loads(line).get("automatedEvaluationResult", {})
                for score in result.get("scores", []):
                    scores.setdefault(score["metricName"], []).append(score.get("result"))
    return scores

def plot_judge_scores(judge_scores, judge_n):
    import numpy as np

    metrics = sorted({m for s in judge_scores.values() for m in s})
    JUDGE_COLOURS = {"base": BASE_C, "tuned": TUNED_C}

    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for k, (name, scores) in enumerate(judge_scores.items()):
        means, coverage = [], []
        for metric in metrics:
            vals = scores.get(metric, [])
            got = [v for v in vals if v is not None]
            means.append(sum(got) / len(got) if got else np.nan)
            coverage.append(f"{len(got)}/{len(vals)}")
        bars = ax.bar(x + (k - 0.5) * width, means, width, label=name,
                      color=JUDGE_COLOURS.get(name, "grey"))
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in means], fontsize=9, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylabel("mean score (0-1)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(f"LLM-as-a-Judge over {judge_n} contracts")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


def show_error_directions(rows, documents):
    print("the fine-tuned model's most common confusions:")
    for pair, n in error_directions(rows, documents):
        print(f"  {pair:34s} {n}")


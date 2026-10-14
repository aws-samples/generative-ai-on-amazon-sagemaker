"""Supporting implementation for the ContractNLI evaluation notebook.

The notebook keeps the SageMaker evaluator configuration and submission visible.
This module contains the original parsing, local scoring, Bedrock retry/concurrency,
result loading, custom judge prompts, diagnostics, charts, and snapshot exports.
Importing it creates no AWS clients and starts no jobs.
"""

import ast
import hashlib
import io
import json
import pathlib
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import contractnli as C

BASE_COLOUR, FRONTIER_COLOUR, TUNED_COLOUR = "#b0b0b0", "#ff7f0e", "steelblue"
JUDGE_COLOURS = {"base": BASE_COLOUR, "fine-tuned": TUNED_COLOUR}
SHOW = ["label_correct", "evidence_f1", "contradiction_correct", "json_valid"]
LOCAL_DETAILS = {"EvaluateBaseModel": "base_details.parquet",
                 "EvaluateCustomModel": "tuned_details.parquet"}
LOCAL_RESULTS = {"EvaluateBaseModel": "base_scored.json",
                 "EvaluateCustomModel": "tuned_scored.json"}
LOCAL_JUDGE = {"base-llmaj-eval": "base_judge.jsonl",
               "custom-llmaj-eval": "custom_judge.jsonl"}
ASSET_BUCKETS = {"us-east-1": "ws-assets-prod-iad-r-iad-ed304a55c2ca1aee",
                "us-west-2": "ws-assets-prod-iad-r-pdx-f3b3f9f1a7d6a3d0"}
ASSET_PREFIX = "548b5be9-2da8-4c93-82f7-b0b474108ab3/lab1/eval_results"
DEFAULT_FRONTIER = "global.anthropic.claude-sonnet-4-6"


def model_package_group_name(base_model_id):
    """Use the same stable, length-limited name as training and deployment."""
    suffix = "-contractnli-sft-mpg"
    candidate = f"{base_model_id}{suffix}"
    if len(candidate) <= 63:
        return candidate
    digest = hashlib.sha1(base_model_id.encode()).hexdigest()[:6]
    keep = 63 - len(suffix) - len(digest) - 1
    truncated = base_model_id[:keep].rstrip("-")
    return f"{truncated}-{digest}{suffix}"


def parse_verdicts(text):
    """Pull the JSON verdict object out of a model reply. None if unparseable.

    Models wrap JSON in prose, markdown fences or <think> blocks, so we strip those
    and then walk the braces to find where the first object closes, a plain
    `json.loads` on the whole reply fails on any trailing commentary.
    """
    body = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.DOTALL)
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.DOTALL)
    if fence:
        body = fence.group(1)
    start = body.find("{")
    if start == -1:
        return None
    # Models sometimes echo the doubled braces from the prompt template.
    if body[start:start + 2] == "{{":
        start += 1
    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(body[start:], start):
        if in_string:
            escaped = (ch == "\\") and not escaped
            if ch == '"' and not escaped:
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(body[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None

def f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 100 * 2 * precision * recall / (precision + recall) if precision + recall else 0.0

def metrics(rows, parse_failures=0):
    n = len(rows) or 1

    accuracy = 100 * sum(r["pred"] == r["gold"] for r in rows) / n

    # Contradiction as a one-vs-rest binary problem over the rows.
    tp = sum(r["gold"] == r["pred"] == "Contradiction" for r in rows)
    fp = sum(r["pred"] == "Contradiction" != r["gold"] for r in rows)
    fn = sum(r["gold"] == "Contradiction" != r["pred"] for r in rows)

    # Evidence: micro-averaged over spans, only where the gold cites something.
    cited = [r for r in rows if r["gold_ev"]]
    etp = sum(len(r["gold_ev"] & r["pred_ev"]) for r in cited)
    efp = sum(len(r["pred_ev"] - r["gold_ev"]) for r in cited)
    efn = sum(len(r["gold_ev"] - r["pred_ev"]) for r in cited)

    return {"decisions": len(rows),
            "accuracy": round(accuracy, 1),
            "contradiction_f1": round(f1(tp, fp, fn), 1),
            "evidence_f1": round(f1(etp, efp, efn), 1),
            "parse_failures": parse_failures}

def _jsonable(value):
    """Convert what json.dump cannot: score rows carry sets and numpy scalars."""
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")

def grouped_bars(ax, labels, series, colours, ylabel):
    """Grouped bar chart: one group per label, one bar per named series."""
    positions = np.arange(len(labels))
    width = 0.8 / len(series)
    for i, ((name, values), colour) in enumerate(zip(series.items(), colours)):
        offset = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(positions + offset, values, width, label=name, color=colour)
        for bar, value in zip(bars, values):
            if value == value:                       # skip NaN
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                        f"{value:.0%}", ha="center", va="bottom", size=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

def plot_scorer_spread(base_details, tuned_details):
    """Per-contract distributions behind the base/fine-tuned aggregates."""
    fig, axes = plt.subplots(1, len(SHOW), figsize=(13, 4), sharey=True)
    for ax, name in zip(axes, SHOW):
        data = [base_details[name].to_numpy(), tuned_details[name].to_numpy()]
        box = ax.boxplot(data, patch_artist=True, widths=0.6,
                         medianprops=dict(color="black"))
        # Labels are set here rather than through boxplot's own keyword, which was
        # renamed in matplotlib 3.9 and would tie this cell to one version.
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["base", "fine-tuned"])
        for patch, colour in zip(box["boxes"], [BASE_COLOUR, TUNED_COLOUR]):
            patch.set_facecolor(colour)
        for i, values in enumerate(data, start=1):
            # Jittered points: several of these metrics pile up on 0 and 1, and a box
            # plot alone cannot show how many contracts sit on each.
            jitter = (np.arange(len(values)) % 7 - 3) / 20
            ax.plot(i + jitter, values, ".", color="black", alpha=0.15, markersize=3)
        ax.set_title(name, size=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("per-contract score")
    fig.suptitle("Every contract, not just the mean", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_judge_bars(judged):
    """Judge metrics per model, annotated with the judge's own coverage.

    `judged` maps a model label to its score frame, so this draws one model or two.
    Both are judged here, `evaluate_base_model=True` on the launch cell scores the base
    and fine-tuned model in the same job. The coverage annotation is the reason to read
    this chart rather than the means alone: a record the harness could not score is
    excluded from the mean, and which records those are differs between the two models,
    so a bar pair with unequal coverage is not a before/after on the same contracts.
    """
    means = {tag: f.groupby("metric")["score"].mean() for tag, f in judged.items()}
    metrics_shown = sorted(set().union(*(m.index for m in means.values())))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    grouped_bars(ax, metrics_shown,
                 {tag: [m.get(name, np.nan) for name in metrics_shown]
                  for tag, m in means.items()},
                 [JUDGE_COLOURS[tag] for tag in means], "mean score (0-1)")

    # `count` skips NaN, `size` does not, so the pair is the judge's coverage.
    labels = []
    for name in metrics_shown:
        parts = [name]
        for tag, scores in judged.items():
            group = scores[scores["metric"] == name]["score"]
            parts.append(f"{tag} scored {group.count()}/{group.size}")
        labels.append("\n".join(parts))
    ax.set_xticklabels(labels, size=8)
    ax.set_title(f"LLM-as-a-Judge, {' vs '.join(judged)}")
    plt.tight_layout()
    plt.show()

def plot_judge_radar(judged):
    """The same means as a shape: a regression on any one axis shows at a glance."""
    means = {tag: f.groupby("metric")["score"].mean() for tag, f in judged.items()}
    metrics_shown = sorted(set.intersection(*(set(m.index) for m in means.values())))
    angles = np.linspace(0, 2 * np.pi, len(metrics_shown), endpoint=False).tolist()
    angles += angles[:1]                     # close the polygon

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    for label, model_mean in means.items():
        colour = JUDGE_COLOURS[label]
        values = [model_mean[m] for m in metrics_shown]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=colour, label=label)
        ax.fill(angles, values, alpha=0.2, color=colour)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_shown, size=8)
    ax.set_ylim(0, 1)
    ax.set_title("Judge metrics overview")
    ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05), fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_error_directions(rows, title, top=6):
    """Which way the model got it wrong, not just how often.

    Bars are coloured by consequence: over-claiming a protection that is not there
    (anything -> Entailment) is the error that misleads a reviewer, so it is
    highlighted rather than left to be picked out of a legend.
    """
    counts = Counter(f"{r['gold']} -> {r['pred']}"
                     for r in rows if r["pred"] != r["gold"])
    if not counts:
        print("no errors to plot")
        return
    pairs = counts.most_common(top)[::-1]           # largest at the top of an h-bar
    labels = [p for p, _ in pairs]
    values = [n for _, n in pairs]
    total = sum(counts.values())
    colours = ["#d62728" if p.endswith("-> Entailment") else "steelblue" for p in labels]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(pairs) + 1.6))
    bars = ax.barh(labels, values, color=colours)
    for bar, value in zip(bars, values):
        ax.text(value + total * 0.008, bar.get_y() + bar.get_height() / 2,
                f"{value}  ({value / total:.0%})", va="center", size=9)
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel(f"decisions wrong (of {len(rows)})")
    ax.set_title(f"{title}: error directions\nred = claims a protection that is not there")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()

def score_contract(doc, prediction, checklist):
    """One row per checklist item: gold label/spans vs predicted label/spans."""
    gold = C.gold_for(doc)
    rows = []
    for key in checklist:
        if key not in gold:
            continue
        answer = (prediction or {}).get(key) or {}
        label = answer.get("label") if isinstance(answer, dict) else None
        cited = answer.get("evidence") if isinstance(answer, dict) else None
        cited = ({int(x) for x in cited if str(x).strip().lstrip("-").isdigit()}
                 if isinstance(cited, list) else set())
        rows.append({"key": key,
                     "gold": gold[key]["choice"], "pred": label,
                     "gold_ev": set(gold[key]["spans"]), "pred_ev": cited})
    return rows

def ask_bedrock(doc, model_id=None, *, client, labels, max_tokens=4000, attempts=5):
    """One Converse call for one contract, retrying on throttling.

    Both turns come from `contractnli.py`, so the frontier model is asked exactly what
    the fine-tuned model was trained on. Converse carries the system turn in its own
    `system` parameter, the API's equivalent of the `system` message in the training
    records, rather than pasted on the front of the user turn.
    """
    model_id = model_id or DEFAULT_FRONTIER
    config = {"maxTokens": max_tokens, "temperature": 0.0}
    delay = 2.0
    for _ in range(attempts):
        try:
            response = client.converse(
                modelId=model_id,
                system=[{"text": C.build_system(labels)}],
                messages=[{"role": "user",
                    "content": [{"text": C.build_user(doc, no_think=False)}]}],
                inferenceConfig=config)
            return "".join(part.get("text", "")
                           for part in response["output"]["message"]["content"])
        except Exception as error:
            message = str(error)
            if "temperature" in message.lower():   # some models reject the field
                config.pop("temperature", None)
                continue
            if not any(k in message for k in ("Throttl", "429", "503", "Timeout")):
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise RuntimeError(f"{model_id}: giving up after {attempts} attempts")

def evaluate_with_bedrock(docs, model_id, *, client, labels, workers=6):
    """Review every contract, then score. Returns (metrics, rows).

    `workers` contracts are reviewed concurrently. Each one is a single Bedrock call
    carrying a full NDA (~3,200 tokens in on average, ~1,000 out), so the whole pass takes
    several minutes on the full test split. The progress bar is what tells you it is
    working rather than hanging, and its `json_ok` counter shows whether the replies
    are usable as they arrive rather than only at the end.
    """
    predictions = [None] * len(docs)
    unanswered = 0

    # Return the index with the result: completions arrive out of order, and rows must
    # line up with `docs` for scoring.
    def review(i):
        return i, parse_verdicts(ask_bedrock(docs[i], model_id, client=client, labels=labels))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(review, i) for i in range(len(docs))]
        with tqdm(total=len(futures), desc=model_id.split(".")[-1],
                  unit="contract") as bar:
            parsed = 0
            for future in as_completed(futures):
                try:
                    i, prediction = future.result()
                    predictions[i] = prediction
                    parsed += prediction is not None
                except Exception as error:
                    # One contract exhausting its retries must not discard the pass.
                    unanswered += 1
                    bar.write(f"  no reply: {type(error).__name__}: {error}")
                bar.update(1)
                bar.set_postfix_str(f"json_ok {parsed}/{bar.n}")

    rows = []
    for doc, prediction in zip(docs, predictions):
        rows += score_contract(doc, prediction, labels)
    # A missing prediction is either an unparseable reply or a call that never
    # returned. Both are unusable, so both count as a failure.
    failures = sum(p is None for p in predictions)
    if unanswered:
        print(f"{unanswered} contract(s) never returned a reply after retries")
    return metrics(rows, failures), rows

def plot_statistical_comparison(base, frontier, tuned, frontier_name, *, num_contracts, checklist_items):
    """Three models on the two metrics all three report.

    `base` and `tuned` come from the managed scorer on a 0-1 scale; `frontier` comes
    from the local `metrics()` pass on a 0-100 scale and is rescaled here.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped_bars(
        ax,
        ["accuracy", "evidence-F1"],
        {"base": [base["label_correct"], base["evidence_f1"]],
         frontier_name: [frontier["accuracy"] / 100, frontier["evidence_f1"] / 100],
         "fine-tuned": [tuned["label_correct"], tuned["evidence_f1"]]},
        [BASE_COLOUR, FRONTIER_COLOUR, TUNED_COLOUR],
        "score")
    ax.set_title(f"Statistical scoring, {num_contracts} held-out contracts")

    # This is the figure that gets screenshotted, so the caveat belongs on it rather
    # than in prose above it. A bar is a capability score only to the extent the model
    # returned parseable JSON at all; a bare grey bar reads as a verdict on contract
    # reading even when much of it is format failure.
    if base.get("json_valid", 1) < 0.95:
        valid = base["json_valid"]
        ax.set_xlabel(
            f"base returned parseable JSON on {valid:.0%} of contracts "
            f"({valid * num_contracts:.0f}/{num_contracts}), and an unparseable answer "
            f"scores 0 on all {checklist_items} items.\nOn the ones it did answer, base is "
            f"~{base['label_correct'] / valid:.0%} / {base['evidence_f1'] / valid:.0%}: "
            "the grey bars are part output format, not only contract reading.",
            fontsize=8, color="#555555", labelpad=10)

    plt.tight_layout()
    plt.show()

def check_scorer(doc):
    """Keep the gold-versus-lazy-answer guard from the notebook."""
    from contractnli_scorer import score_record

    gold = {k: {"label": v["choice"], "evidence": list(v["spans"])}
            for k, v in C.gold_for(doc).items()}
    reference = json.dumps(gold)

    # The guard: a deliberately poor answer must score *below the gold answer* on the same
    # contract. Comparing the two is what makes this robust: an absolute threshold is not.
    # `aggregate_reward_score` is 0.6*label_correct + 0.4*evidence_f1, and an all-NotMentioned
    # answer earns nothing on evidence, so its ceiling is 0.6 x the share of items that really
    # are NotMentioned. On the contracts where that share is high it clears 0.5 legitimately,
    # true for 5 of the 123 test contracts: so a `< 0.5` assertion would fail on a scorer
    # that is working perfectly. Gold scoring 1.0 while the lazy answer scores less cannot be
    # satisfied by a scorer reading the wrong field, whichever contract you land on.
    lazy = json.dumps({k: {"label": "NotMentioned", "evidence": []} for k in gold})
    lazy_score = score_record({"id": "lazy", "model_response": lazy,
                               "reference_answer": reference})["aggregate_reward_score"]
    gold_score = score_record({"id": "gold", "model_response": reference,
                               "reference_answer": reference})["aggregate_reward_score"]

    not_mentioned = sum(v["label"] == "NotMentioned" for v in gold.values())
    print(f"gold answer            -> {gold_score:.3f}")
    print(f"all-NotMentioned answer -> {lazy_score:.3f}   "
          f"({not_mentioned}/{len(gold)} items really are NotMentioned)")
    assert gold_score == 1.0, "scorer does not score a perfect answer as perfect"
    assert lazy_score < gold_score, "scorer is not reading model_response"
    print("scorer reads the model's answer \u2713")


def show_scorer_source():
    """Inspect the standalone file uploaded to SageMaker."""
    import inspect

    from contractnli_scorer import score_record

    print(inspect.getsource(score_record))


def build_judge_metrics():
    """Return all three original judge prompts and their rating scales."""
    return [
        {
            "customMetricDefinition": {
                "name": "EvidenceSupportsVerdict",
                "instructions": (
                    "Judge ONLY the final JSON object in the response. These models emit a "
                    "<think> reasoning block first; ignore it completely, a confident-sounding "
                    "deliberation that reaches the wrong label is still wrong. The response may "
                    "also arrive wrapped in a stringified list; unwrap it.\n"
                    "The contract is given as numbered spans in the prompt. For each checklist "
                    "item the response gives a verdict and cites span numbers. Look up the cited "
                    "spans in the prompt and judge whether their text actually justifies the "
                    "verdict, ignore whether they match the reference span numbers, since a "
                    "different clause may state the same thing.\n"
                    "Prompt: {{prompt}}\nResponse: {{prediction}}\nReference: {{ground_truth}}"
                ),
                "ratingScale": [
                    {"definition": "Every committed verdict is supported by the clauses it cites",
                     "value": {"floatValue": 3}},
                    {"definition": "Most verdicts are supported; a few citations are off",
                     "value": {"floatValue": 2}},
                    {"definition": "Several citations do not support their verdict",
                     "value": {"floatValue": 1}},
                    {"definition": "Citations are largely unrelated to the verdicts",
                     "value": {"floatValue": 0}},
                ],
            }
        },
        {
            "customMetricDefinition": {
                "name": "CarveOutHandling",
                "instructions": (
                    "Judge ONLY the final JSON object in the response. These models emit a "
                    "<think> reasoning block first; ignore it completely, a confident-sounding "
                    "deliberation that reaches the wrong label is still wrong. The response may "
                    "also arrive wrapped in a stringified list; unwrap it.\n"
                    "A checklist item stated absolutely is contradicted by a clause with no "
                    "exception for it. Judge whether the response handled exceptions and "
                    "carve-outs correctly, rather than defaulting to NotMentioned when the "
                    "contract is in fact silent, or to Entailment when a protection is absent.\n"
                    "Prompt: {{prompt}}\nResponse: {{prediction}}\nReference: {{ground_truth}}"
                ),
                "ratingScale": [
                    {"definition": "Every exception and carve-out is handled correctly",
                     "value": {"floatValue": 3}},
                    {"definition": "Most are handled; one or two are missed",
                     "value": {"floatValue": 2}},
                    {"definition": "Several are missed, defaulting to NotMentioned or Entailment",
                     "value": {"floatValue": 1}},
                    {"definition": "Exceptions are ignored throughout",
                     "value": {"floatValue": 0}},
                ],
            }
        },
        {
            "customMetricDefinition": {
                "name": "ChecklistCompleteness",
                "instructions": (
                    "Judge ONLY the final JSON object in the response. These models emit a "
                    "<think> reasoning block first; ignore it completely, a confident-sounding "
                    "deliberation that reaches the wrong label is still wrong. The response may "
                    "also arrive wrapped in a stringified list; unwrap it.\n"
                    "Check that the response is valid JSON containing an entry for every "
                    "checklist key listed in the prompt, each with a label and an evidence "
                    "list, and no commentary outside the JSON.\n"
                    "Prompt: {{prompt}}\nResponse: {{prediction}}"
                ),
                "ratingScale": [
                    {"definition": "Complete and well-formed", "value": {"floatValue": 1}},
                    {"definition": "Missing entries or malformed", "value": {"floatValue": 0}},
                ],
            }
        },
    ]


def wait_for_pipeline(sm, execution_arn, poll_seconds=60):
    """Display the same pipeline polling progress as the original notebook."""
    while True:
        status = sm.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn)["PipelineExecutionStatus"]
        print(f"{time.strftime('%H:%M:%S')}  {status}")
        if status != "Executing":
            return status
        time.sleep(poll_seconds)


def show_error_directions(rows, model_id, top=6):
    """Print the error counts and retain the consequence-colored chart."""
    errors = Counter(f"{r['gold']} -> {r['pred']}"
                     for r in rows if r["pred"] != r["gold"])
    total = sum(errors.values())
    print(f"{model_id}: {total} errors in {len(rows)} decisions")
    for change, n in errors.most_common(5):
        print(f"   {change:34s} x{n:3d}  ({100 * n / total:.0f}%)")
    plot_error_directions(rows, model_id.split(".")[-1], top=top)


class EvaluationResults:
    """Read one evaluation's artifacts, check them, plot them, and save snapshots."""

    def __init__(self, *, s3, sm, bucket, documents, checklist, base_model_id,
                 frontier_model, scorer_output, judge_output, output_dir="./eval_result"):
        self.s3 = s3
        self.sm = sm
        self.bucket = bucket
        self.documents = documents
        self.checklist = list(checklist)
        self.base_model_id = base_model_id
        self.frontier_model = frontier_model
        self.scorer_prefix = scorer_output.split(f"{bucket}/", 1)[1]
        self.judge_output = judge_output
        self.output_dir = pathlib.Path(output_dir)
        self.precomputed_dir = None

    def resolve_execution(self, execution, kind):
        """Keep support for reading results after restarting the kernel."""
        return (execution.arn if execution is not None else None
                if self.precomputed_dir else self.last_successful_execution(kind))


    def precomputed(self, name):
        """Path to a pre-computed results file, or None when reading from S3."""
        if not (self.precomputed_dir and name):
            return None
        path = pathlib.Path(self.precomputed_dir) / name
        return path if path.exists() else None

    def find_s3_keys(self, prefix, *must_contain, suffix="", execution_arn=None):
        """Matching keys under `prefix`, oldest first, narrowed to one run where possible.

        Every evaluation you launch writes into the same output prefix, so after a second
        run the prefix holds two sets of results. Passing the execution ARN pins the
        lookup to that run; without it the newest is used, which is only the run you want
        if nothing else has written since.
        """
        objects = [o for page in self.s3.get_paginator("list_objects_v2").paginate(
                       Bucket=self.bucket, Prefix=prefix)
                   for o in page.get("Contents", [])
                   if all(m in o["Key"] for m in must_contain) and o["Key"].endswith(suffix)]
        if execution_arn:
            execution_id = execution_arn.rsplit("/", 1)[-1]
            objects = [o for o in objects if execution_id in o["Key"]] or objects
        return [o["Key"] for o in sorted(objects, key=lambda o: o["LastModified"])]

    def last_successful_execution(self, name_contains, status="Succeeded"):
        """The ARN of the most recent successful execution of an evaluation pipeline.

        `evaluate()` hands back an object with the ARN on it, but that object lives in the
        kernel: restart, or come back tomorrow to read yesterday's numbers, and the read
        cells below have nothing to point at. The SDK names its pipelines
        `SagemakerEvaluation-<kind>-<uuid>`, so the kind is enough to find the run again -
        "CustomScorerEvaluation" or "LLMAJEvaluation".
        """
        runs = []
        for page in self.sm.get_paginator("list_pipelines").paginate():
            for pipeline in page["PipelineSummaries"]:
                if name_contains not in pipeline["PipelineName"]:
                    continue
                for run_page in self.sm.get_paginator("list_pipeline_executions").paginate(
                        PipelineName=pipeline["PipelineName"]):
                    runs += [r for r in run_page["PipelineExecutionSummaries"]
                             if r["PipelineExecutionStatus"] == status]
        if not runs:
            raise ValueError(f"no {status} pipeline execution whose name contains "
                             f"{name_contains!r} - launch one with the cell above first")
        latest = max(runs, key=lambda r: r["StartTime"])
        print(f"reading {latest['PipelineExecutionArn'].rsplit('/', 1)[-1]} "
              f"({latest['StartTime']:%Y-%m-%d %H:%M}, newest of {len(runs)} successful)")
        return latest["PipelineExecutionArn"]

    def save_json(self, name, payload):
        """Write one JSON file into ./eval_result and return its path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=_jsonable)
        print(f"  saved {path} ({path.stat().st_size:,} bytes)")
        return path

    def read_details(self, step_marker, execution_arn=None):
        """The per-record Parquet file the given pipeline step wrote, as a DataFrame."""
        local = self.precomputed(LOCAL_DETAILS.get(step_marker))
        if local:
            return pd.read_parquet(local)
        keys = self.find_s3_keys(self.scorer_prefix, step_marker, "/details/", suffix=".parquet",
                            execution_arn=execution_arn)
        if not keys:
            return None
        return pd.read_parquet(io.BytesIO(
            self.s3.get_object(Bucket=self.bucket, Key=keys[-1])["Body"].read()))

    def read_results(self, prefix, step_marker, execution_arn=None):
        """Pull the aggregate results file the given pipeline step wrote."""
        local = self.precomputed(LOCAL_RESULTS.get(step_marker))
        if local:
            return json.loads(local.read_text())
        keys = self.find_s3_keys(prefix, step_marker, "/results_", suffix=".json",
                            execution_arn=execution_arn)
        if not keys:
            return None
        payload = json.loads(self.s3.get_object(Bucket=self.bucket, Key=keys[-1])["Body"].read())
        return list(payload["results"].values())[0]

    def load_scorer_details(self, step_marker, execution_arn=None):
        """One row per contract of the metrics the managed scorer returned.

        Reads the same Parquet file the prompt check above read, but keeps the `metrics`
        column: the scorer's own metrics arrive nested under `custom_lambda_metric`, and
        the rest of the columns are the harness's built-in text metrics (ROUGE, BLEU),
        which do not mean much on JSON output and are ignored here.
        """
        frame = self.read_details(step_marker, execution_arn)
        if frame is None:
            return None
        rows = []
        for raw in frame["metrics"]:
            entry = ast.literal_eval(raw) if isinstance(raw, str) else dict(raw)
            nested = entry.pop("custom_lambda_metric", None)
            if isinstance(nested, str):
                nested = ast.literal_eval(nested)
            if isinstance(nested, dict):
                entry.update(nested)
            rows.append(entry)
        return pd.DataFrame(rows)

    def load_judge_scores(self, job_marker, execution_arn=None):
        """One row per (contract, metric) of the judge's scores, on a 0-1 scale.

        `result` is None where the harness could not parse a score out of the judge's
        reply; it is kept as NaN so pandas excludes it from the mean instead of reading it
        as a zero. Those records still carry the judge's written reasoning.
        """
        local = self.precomputed(LOCAL_JUDGE.get(job_marker))
        if local:
            body = local.read_text()
        else:
            keys = self.find_s3_keys(self.judge_output.split(f"{self.bucket}/")[1], job_marker,
                                suffix="_output.jsonl", execution_arn=execution_arn)
            if not keys:
                return None
            body = self.s3.get_object(Bucket=self.bucket, Key=keys[-1])["Body"].read().decode()
        return pd.DataFrame(
            [{"metric": score["metricName"], "score": score["result"]}
             for line in body.splitlines() if line.strip()
             for score in json.loads(line)["automatedEvaluationResult"]["scores"]])

    def load_precomputed(self, region, directory="./eval_results"):
        """Download all eight original artifacts and restore frontier evidence sets."""
        self.precomputed_dir = pathlib.Path(directory)
        self.precomputed_dir.mkdir(parents=True, exist_ok=True)
        bucket = ASSET_BUCKETS[region]
        for name in (*LOCAL_RESULTS.values(), *LOCAL_DETAILS.values(),
                     *LOCAL_JUDGE.values(), "frontier_metrics.json", "frontier_rows.json"):
            self.s3.download_file(bucket, f"{ASSET_PREFIX}/{name}",
                                  str(self.precomputed_dir / name))
            print(f"  {name}")
        frontier_metrics = json.loads((self.precomputed_dir / "frontier_metrics.json").read_text())
        frontier_rows = json.loads((self.precomputed_dir / "frontier_rows.json").read_text())
        for row in frontier_rows:
            row["gold_ev"] = set(row["gold_ev"])
            row["pred_ev"] = set(row["pred_ev"])
        print(f"\nDownloaded pre-computed results to {self.precomputed_dir}/")
        print(f"frontier baseline loaded: {len(frontier_rows)} decisions, "
              f"accuracy {frontier_metrics['accuracy']}, "
              f"evidence F1 {frontier_metrics['evidence_f1']}")
        return frontier_metrics, frontier_rows


    def verify_prompts(self, execution_arn=None):
        """Verify that both inference paths actually received the checklist."""
        prompts = {step: self.read_details(step, execution_arn)["full_prompt"].iloc[0]
                   for step in ("EvaluateBaseModel", "EvaluateCustomModel")}

        sent = prompts["EvaluateCustomModel"]
        turns = ast.literal_eval(sent) if isinstance(sent, str) else sent

        print(f"{len(turns)} turn(s) sent: {[t['role'] for t in turns]}\n")
        for turn in turns:
            print(f"--- {turn['role']} ({len(turn['content'])} chars) ---")
            print(turn["content"][:400].rstrip(), "...\n")

        # The checklist *is* the instruction. If it never arrived, every score below measures
        # a model answering a question it was not asked: which looks like a bad model. Both
        # steps are checked, not just the one printed above: they are separate inference paths
        # (the base step writes its details under `additonals3data`, the custom step under
        # `hf_merged`), so an instruction that arrived for one is no evidence for the other.
        for step, raw_prompt in prompts.items():
            step_turns = ast.literal_eval(raw_prompt) if isinstance(raw_prompt, str) else raw_prompt
            assert "CHECKLIST:" in " ".join(t["content"] for t in step_turns), (
                f"{step}: the evaluation container did not pass the dataset's `system` field "
                "through: the model never saw the checklist. Fold the instruction into `query` "
                "in notebook 1 and re-register the test dataset.")
        print("the standing instruction reached both models ✓")


    def scorer_results(self, execution_arn=None):
        """Read, display, and export both models' aggregate metrics."""
        base_scored = self.read_results(self.scorer_prefix, "EvaluateBaseModel", execution_arn)
        tuned_scored = self.read_results(self.scorer_prefix, "EvaluateCustomModel", execution_arn)

        SHOW = ["label_correct", "evidence_f1", "contradiction_correct", "json_valid"]

        # The scorer runs as a Lambda, and an invocation that fails is counted rather than
        # raised. A non-zero count means some contracts went unscored, so every average below
        # is over fewer records than the test split.
        for tag, scored in (("base", base_scored), ("fine-tuned", tuned_scored)):
            if (scored or {}).get("byoc_failure_count"):
                print(f"warning: {tag} had {scored['byoc_failure_count']:.0f} "
                      f"scorer invocation failure(s)")

        print(f"{'metric':24s} {'base':>9s} {'fine-tuned':>11s}")
        for name in SHOW:
            b = f"{base_scored[name]*100:9.1f}" if base_scored and name in base_scored else f"{'-':>9s}"
            t = f"{tuned_scored[name]*100:11.1f}" if tuned_scored and name in tuned_scored else f"{'-':>11s}"
            print(f"{name:24s} {b} {t}")

        self.save_json("scorer_base", base_scored)
        self.save_json("scorer_tuned", tuned_scored)
        return base_scored, tuned_scored


    def compare_models(self, base_scored, frontier_metrics, tuned_scored):
        """Keep the comparison table, parse-rate warnings, reference discussion, and summary export."""
        print(f"{'model':22s} {'accuracy':>9s} {'evidence-F1':>12s}   how")
        print("-" * 62)
        print(f"{'base Qwen3-4B':22s} "
              f"{base_scored['label_correct']*100 if base_scored else float('nan'):9.1f} "
              f"{base_scored['evidence_f1']*100 if base_scored else float('nan'):12.1f}   managed scorer")
        print(f"{self.frontier_model.split('.')[-1]:22s} "
              f"{frontier_metrics['accuracy']:9.1f} {frontier_metrics['evidence_f1']:12.1f}   local pass")
        print(f"{'fine-tuned Qwen3-4B':22s} "
              f"{tuned_scored['label_correct']*100 if tuned_scored else float('nan'):9.1f} "
              f"{tuned_scored['evidence_f1']*100 if tuned_scored else float('nan'):12.1f}   managed scorer")

        # An answer that does not parse scores zero on every item of its contract, so
        # json_valid has to be read before the two columns above mean anything. With `/no_think`
        # in the prompt both models parse almost everything, but print it rather than assume it:
        # if it ever drops, the rows above are part formatting failure and not comparable.
        if base_scored:
            valid = base_scored["json_valid"]
            print(f"\nbase json_valid: {valid:.1%}, {valid * len(self.documents):.0f} of "
                  f"{len(self.documents)} contracts returned parseable JSON")
            if valid < 0.95:
                print(f"  an unparseable answer scores 0 on all {len(self.checklist)} items, so the "
                      "base row above is part format")
                print("  failure. On the contracts it did answer, base is roughly "
                      f"{base_scored['label_correct'] / valid * 100:.1f} / "
                      f"{base_scored['evidence_f1'] / valid * 100:.1f}.")
                print("  That is a prompting bug, not a reading one: check that `/no_think` "
                      "survived into the")
                print("  test records, since a reasoning trace that never closes cannot reach "
                      "the JSON.")

        print("\nfull 123-contract test set:")
        print("  base 64.5 / 48.8   Sonnet 5 83.6 / 67.1   fine-tuned 87.4 / 75.4")
        print("  the fine-tuned 4B model scores above the frontier model on both metrics, by")
        print("  3.8 and 8.3 points; the two rows are not averaged the same way (see above),")
        print("  which accounts for about a point of that")
        print("  the base row is a real score, not a formatting artefact: base json_valid was")
        print("  99.2%, so 64.5 / 48.8 is what the base model actually knows about this")
        print("  checklist. Fine-tuning moved accuracy +23 and evidence-F1 +27 points from a")
        print("  working starting point, and this training set is small, so expect several")
        print("  points of movement on your own run rather than these decimals")

        self.save_json("summary", {
            "eval_contracts": len(self.documents),
            "checklist_items": len(self.checklist),
            "models": {
                "base": {"model": self.base_model_id, "how": "managed scorer",
                         "accuracy": base_scored["label_correct"] * 100 if base_scored else None,
                         "evidence_f1": base_scored["evidence_f1"] * 100 if base_scored else None},
                "frontier": {"model": self.frontier_model, "how": "local pass",
                             "accuracy": frontier_metrics["accuracy"],
                             "evidence_f1": frontier_metrics["evidence_f1"]},
                "fine_tuned": {"model": self.base_model_id, "how": "managed scorer",
                               "accuracy": tuned_scored["label_correct"] * 100 if tuned_scored else None,
                               "evidence_f1": tuned_scored["evidence_f1"] * 100 if tuned_scored else None},
            },
        })


    def scorer_distributions(self, base_scored, tuned_scored, execution_arn=None):
        """Check every aggregate against per-record data and draw the original distributions."""
        base_details = self.load_scorer_details("EvaluateBaseModel", execution_arn)
        tuned_details = self.load_scorer_details("EvaluateCustomModel", execution_arn)

        # If the per-record rows do not average to the aggregates, we are reading a different
        # run's file and every chart below it would be mislabelled.
        for name in SHOW:
            for aggregate, details in ((base_scored, base_details), (tuned_scored, tuned_details)):
                assert abs(details[name].mean() - aggregate[name]) < 1e-6, name
        print(f"{len(base_details)} per-contract rows per model, "
              f"reproducing every aggregate ✓")

        plot_scorer_spread(base_details, tuned_details)
        return base_details, tuned_details


    def write_judge_sample(self, dataset_prefix, count):
        """Write the same first-N test records; the notebook registers the dataset."""
        judge_key = f"datasets/{dataset_prefix}-judge-{count}/dataset.jsonl"
        test_key = self.find_s3_keys(f"datasets/{dataset_prefix}", "/test/", suffix=".jsonl")[-1]
        lines = self.s3.get_object(Bucket=self.bucket, Key=test_key)["Body"].read().decode().splitlines()
        self.s3.put_object(Bucket=self.bucket, Key=judge_key,
                           Body="\n".join(lines[:count]).encode() + b"\n")
        return f"s3://{self.bucket}/{judge_key}"


    def judge_results(self, execution_arn=None):
        """Retain missing-score coverage, both judge charts, and JSON exports."""
        judge_base = self.load_judge_scores("base-llmaj-eval", execution_arn)
        judge_tuned = self.load_judge_scores("custom-llmaj-eval", execution_arn)
        judged = {"base": judge_base, "fine-tuned": judge_tuned}

        plot_judge_bars(judged)
        plot_judge_radar(judged)

        # `to_json` renders missing scores as null; a plain dict would emit invalid `NaN`.
        for tag, scores in (("judge_base", judge_base), ("judge_tuned", judge_tuned)):
            if scores is not None:
                self.save_json(tag, json.loads(scores.to_json(orient="records")))
        return judged


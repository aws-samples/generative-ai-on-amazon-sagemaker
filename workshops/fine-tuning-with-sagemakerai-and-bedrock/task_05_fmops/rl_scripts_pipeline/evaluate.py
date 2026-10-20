"""Evaluate one or more models on the held-out benchmark and compare them.

For every model this script generates answers on the same prompts with the same
decoding settings as Lab 1 (temperature 0.6, top_p 0.9, 512 new tokens) and scores
each answer with:

* the Lab 1 benchmark: ROUGE-1, ROUGE-2 and ROUGE-L F1 against the reference
* the RLVR verifier score (``rlvr``) and answer length (``words``)
* an independent Bedrock judge (``judge``) from a different model family than the
  judge used during RL, so the RL model is not graded by the model it optimized for
* ``reward``: the same composite as training, recomputed with the independent judge

Outputs in --output-dir:
  per_sample_<name>.jsonl   one row per prompt, joinable on ``id``
  summary.json              mean metrics per model
  evaluation.json           written when --compare is set: paired significance tests
                            per comparison, plus a flat ``gate`` block for the pipeline

Examples:
  python evaluate.py --models base=Qwen/Qwen3-0.6B sft=/opt/ml/processing/input/sft
  python evaluate.py --models sft=... rl=... --baseline-dir /opt/ml/processing/input/baseline \\
      --compare base:rl base:sft sft:rl
"""

import argparse
import glob
import json
import os
import statistics

import model_io
import rewards
import rl_stats
import tracking

BENCHMARK_METRICS = ["rouge1", "rouge2", "rougeL"]
REWARD_METRICS = ["reward", "judge", "rlvr"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="name=source pairs; source is a dir, a model.tar.gz dir or an HF id")
    p.add_argument("--eval-data", default="/opt/ml/processing/input/eval")
    p.add_argument("--output-dir", default="/opt/ml/processing/output/eval")
    p.add_argument("--baseline-dir", default=None, help="Earlier evaluation output to compare against")
    p.add_argument("--compare", nargs="*", default=[], help="baseline:candidate pairs, e.g. base:rl")
    p.add_argument("--gate-comparison", default="base:rl", help="Comparison exposed in the flat gate block")
    p.add_argument("--primary-metric", default="reward")
    p.add_argument("--benchmark-metric", default="rougeL")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-judge-model-id", default="", help="Bedrock judge for evaluation; empty disables the judge")
    p.add_argument("--judge-weight", type=float, default=0.7)
    p.add_argument("--bedrock-region", default="")
    p.add_argument("--judge-workers", type=int, default=8)
    return p.parse_args()


def rouge_scores(prediction: str, reference: str):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(BENCHMARK_METRICS, use_stemmer=True)
    s = scorer.score(reference, prediction)
    return {k: s[k].fmeasure for k in BENCHMARK_METRICS}


def evaluate_model(name, source, records, args):
    model_dir = model_io.resolve_model_dir(source)
    model, tokenizer = model_io.load_for_generation(model_dir)
    splits = [model_io.split_record(r) for r in records]
    outputs = model_io.generate(
        model, tokenizer, [s[0] for s in splits], num_return_sequences=1,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_p=args.top_p, batch_size=args.batch_size,
    )
    del model
    model_io.release_cuda()

    mode = "hybrid" if args.eval_judge_model_id else "rlvr"
    cfg = rewards.RewardConfig(mode=mode, judge_model_id=args.eval_judge_model_id,
                               judge_weight=args.judge_weight, region=args.bedrock_region)
    items = [rewards.RewardInput(q, out[0], ref, ans) for (_, q, ref, ans), out in zip(splits, outputs)]
    scored = rewards.score_many(items, cfg, max_workers=args.judge_workers)

    rows = []
    for rec, item, score in zip(records, items, scored):
        row = {"id": rec["id"], "model": name, "question": item.question, "prediction": item.completion}
        row.update(rouge_scores(item.completion, item.reference))
        row["rlvr"] = score["rlvr"]
        row["judge"] = score["judge"]
        row["reward"] = score["reward"]
        row["words"] = score["rlvr_detail"]["words"]
        if score["judge_detail"]:
            row.update({f"judge_{k}": v for k, v in score["judge_detail"].items() if k != "score"})
        rows.append(row)
    return rows


def mean_of(rows, metric):
    vals = [r[metric] for r in rows if r.get(metric) is not None]
    return statistics.fmean(vals) if vals else None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    records = model_io.load_records(args.eval_data)[: args.max_samples]
    print(f"Evaluating on {len(records)} held-out prompts")

    results = {}
    if args.baseline_dir:
        for path in glob.glob(os.path.join(args.baseline_dir, "per_sample_*.jsonl")):
            name = os.path.basename(path)[len("per_sample_"):-len(".jsonl")]
            with open(path, encoding="utf-8") as f:
                results[name] = [json.loads(line) for line in f if line.strip()]
            print(f"Loaded baseline results for '{name}' from {path}")

    for spec in args.models:
        name, source = spec.split("=", 1)
        print(f"=== Evaluating {name} from {source}")
        results[name] = evaluate_model(name, source, records, args)
        with open(os.path.join(args.output_dir, f"per_sample_{name}.jsonl"), "w", encoding="utf-8") as f:
            for row in results[name]:
                f.write(json.dumps(row) + "\n")

    metrics = BENCHMARK_METRICS + REWARD_METRICS + ["words"]
    summary = {name: {m: mean_of(rows, m) for m in metrics} | {"n": len(rows)} for name, rows in results.items()}
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("summary:", json.dumps(summary, indent=2))

    if not args.compare:
        log_to_mlflow(args, records, summary, None)
        return
    tested = BENCHMARK_METRICS + [m for m in REWARD_METRICS if any(r.get(m) is not None for rows in results.values() for r in rows)]
    comparisons = {}
    for pair in args.compare:
        a, b = pair.split(":")
        if a not in results or b not in results:
            raise ValueError(f"Cannot compare {pair}: available models are {sorted(results)}")
        comparisons[f"{a}_vs_{b}"] = rl_stats.compare_models(results[a], results[b], tested, alpha=args.alpha)

    ga, gb = args.gate_comparison.split(":")
    gate_cmp = comparisons[f"{ga}_vs_{gb}"]
    primary = gate_cmp.get(args.primary_metric) or gate_cmp["rlvr"]
    bench = gate_cmp[args.benchmark_metric]
    gate = {
        "comparison": args.gate_comparison,
        "primary_metric": args.primary_metric if args.primary_metric in gate_cmp else "rlvr",
        "primary_delta": primary["delta"],
        "primary_ci_low": primary["ci_low"],
        "primary_p_value": primary["p_value_holm"],
        "primary_significant": 1 if primary["significant"] else 0,
        "benchmark_metric": args.benchmark_metric,
        "benchmark_delta": bench["delta"],
        "candidate_primary_mean": primary["candidate_mean"],
        "candidate_benchmark_mean": bench["candidate_mean"],
    }
    evaluation = {"summary": summary, "comparisons": comparisons, "gate": gate}
    with open(os.path.join(args.output_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)
    print("gate:", json.dumps(gate, indent=2))
    log_to_mlflow(args, records, summary, evaluation)


def log_to_mlflow(args, records, summary, evaluation):
    """Log the evaluation to MLflow when tracking is on (see tracking.py)."""
    with tracking.run("evaluation") as mlf:
        tracking.log_params(mlf, {"samples": len(records), "judge": args.eval_judge_model_id or "none",
                                  "models": ",".join(summary), "temperature": args.temperature,
                                  "max_new_tokens": args.max_new_tokens}, prefix="eval.")
        tracking.log_metrics(mlf, summary, prefix="eval.")
        if evaluation:
            tracking.log_metrics(mlf, evaluation["comparisons"], prefix="significance.")
            tracking.log_metrics(mlf, evaluation["gate"], prefix="gate.")
        tracking.log_artifacts(mlf, glob.glob(os.path.join(args.output_dir, "*.json*")), artifact_path="evaluation")


if __name__ == "__main__":
    main()

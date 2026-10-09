"""RL pass on top of the SFT model: on-policy preference building followed by DPO.

The job runs in two phases on one GPU instance.

Phase 1 builds preferences from the SFT model itself (on-policy):
  1. Sample ``num_candidates`` answers per prompt from the SFT model, with one generation
     process per GPU (generate_worker.py) on multi-GPU instances.
  2. Score every answer with the reward signal from ``rewards.py``
     (RLVR verifier, RLAIF Bedrock judge, or a hybrid of both).
  3. Keep the best and worst answer of each prompt as a (chosen, rejected) pair
     when their reward gap is at least ``min_reward_margin``.

Phase 2 runs Direct Preference Optimization with QLoRA. DPO optimizes the same
KL-regularized objective as PPO-based RLHF:

    max  E[r(x, y)]  -  beta * KL( pi_theta(y|x) || pi_ref(y|x) )

where the reference policy pi_ref is the frozen SFT model. ``beta`` is the KL
penalty coefficient: a larger beta keeps the policy closer to the SFT model and
limits reward hacking, a smaller beta lets the policy move further toward the
reward. With a PEFT adapter the reference model is the same network with the
adapter disabled, so no second copy of the weights is held in memory.

Phase 3 (optional, --run_final_eval true) evaluates the base, SFT and RL models
on the held-out set with evaluate.py in the same job. The models are already on
local disk, so this avoids a separate processing job with its own instance
start-up, package install and model download.

Inputs (SageMaker channels):
  /opt/ml/input/data/model    SFT model.tar.gz, or pass --sft_model_path <HF id>
  /opt/ml/input/data/prompts  dataset.json in the 02.01 "messages" format
  /opt/ml/input/data/eval     held-out evaluation set (only with --run_final_eval true)
Outputs:
  /opt/ml/model               merged RL model, deployable like the SFT model
  /opt/ml/output/data         preferences.jsonl, preference_stats.json, dpo_log_history.json
  /opt/ml/output/data/eval    per_sample_<model>.jsonl, summary.json, evaluation.json
"""

import json
import os
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

# This script runs as one process on one GPU. On a multi-GPU instance such as ml.g5.12xlarge the Hugging Face
# Trainer would otherwise wrap the model in DataParallel, which bitsandbytes 4-bit layers do not support.
# It must be set before torch initializes CUDA. The evaluation subprocess inherits it.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import DPOConfig, DPOTrainer, TrlParser

import model_io
import rewards
import tracking


@dataclass
class ScriptArguments:
    sft_model_path: str = field(default="/opt/ml/input/data/model", metadata={"help": "SFT model dir, model.tar.gz dir or HF id"})
    prompts_path: str = field(default="/opt/ml/input/data/prompts", metadata={"help": "Prompt dataset (messages format)"})
    max_prompts: int = field(default=64, metadata={"help": "Number of prompts used to build preferences"})
    num_candidates: int = field(default=4, metadata={"help": "Sampled answers per prompt"})
    gen_max_new_tokens: int = field(default=512)
    gen_temperature: float = field(default=0.9, metadata={"help": "High enough to give diverse candidates"})
    gen_top_p: float = field(default=0.95)
    gen_batch_size: int = field(default=16)
    min_reward_margin: float = field(default=0.05, metadata={"help": "Minimum reward gap between chosen and rejected"})
    min_pairs: int = field(default=16, metadata={"help": "Fail the job if fewer usable pairs remain"})
    judge_workers: int = field(default=16)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    merge_weights: bool = field(default=True)
    reward_config: Optional[str] = field(default=None, metadata={"help": "JSON reward config; REWARD_CONFIG env var wins"})
    # Phase 3: evaluation inside this job. Names carry a final_eval_ prefix so they never
    # collide with the eval_* fields of DPOConfig / TrainingArguments.
    run_final_eval: bool = field(default=False, metadata={"help": "Evaluate base, SFT and RL models after training"})
    final_eval_data_path: str = field(default="/opt/ml/input/data/eval")
    final_eval_base_model_id: str = field(default="", metadata={"help": "HF id of the base model"})
    final_eval_judge_model_id: str = field(default="", metadata={"help": "Independent Bedrock judge; empty disables it"})
    final_eval_bedrock_region: str = field(default="")
    final_eval_max_samples: int = field(default=60)
    final_eval_batch_size: int = field(default=16)


def instance_gpu_count() -> int:
    """GPUs on the instance. CUDA_VISIBLE_DEVICES is pinned to one GPU above, so ask SageMaker or nvidia-smi."""
    if os.environ.get("SM_NUM_GPUS", "").isdigit():
        return int(os.environ["SM_NUM_GPUS"])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return max(1, sum(1 for line in out.splitlines() if line.startswith("GPU ")))
    except (OSError, subprocess.CalledProcessError):
        return 1


def generate_candidates(args: ScriptArguments, model_dir: str, prompts, out_dir: str):
    """Sample candidate answers for every prompt, spread over all GPUs of the instance.

    Each GPU runs its own generate_worker.py process on a contiguous shard of the prompts, so a 4-GPU
    instance generates about 4 times faster. The training phase still uses one GPU.
    """
    n_gpus = min(instance_gpu_count(), len(prompts))
    if n_gpus <= 1:
        model, tokenizer = model_io.load_for_generation(model_dir)
        candidates = model_io.generate(
            model, tokenizer, prompts, num_return_sequences=args.num_candidates,
            max_new_tokens=args.gen_max_new_tokens, temperature=args.gen_temperature,
            top_p=args.gen_top_p, batch_size=args.gen_batch_size,
        )
        del model
        model_io.release_cuda()
        return candidates

    work_dir = os.path.join("/tmp", "generation_shards")
    os.makedirs(work_dir, exist_ok=True)
    shard_size = -(-len(prompts) // n_gpus)  # ceiling division
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_worker.py")
    procs = []
    for gpu in range(n_gpus):
        shard = prompts[gpu * shard_size:(gpu + 1) * shard_size]
        if not shard:
            continue
        in_path, out_path = os.path.join(work_dir, f"in_{gpu}.json"), os.path.join(work_dir, f"out_{gpu}.json")
        with open(in_path, "w", encoding="utf-8") as f:
            json.dump(shard, f)
        cmd = [sys.executable, worker, "--model-dir", model_dir, "--prompts", in_path, "--output", out_path,
               "--num-return-sequences", str(args.num_candidates), "--max-new-tokens", str(args.gen_max_new_tokens),
               "--temperature", str(args.gen_temperature), "--top-p", str(args.gen_top_p),
               "--batch-size", str(args.gen_batch_size), "--seed", str(42 + gpu)]
        procs.append((gpu, out_path, subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})))
    print(f"Generating {len(prompts)} prompts x {args.num_candidates} candidates on {len(procs)} GPUs")

    candidates = []
    for gpu, out_path, proc in procs:
        if proc.wait() != 0:
            for _, _, other in procs:
                other.kill()
            raise RuntimeError(f"Generation worker on GPU {gpu} failed with exit code {proc.returncode}")
        with open(out_path, encoding="utf-8") as f:
            candidates.extend(json.load(f))
    assert len(candidates) == len(prompts), (len(candidates), len(prompts))
    return candidates


def build_preferences(args: ScriptArguments, cfg: rewards.RewardConfig, sft_dir: str, out_dir: str):
    records = model_io.load_records(args.prompts_path)
    random.Random(42).shuffle(records)
    records = records[: args.max_prompts]
    print(f"Building preferences from {len(records)} prompts x {args.num_candidates} candidates, reward mode={cfg.mode}")

    splits = [model_io.split_record(r) for r in records]
    candidates = generate_candidates(args, sft_dir, [s[0] for s in splits], out_dir)

    items, owners = [], []
    for rec_idx, ((_, question, reference, ref_answer), cands) in enumerate(zip(splits, candidates)):
        for cand in cands:
            items.append(rewards.RewardInput(question, cand, reference, ref_answer))
            owners.append(rec_idx)
    scored = rewards.score_many(items, cfg, max_workers=args.judge_workers)

    per_prompt = {}
    for owner, item, score in zip(owners, items, scored):
        if score["reward"] is not None:
            per_prompt.setdefault(owner, []).append((score["reward"], item.completion, score))

    pairs, margins, chosen_r, rejected_r = [], [], [], []
    for rec_idx, scored_cands in per_prompt.items():
        if len(scored_cands) < 2:
            continue
        scored_cands.sort(key=lambda t: t[0], reverse=True)
        best, worst = scored_cands[0], scored_cands[-1]
        margin = best[0] - worst[0]
        if margin < args.min_reward_margin:
            continue
        prompt_msgs = splits[rec_idx][0]
        pairs.append({
            "id": records[rec_idx]["id"],
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant", "content": best[1]}],
            "rejected": [{"role": "assistant", "content": worst[1]}],
            "chosen_reward": best[0],
            "rejected_reward": worst[0],
        })
        margins.append(margin)
        chosen_r.append(best[0])
        rejected_r.append(worst[0])

    all_rewards = [s["reward"] for s in scored if s["reward"] is not None]
    stats = {
        "reward_mode": cfg.mode,
        "prompts": len(records),
        "candidates_scored": len(all_rewards),
        "candidates_unscored": len(scored) - len(all_rewards),
        "pairs": len(pairs),
        "mean_candidate_reward": statistics.fmean(all_rewards) if all_rewards else None,
        "mean_chosen_reward": statistics.fmean(chosen_r) if chosen_r else None,
        "mean_rejected_reward": statistics.fmean(rejected_r) if rejected_r else None,
        "mean_margin": statistics.fmean(margins) if margins else None,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "preferences.jsonl"), "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    with open(os.path.join(out_dir, "preference_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("preference_stats:", json.dumps(stats))
    if len(pairs) < args.min_pairs:
        raise RuntimeError(
            f"Only {len(pairs)} preference pairs passed min_reward_margin={args.min_reward_margin}; "
            f"need {args.min_pairs}. Add prompts, raise num_candidates or lower min_reward_margin."
        )
    return pairs


def train_dpo(args: ScriptArguments, training_args: DPOConfig, sft_dir: str, pairs, out_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(sft_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs]
    dataset = Dataset.from_list(rows)
    eval_dataset = None
    if len(rows) >= 40:
        split = dataset.train_test_split(test_size=0.1, seed=42)
        dataset, eval_dataset = split["train"], split["test"]
        training_args.eval_strategy = "epoch"

    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        sft_dir,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype,
        ),
        torch_dtype=compute_dtype,
        use_cache=not training_args.gradient_checkpointing,
    )
    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
    )

    training_args.save_strategy = "no"
    training_args.logging_steps = 1
    print(f"DPO: {len(dataset)} train pairs, beta (KL coefficient) = {training_args.beta}, loss = {training_args.loss_type}")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # reference policy = SFT weights with the adapter disabled
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    with open(os.path.join(out_dir, "dpo_log_history.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    adapter_dir = "/tmp/dpo_adapter"
    trainer.save_model(adapter_dir)
    del trainer, model
    model_io.release_cuda()

    final_dir = training_args.output_dir
    if args.merge_weights:
        print("Merging the DPO adapter into the SFT weights")
        base = AutoModelForCausalLM.from_pretrained(sft_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        merged.save_pretrained(final_dir, safe_serialization=True)
    else:
        import shutil

        shutil.copytree(adapter_dir, final_dir, dirs_exist_ok=True)
    tokenizer.save_pretrained(final_dir)
    print(f"RL model saved to {final_dir}")
    return final_dir


def run_final_eval(args: ScriptArguments, sft_dir: str, rl_dir: str, out_dir: str, mlflow_run_id=None):
    """Run evaluate.py as a child process, so all training GPU memory is released first.

    With MLflow tracking on, the child resumes this job's run (MLFLOW_RUN_ID), so the evaluation metrics and
    significance tests land in the same run as the preference stats and the DPO curves.
    """
    if not args.merge_weights:
        print("Skipping the final evaluation: it needs merged RL weights (merge_weights true).")
        return
    base_id = args.final_eval_base_model_id
    # RL straight from the base model: the starting policy IS the base model, so evaluate it once as "base".
    rl_from_base = bool(base_id) and args.sft_model_path == base_id
    models = [f"rl={rl_dir}"]
    if base_id and rl_from_base:
        models.insert(0, f"base={base_id}")
        compare = ["base:rl"]
    elif base_id:
        models[:0] = [f"base={base_id}", f"sft={sft_dir}"]
        compare = ["base:sft", "base:rl", "sft:rl"]
    else:
        models.insert(0, f"sft={sft_dir}")
        compare = ["sft:rl"]
    cmd = [
        sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py"),
        "--models", *models,
        "--eval-data", args.final_eval_data_path,
        "--output-dir", os.path.join(out_dir, "eval"),
        "--compare", *compare,
        "--gate-comparison", "base:rl" if base_id else "sft:rl",
        "--max-samples", str(args.final_eval_max_samples),
        "--batch-size", str(args.final_eval_batch_size),
        "--eval-judge-model-id", args.final_eval_judge_model_id,
        "--bedrock-region", args.final_eval_bedrock_region,
        "--judge-workers", str(args.judge_workers),
    ]
    env = dict(os.environ)
    if mlflow_run_id:
        env["MLFLOW_RUN_ID"] = mlflow_run_id
    print("Final evaluation:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = TrlParser((ScriptArguments, DPOConfig))
    args, training_args = parser.parse_args_and_config()
    set_seed(training_args.seed)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    cfg = rewards.RewardConfig.from_json(os.environ.get("REWARD_CONFIG") or args.reward_config)
    print("reward_config:", cfg.to_json())
    out_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

    with tracking.run("rl-dpo") as mlf:
        tracking.log_params(mlf, json.loads(cfg.to_json()), prefix="reward.")
        tracking.log_params(mlf, {k: getattr(args, k) for k in (
            "sft_model_path", "max_prompts", "num_candidates", "gen_temperature", "gen_top_p", "gen_max_new_tokens",
            "min_reward_margin", "lora_r", "lora_alpha", "lora_dropout")}, prefix="rl.")
        tracking.log_params(mlf, {k: getattr(training_args, k) for k in (
            "beta", "loss_type", "learning_rate", "num_train_epochs", "per_device_train_batch_size",
            "gradient_accumulation_steps", "max_length", "max_prompt_length", "lr_scheduler_type", "warmup_ratio")},
            prefix="dpo.")

        sft_dir = model_io.resolve_model_dir(args.sft_model_path)
        pairs = build_preferences(args, cfg, sft_dir, out_dir)
        with open(os.path.join(out_dir, "preference_stats.json"), encoding="utf-8") as f:
            tracking.log_metrics(mlf, json.load(f), prefix="preferences.")

        # With report_to="mlflow", the DPO trainer streams its curves (loss, rewards/margins, rewards/accuracies,
        # rewards/chosen, rewards/rejected) into this same active run while it trains.
        rl_dir = train_dpo(args, training_args, sft_dir, pairs, out_dir)
        tracking.log_artifacts(mlf, [os.path.join(out_dir, n) for n in ("preference_stats.json", "dpo_log_history.json")],
                               artifact_path="rl")

        if args.run_final_eval:
            run_final_eval(args, sft_dir, rl_dir, out_dir, mlflow_run_id=tracking.active_run_id(mlf))


if __name__ == "__main__":
    main()

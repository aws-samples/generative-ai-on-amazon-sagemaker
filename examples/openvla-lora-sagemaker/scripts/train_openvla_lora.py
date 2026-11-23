#!/usr/bin/env python
"""LoRA fine-tuning of OpenVLA-7B on a SageMaker Training Job (Accelerate DDP).

Channels (see 2-fine-tune-vla.ipynb):
    train  ->  /opt/ml/input/data/train   HF DatasetDict saved with save_to_disk(); splits train/validation,
                                           columns: image (PIL), instruction (str), action (7 floats in [-1, 1])
Outputs:
    /opt/ml/checkpoints/step-<N>/          LoRA adapter + optimizer state (synced to S3, used to resume)
    /opt/ml/model/adapter/                 final LoRA adapter
    /opt/ml/model/hf_merged/               adapter merged into the base model (deployable HF checkpoint)
"""
import argparse
import glob
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, get_scheduler

sys.path.insert(0, str(Path(__file__).parent))
from openvla_utils import ActionTokenizer, build_training_ids  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="YAML recipe; CLI flags override it")
    p.add_argument("--base_model", default="openvla/openvla-7b")
    p.add_argument("--data_dir", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--checkpoint_dir", default="/opt/ml/checkpoints")
    p.add_argument("--num_train_epochs", type=int, default=10)
    p.add_argument("--max_train_steps", type=int, default=0, help=">0 overrides epochs (smoke tests)")
    p.add_argument("--per_device_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--global_batch_size", type=int, default=0,
                   help=">0 derives gradient_accumulation_steps from the number of GPUs")
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--lr_scheduler", default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_modules", default="all-linear")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--checkpoint_steps", type=int, default=200)
    p.add_argument("--keep_checkpoints", type=int, default=2)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_merge", action="store_true")

    args, _ = p.parse_known_args()
    if args.config:
        with open(args.config) as f:
            recipe = yaml.safe_load(f) or {}
        unknown = set(recipe) - set(vars(args))
        if unknown:
            raise ValueError(f"Unknown keys in {args.config}: {sorted(unknown)}")
        p.set_defaults(**recipe)
        args, _ = p.parse_known_args()
    return args


class MlflowLogger:
    """Logs to the workshop's SageMaker MLflow App when MLFLOW_TRACKING_URI is set; never fails the job."""

    def __init__(self, enabled, params):
        self.run = None
        if not (enabled and os.environ.get("MLFLOW_TRACKING_URI")):
            return
        try:
            import mlflow

            mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "openvla-lora"))
            self.run = mlflow.start_run(run_name=os.environ.get("TRAINING_JOB_NAME"))
            mlflow.log_params({k: str(v) for k, v in params.items()})
            self._mlflow = mlflow
        except Exception as e:  # tracking is best-effort
            print(f"[mlflow] disabled: {e}", flush=True)

    def log(self, metrics, step):
        print(f"[step {step}] " + " ".join(f"{k}={v:.5f}" for k, v in metrics.items()), flush=True)
        if self.run:
            try:
                self._mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                print(f"[mlflow] log failed: {e}", flush=True)

    def close(self):
        if self.run:
            self._mlflow.end_run()


def make_collate(processor, action_tokenizer):
    tok = processor.tokenizer

    def collate(examples):
        ids, labels, pixels = [], [], []
        for ex in examples:
            i, l = build_training_ids(tok, action_tokenizer, ex["instruction"], ex["action"])
            ids.append(torch.tensor(i))
            labels.append(torch.tensor(l))
            pixels.append(processor.image_processor.apply_transform(ex["image"].convert("RGB")))
        input_ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=tok.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(tok.pad_token_id),
            "pixel_values": torch.stack(pixels).to(torch.bfloat16),
            "labels": labels,
        }

    return collate


def action_metrics(logits, batch, action_tokenizer):
    """Teacher-forced action-token accuracy and L1 (as in openvla/vla-scripts/finetune.py)."""
    n_patches = logits.shape[1] - batch["input_ids"].shape[1]
    preds = logits[:, n_patches:-1].argmax(dim=2)
    gt = batch["labels"][:, 1:]
    mask = gt > action_tokenizer.action_token_begin_idx
    correct = ((preds == gt) & mask).sum().float()
    total = mask.sum().float()
    pred_a = torch.tensor(action_tokenizer.decode(preds[mask].cpu().numpy()))
    gt_a = torch.tensor(action_tokenizer.decode(gt[mask].cpu().numpy()))
    l1_sum = (pred_a - gt_a).abs().sum().float().to(logits.device)
    return correct, total, l1_sum


def save_checkpoint(accelerator, model, optimizer, scheduler, step, args):
    if not accelerator.is_main_process:
        return
    path = Path(args.checkpoint_dir) / f"step-{step:06d}"
    accelerator.unwrap_model(model).save_pretrained(path / "adapter", save_embedding_layers=False)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step},
               path / "trainer_state.pt")
    for old in sorted(glob.glob(f"{args.checkpoint_dir}/step-*"))[: -args.keep_checkpoints]:
        shutil.rmtree(old, ignore_errors=True)
    print(f"saved checkpoint {path}", flush=True)


def latest_checkpoint(args):
    done = [p for p in sorted(glob.glob(f"{args.checkpoint_dir}/step-*")) if os.path.exists(f"{p}/trainer_state.pt")]
    return done[-1] if done else None


@torch.no_grad()
def evaluate(accelerator, model, loader, action_tokenizer):
    model.eval()
    sums = torch.zeros(4, device=accelerator.device)  # loss*n, n, correct, l1_sum
    for batch in loader:
        out = model(**batch)
        correct, total, l1_sum = action_metrics(out.logits, batch, action_tokenizer)
        sums += torch.stack([out.loss.detach() * total, total, correct, l1_sum])
    sums = accelerator.reduce(sums, reduction="sum")
    model.train()
    n = sums[1].clamp(min=1)
    return {"val/loss": (sums[0] / n).item(), "val/action_accuracy": (sums[2] / n).item(),
            "val/l1": (sums[3] / n).item()}


def merge_and_save(args, processor):
    """Merges the final adapter into a fresh bf16 base and writes a self-contained HF checkpoint."""
    from huggingface_hub import snapshot_download

    out = Path(args.model_dir) / "hf_merged"
    base = AutoModelForVision2Seq.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                  low_cpu_mem_usage=True, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, Path(args.model_dir) / "adapter").merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    processor.save_pretrained(out)
    # trust_remote_code models need their modeling/processing *.py next to the weights
    for f in Path(snapshot_download(args.base_model, allow_patterns=["*.py"])).glob("*.py"):
        if not (out / f.name).exists():
            shutil.copy(f, out / f.name)
    stats = Path(args.data_dir) / "action_stats.json"
    if stats.exists():  # action normalization stats from notebook 1, used by the endpoint to un-normalize
        shutil.copy(stats, out / stats.name)
    print(f"merged model written to {out}", flush=True)


def main():
    args = parse_args()
    if args.global_batch_size:
        world = int(os.environ.get("WORLD_SIZE", 1))
        args.gradient_accumulation_steps = max(1, args.global_batch_size // (args.per_device_batch_size * world))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    set_seed(args.seed)
    is_main = accelerator.is_main_process
    tracker = MlflowLogger(is_main, vars(args))

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="eager",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    target = args.lora_target_modules
    target = target if target == "all-linear" else [m.strip() for m in target.split(",")]
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=target, init_lora_weights="gaussian",
    ))
    if is_main:
        model.print_trainable_parameters()

    resume = latest_checkpoint(args)
    if resume:
        set_peft_model_state_dict(model, load_file(f"{resume}/adapter/adapter_model.safetensors"))

    data = load_from_disk(args.data_dir)
    collate = make_collate(processor, action_tokenizer)
    train_loader = DataLoader(data["train"], batch_size=args.per_device_batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(data["validation"], batch_size=args.per_device_batch_size,
                            collate_fn=collate, num_workers=args.num_workers) if "validation" in data else None

    steps_per_epoch = math.ceil(len(train_loader) / accelerator.num_processes / args.gradient_accumulation_steps)
    total_steps = args.max_train_steps or args.num_train_epochs * steps_per_epoch
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    # Accelerate's prepared scheduler advances num_processes times per optimizer step
    scheduler = get_scheduler(args.lr_scheduler, optimizer,
                              num_warmup_steps=int(args.warmup_ratio * total_steps) * accelerator.num_processes,
                              num_training_steps=total_steps * accelerator.num_processes)

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    step = 0
    if resume:
        state = torch.load(f"{resume}/trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        accelerator.print(f"resumed from {resume} at step {step}")

    accelerator.print(f"train samples={len(data['train'])} gpus={accelerator.num_processes} "
                      f"global_batch={args.per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps} "
                      f"steps/epoch={steps_per_epoch} total_steps={total_steps}")
    model.train()
    t0, window = time.time(), torch.zeros(4, device=accelerator.device)
    while step < total_steps:
        epoch, offset = divmod(step, steps_per_epoch)
        train_loader.set_epoch(epoch)
        loader = accelerator.skip_first_batches(train_loader, offset * args.gradient_accumulation_steps) \
            if offset else train_loader
        for batch in loader:
            with accelerator.accumulate(model):
                out = model(**batch)
                accelerator.backward(out.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            correct, total, l1_sum = action_metrics(out.logits.detach(), batch, action_tokenizer)
            window += torch.stack([out.loss.detach() * total, total, correct, l1_sum])
            if not accelerator.sync_gradients:
                continue
            step += 1
            if step % args.logging_steps == 0 or step == total_steps:
                w = accelerator.reduce(window, reduction="sum")
                n = w[1].clamp(min=1)
                tracker.log({"train/loss": (w[0] / n).item(), "train/action_accuracy": (w[2] / n).item(),
                             "train/l1": (w[3] / n).item(), "train/lr": scheduler.get_last_lr()[0],
                             "train/epoch": step / steps_per_epoch,
                             "train/sec_per_step": (time.time() - t0) / args.logging_steps}, step)
                t0, window = time.time(), torch.zeros_like(window)
            if val_loader is not None and (step % args.eval_steps == 0 or step == total_steps):
                metrics = evaluate(accelerator, model, val_loader, action_tokenizer)
                tracker.log(metrics, step)
            if step % args.checkpoint_steps == 0:
                save_checkpoint(accelerator, model, optimizer, scheduler, step, args)
            if step >= total_steps:
                break

    accelerator.wait_for_everyone()
    if is_main:
        # "all-linear" also puts LoRA on lm_head; PEFT would then save the full (unchanged) base lm_head too,
        # doubling the adapter to ~460 MB. The base weights come from the base model, so skip them.
        accelerator.unwrap_model(model).save_pretrained(Path(args.model_dir) / "adapter", save_embedding_layers=False)
    tracker.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()
    if not is_main:
        return
    del model, optimizer, scheduler, train_loader, val_loader
    accelerator.free_memory()
    if not args.skip_merge:
        merge_and_save(args, processor)


if __name__ == "__main__":
    main()

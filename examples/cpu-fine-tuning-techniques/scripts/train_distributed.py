#!/usr/bin/env python
"""Distributed full fine-tuning on CPU: one process, DDP or FSDP (SageMaker training job entry point, notebook 05).

Launched by torchrun (ModelTrainer's Torchrun()), one process per instance; the processes talk over the gloo
backend, PyTorch's CPU counterpart of NCCL. Every strategy trains the same model on the same global batch:

* single: one process does everything (the baseline)
* ddp:    each rank holds a full copy of the model and trains on its share of the batch; gradients are averaged
          across ranks (all-reduce) after every backward pass
* fsdp:   each rank holds only a shard of the weights, gradients and optimizer state; the full weights of a layer
          are gathered just in time for its forward/backward pass and freed again (FSDP2 `fully_shard`)

This is a full fine-tune, not LoRA: FSDP's point is to shard what every rank would otherwise hold in full, and for
a LoRA adapter that is only a few MB. Rank 0 evaluates the trained model (exact JSON match) and writes
training_metrics.json with per-rank memory and timing.
"""
import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent))
from booking_data import make_splits, messages  # noqa: E402
from common import Tracker, chat, evaluate_booking, load  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["single", "ddp", "fsdp"], default="ddp")
    p.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--global_batch_size", type=int, default=16, help="examples per step, summed over all ranks")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--n_test", type=int, default=30)
    p.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    return p.parse_args()


def peak_rss_gb():
    """Peak resident memory of this process (ru_maxrss is KB on Linux, bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 3 if platform.system() == "Darwin" else 1024 ** 2)


def encode(tok, ex):
    """Prompt + answer token ids; the loss covers the answer only (prompt labels = -100)."""
    prompt = chat(tok, messages(ex["request"]), generation=True)
    ids = tok(prompt + ex["target"] + tok.eos_token, add_special_tokens=False).input_ids
    n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
    return ids, [-100] * n_prompt + ids[n_prompt:]


def collate(batch, pad_id):
    width = max(len(ids) for ids, _ in batch)
    ids = torch.tensor([i + [pad_id] * (width - len(i)) for i, _ in batch])
    labels = torch.tensor([l + [-100] * (width - len(l)) for _, l in batch])
    return ids, (ids != pad_id) | (labels != -100), labels


def main():
    args = parse_args()
    t0 = time.time()
    distributed = "WORLD_SIZE" in os.environ  # set by torchrun
    if distributed:
        # gloo picks a network interface from the hostname; on SageMaker use the one the platform reports
        if "SM_NETWORK_INTERFACE_NAME" in os.environ:
            os.environ.setdefault("GLOO_SOCKET_IFNAME", os.environ["SM_NETWORK_INTERFACE_NAME"])
        dist.init_process_group("gloo")
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    if args.strategy != "single" and world < 2:
        print(f"[rank {rank}] note: {args.strategy} with a single process (world size 1)", flush=True)
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    tracker = Tracker("distributed", {**vars(args), "world_size": world}) if rank == 0 else None

    train, test = make_splits(n_train=400, n_test=args.n_test)
    tok, model = load(args.model_id)
    torch.set_num_threads(os.cpu_count())
    n_params = sum(p.numel() for p in model.parameters())
    log(f"world size {world}, strategy {args.strategy}, {n_params / 1e6:.0f} M parameters (all trained), "
        f"{os.cpu_count()} CPUs per rank")

    if args.strategy == "ddp" and distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model)  # CPU module: no device_ids; gradients are all-reduced during backward
    elif args.strategy == "fsdp" and distributed:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        mesh = init_device_mesh("cpu", (world,))  # one dimension: shard across all ranks
        for layer in model.model.layers:  # shard each decoder layer, then the rest (embeddings, norm, head)
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
    local_params = sum(p.to_local().numel() if hasattr(p, "to_local") else p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    encoded = [encode(tok, ex) for ex in train]
    per_rank = args.global_batch_size // world
    model.train()
    t_train, losses = time.time(), []
    for step in range(1, args.steps + 1):
        start = (step - 1) * args.global_batch_size
        batch = [encoded[(start + j) % len(encoded)] for j in range(args.global_batch_size)]
        ids, mask, labels = collate(batch[rank * per_rank:(rank + 1) * per_rank], tok.pad_token_id)  # this rank's share
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        # average the loss over ranks for logging (training itself already synchronized the gradients)
        mean_loss = loss.detach().clone()
        if distributed:
            dist.all_reduce(mean_loss)
            mean_loss /= world
        losses.append(round(mean_loss.item(), 4))
        if tracker:
            tracker.log({"loss": mean_loss.item()}, step=step)
        if step == 1 or step % 5 == 0:
            log(f"[step {step}] loss={mean_loss.item():.4f} sec/step={(time.time() - t_train) / step:.2f}")
    seconds_training = time.time() - t_train

    # per-rank memory and parameter count, gathered on rank 0
    mine = torch.tensor([peak_rss_gb(), local_params / 1e6], dtype=torch.float64)
    per_rank_stats = [mine.clone() for _ in range(world)]
    if distributed:
        dist.all_gather(per_rank_stats, mine)

    # Full weights on rank 0 for evaluation: FSDP gathers the shards; DDP/single already hold them
    if args.strategy == "fsdp" and distributed:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
        full = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank == 0:
        if args.strategy == "fsdp" and distributed:
            _, eval_model = load(args.model_id)
            eval_model.load_state_dict(full)
        else:
            eval_model = model.module if hasattr(model, "module") else model
        tuned, outputs = evaluate_booking(eval_model, tok, test)
        log(f"finetuned {json.dumps(tuned)}")
        metrics = {
            "strategy": args.strategy, "world_size": world, "model_id": args.model_id, "steps": args.steps,
            "global_batch_size": args.global_batch_size, "per_rank_batch_size": per_rank,
            "parameters_total_m": round(n_params / 1e6, 1), "loss": losses,
            "sec_per_step": round(seconds_training / args.steps, 2), "seconds_training": round(seconds_training),
            "ranks": [{"rank": r, "peak_rss_gb": round(s[0].item(), 2), "parameters_held_m": round(s[1].item(), 1)}
                      for r, s in enumerate(per_rank_stats)],
            "finetuned": tuned, "examples": [{"request": ex["request"], "target": ex["target"], "finetuned": o}
                                             for ex, o in list(zip(test, outputs))[:3]],
            "seconds_total": round(time.time() - t0),
        }
        out = Path(args.model_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
        tracker.log({"sec_per_step": metrics["sec_per_step"], "peak_rss_gb_rank0": metrics["ranks"][0]["peak_rss_gb"],
                     **{f"finetuned_{k}": v for k, v in tuned.items()}})
        log(f"done in {metrics['seconds_total']}s; per rank: {metrics['ranks']}")
        tracker.close(out, "training_metrics.json")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Generate candidate answers for one shard of prompts on one GPU.

dpo_train.py starts one copy of this script per GPU, each with CUDA_VISIBLE_DEVICES set to a single GPU,
then concatenates the outputs in shard order. Output: a JSON list with one list of candidate answers per prompt.
"""

import argparse
import json

import model_io


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompts", required=True, help="JSON list of chat prompts (lists of messages)")
    p.add_argument("--output", required=True)
    p.add_argument("--num-return-sequences", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.prompts, encoding="utf-8") as f:
        prompts = json.load(f)
    model, tokenizer = model_io.load_for_generation(args.model_dir)
    candidates = model_io.generate(
        model, tokenizer, prompts, num_return_sequences=args.num_return_sequences,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
        batch_size=args.batch_size, seed=args.seed,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(candidates, f)


if __name__ == "__main__":
    main()

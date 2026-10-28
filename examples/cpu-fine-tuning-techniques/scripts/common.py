"""Shared by the training job scripts: model loading, prompts, booking-task evaluation, metrics output."""
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from booking_data import messages

FIELDS = ["name", "party_size", "day", "time"]


def load(model_id):
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count())
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32) if _tf5() else \
        AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    return tok, model


def _tf5():
    import transformers
    return int(transformers.__version__.split(".")[0]) >= 5


def chat(tok, msgs, generation):
    kw = {"enable_thinking": False} if "qwen3" in tok.name_or_path.lower() else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=generation, **kw)


def booking_prompt(tok, request):
    return chat(tok, messages(request), generation=True)


def parse_booking(text, lenient=False):
    """The JSON object in `text`, or None. lenient=True also accepts it inside a markdown code block."""
    text = text.strip()
    if lenient and text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        got = json.loads(text)
    except json.JSONDecodeError:
        return None
    return got if isinstance(got, dict) else None


@torch.inference_mode()
def generate(model, tok, prompts, max_new_tokens=48, batch_size=16):
    model.eval()
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += [tok.decode(s, skip_special_tokens=True).strip() for s in out[:, enc.input_ids.shape[1]:]]
    return outs


def evaluate_booking(model, tok, test):
    """Exact JSON match and per-field accuracy on held-out booking requests."""
    outputs = generate(model, tok, [booking_prompt(tok, ex["request"]) for ex in test])
    exact, hits = 0, {f: 0 for f in FIELDS}
    for ex, text in zip(test, outputs):
        want, got = json.loads(ex["target"]), parse_booking(text) or {}
        exact += got == want
        for f in FIELDS:
            hits[f] += got.get(f) == want[f]
    n = len(test)
    return {"exact_match": exact / n, **{f"acc_{f}": h / n for f, h in hits.items()}}, outputs


def save(model_dir, model, tok, metrics, t0):
    out = Path(model_dir)
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    metrics["seconds_total"] = round(time.time() - t0)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"done in {metrics['seconds_total']}s", flush=True)

"""CPU endpoint handler: base model from the Hugging Face Hub + the LoRA adapter shipped in the model data.

Request:  {"request": "Table for four on friday at 7pm, name's Dana."}
          optional "model": "base" answers with the adapter switched off (the base model), for comparison
Response: {"booking": {"name": "Dana", "party_size": 4, "day": "Friday", "time": "19:00"}, "raw": "...",
           "model": "finetuned"}
"""
import contextlib
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Same instruction as scripts/booking_data.py (not packaged with the endpoint): keep the two in sync
INSTRUCTION = ("Extract the booking as JSON with keys name, party_size, day, time. "
               "Reply with the JSON only.")


def model_fn(model_dir, context=None):
    adapter = os.path.join(model_dir, "adapter")
    base_id = json.load(open(os.path.join(adapter, "adapter_config.json")))["base_model_name_or_path"]
    print(f"base model {base_id} (from the Hugging Face Hub) + adapter {adapter}", flush=True)
    tok = AutoTokenizer.from_pretrained(adapter)
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32)
    # Not merged, so the adapter can be switched off per request ("model": "base")
    model = PeftModel.from_pretrained(model, adapter).eval()
    return {"model": model, "tok": tok}


def input_fn(body, content_type="application/json"):
    return json.loads(body)


def predict_fn(data, ctx):
    tok, model = ctx["tok"], ctx["model"]
    which = "base" if data.get("model") == "base" else "finetuned"
    msgs = [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": data["request"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    adapter_off = model.disable_adapter() if which == "base" else contextlib.nullcontext()
    with torch.inference_mode(), adapter_off:
        out = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    raw = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    try:
        booking = json.loads(raw)
    except json.JSONDecodeError:
        booking = None
    return {"booking": booking, "raw": raw, "model": which}


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"

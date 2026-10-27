"""CPU endpoint handler: base model from the Hugging Face Hub + the LoRA adapter shipped in the model data.

Request:  {"request": "Table for four on friday at 7pm, name's Dana."}
Response: {"booking": {"name": "Dana", "party_size": 4, "day": "Friday", "time": "19:00"}, "raw": "..."}
"""
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCTION = ("Extract the booking as JSON with keys name, party_size, day, time. "
               "Reply with the JSON only.")


def model_fn(model_dir, context=None):
    adapter = os.path.join(model_dir, "adapter")
    base_id = json.load(open(os.path.join(adapter, "adapter_config.json")))["base_model_name_or_path"]
    tok = AutoTokenizer.from_pretrained(adapter)
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
    return {"model": model, "tok": tok}


def input_fn(body, content_type="application/json"):
    return json.loads(body)["request"]


def predict_fn(request, ctx):
    tok, model = ctx["tok"], ctx["model"]
    msgs = [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": request}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    raw = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    try:
        booking = json.loads(raw)
    except json.JSONDecodeError:
        booking = None
    return {"booking": booking, "raw": raw}


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"

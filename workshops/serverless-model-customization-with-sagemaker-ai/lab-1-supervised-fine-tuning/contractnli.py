"""
ContractNLI: review an NDA against a fixed 17-point checklist and cite the clause.

Everything the lab needs, in seven short sections:

  1. load the dataset          ensure_dataset, load, doc_spans, gold_for
  2. build the prompt          build_prompt
  3. read the model's answer   robust_parse
  4. score one contract        score_doc
  5. roll up the metrics       aggregate
  6. run a model over data     run

Dataset: Koreeda & Manning, ContractNLI (Findings of EMNLP 2021), CC-BY-4.0.
"""

import io
import json
import os
import pathlib
import re
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config


LABELS = ["Entailment", "Contradiction", "NotMentioned"]
DATA = os.environ.get("CONTRACTNLI_DIR", "./data/contract-nli")
DATASET_URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"


# ---------------------------------------------------------------- 1. the data

def ensure_dataset(target="./data"):
    """Download and unpack ContractNLI once."""
    root = pathlib.Path(target)
    if (root / "contract-nli" / "train.json").exists():
        return str(root / "contract-nli")
    root.mkdir(parents=True, exist_ok=True)
    print(f"downloading ContractNLI from {DATASET_URL} ...")
    ctx = None
    try:  # some managed environments ship a stale SSL_CERT_FILE
        import ssl

        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    with urllib.request.urlopen(DATASET_URL, context=ctx) as r:
        blob = r.read()
    zipfile.ZipFile(io.BytesIO(blob)).extractall(root)
    print(f"unpacked to {root / 'contract-nli'}")
    return str(root / "contract-nli")


def load(split):
    """Return (documents, checklist). split is 'train', 'dev' or 'test'."""
    d = json.load(open(f"{DATA}/{split}.json"))
    return d["documents"], d["labels"]


def doc_spans(doc):
    """The contract as [(span_number, text), ...] using the dataset's offsets."""
    out = []
    for i, (start, end) in enumerate(doc["spans"]):
        text = re.sub(r"\s+", " ", doc["text"][start:end].strip())
        if text:
            out.append((i, text))
    return out


def gold_for(doc):
    """The expert annotation: {checklist_key: {'choice': ..., 'spans': [...]}}."""
    return doc["annotation_sets"][0]["annotations"]


# -------------------------------------------------------------- 2. the prompt

INSTRUCTION = """You are a contract review assistant. You review a non-disclosure agreement (NDA) against a fixed checklist of {n} legal hypotheses.

For EACH hypothesis, decide:
- "Entailment": the contract states or implies the hypothesis is true.
- "Contradiction": the contract states something that conflicts with the hypothesis.
- "NotMentioned": the contract does not address it.

Also cite the span numbers that justify the decision (the exact spans a lawyer would point to). Cite spans only for Entailment or Contradiction; use an empty list for NotMentioned. Read exceptions and carve-outs carefully: a clause with an exception may contradict a hypothesis stated absolutely.

CONTRACT (numbered spans):
{spans}

CHECKLIST:
{checklist}

Respond with JSON only, no other text:
{{"nda-1": {{"label": "Entailment|Contradiction|NotMentioned", "evidence": [span numbers]}}, ...}}
Include an entry for every hypothesis key listed above."""


def build_prompt(doc, labels):
    spans = "\n".join(f"[{i}] {t}" for i, t in doc_spans(doc))
    checklist = "\n".join(f'{k}: {v["hypothesis"]} ({v["short_description"]})'
                          for k, v in labels.items())
    return INSTRUCTION.format(n=len(labels), spans=spans, checklist=checklist)


# ------------------------------------------------------- 3. read the response

def robust_parse(text):
    """Extract the JSON object from a model response.

    Returns (obj_or_None, ok). Small models sometimes close the object twice
    ('...}}\\n}'), so we walk the braces and cut where the object first closes.
    That is a formatting slip, not a reasoning error, and repairing it keeps the
    metric measuring contract review rather than brace matching.
    """
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()

    start = t.find("{")
    if start == -1:
        return None, False

    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(t[start:], start):
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
                    obj = json.loads(t[start:i + 1])
                    return (obj, True) if isinstance(obj, dict) else (None, False)
                except json.JSONDecodeError:
                    return None, False
    return None, False


# --------------------------------------------------- 4. score one contract

def score_doc(doc, pred, checklist_keys):
    """Compare a prediction against gold. Returns one row per checklist item."""
    gold = gold_for(doc)
    rows = []
    for key in checklist_keys:
        if key not in gold:
            continue
        answer = (pred or {}).get(key) or {}
        label = answer.get("label") if isinstance(answer, dict) else answer
        if isinstance(label, str):                      # tolerate case/spacing
            label = next((L for L in LABELS
                          if label.strip().lower().replace(" ", "") == L.lower()),
                         label.strip())
        cited = answer.get("evidence") if isinstance(answer, dict) else None
        cited = {int(x) for x in cited if str(x).strip().lstrip("-").isdigit()} \
            if isinstance(cited, list) else set()
        rows.append({"key": key, "gold": gold[key]["choice"], "pred": label,
                     "gold_ev": set(gold[key]["spans"]), "pred_ev": cited})
    return rows


# --------------------------------------------------------- 5. roll up metrics

def _f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return round(100 * f1, 2), round(100 * precision, 2), round(100 * recall, 2)


def aggregate(rows, json_failures=0):
    """Label accuracy, macro-F1, per-class F1, and evidence-F1."""
    n = len(rows) or 1

    counts = defaultdict(Counter)
    for r in rows:
        if r["pred"] == r["gold"]:
            counts[r["gold"]]["tp"] += 1
        else:
            counts[r["gold"]]["fn"] += 1
            if r["pred"] in LABELS:
                counts[r["pred"]]["fp"] += 1
    per_class = {L: _f1(counts[L]["tp"], counts[L]["fp"], counts[L]["fn"])[0]
                 for L in LABELS}

    # evidence: micro-averaged over every cited/expected span
    tp = sum(len(r["gold_ev"] & r["pred_ev"]) for r in rows if r["gold_ev"])
    fp = sum(len(r["pred_ev"] - r["gold_ev"]) for r in rows if r["gold_ev"])
    fn = sum(len(r["gold_ev"] - r["pred_ev"]) for r in rows if r["gold_ev"])
    ev_f1, ev_precision, ev_recall = _f1(tp, fp, fn)

    return {
        "n": len(rows),
        "label_accuracy": round(100 * sum(r["pred"] == r["gold"] for r in rows) / n, 2),
        "macro_f1": round(sum(per_class.values()) / len(per_class), 2),
        "f1_per_class": per_class,
        "evidence_f1": ev_f1,
        "evidence_precision": ev_precision,
        "evidence_recall": ev_recall,
        "json_failures": json_failures,
    }


# ------------------------------------------------------------- 6. run a model

# Applied to every frontier call so no model gets a different instruction.
SYSTEM_PROMPT = "You are a careful assistant that returns strictly valid JSON."
_RETRYABLE = ("Throttl", "429", "503", "ServiceUnavailable", "Timeout", "timeout",
              "TooManyRequests", "Connection")


def ask_bedrock(client, model_id, prompt, max_tokens=4000, attempts=6):
    """One Bedrock Converse call, retrying only on transient errors.

    Only frontier models are called from this notebook. The base and fine-tuned Qwen
    are scored by the managed evaluation pipeline instead, so there is no endpoint
    plumbing here.
    """
    delay, last, send_temperature = 2.0, None, True
    for _ in range(attempts):
        config = {"maxTokens": max_tokens}
        if send_temperature:
            config["temperature"] = 0.0
        try:
            response = client.converse(
                modelId=model_id,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig=config)
            return "".join(part.get("text", "")
                           for part in response["output"]["message"]["content"])
        except Exception as error:
            last = error
            # Some newer models reject `temperature` outright. Drop it and retry
            # rather than failing the whole evaluation.
            if "temperature" in str(error).lower():
                send_temperature = False
                continue
            if not any(k in str(error) for k in _RETRYABLE):
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError(f"giving up after {attempts} attempts: {last}")


def run(model_id, docs, labels, region="us-east-1", workers=4, max_tokens=4000):
    """Review every contract with one Bedrock model. Returns (metrics, rows)."""
    client = boto3.client("bedrock-runtime", region_name=region,
                          config=Config(read_timeout=300,
                                        retries={"total_max_attempts": 3}))
    checklist_keys = list(labels.keys())
    results, failures = [None] * len(docs), 0

    def review(i, doc):
        text = ask_bedrock(client, model_id, build_prompt(doc, labels), max_tokens)
        pred, ok = robust_parse(text)
        return i, (pred if ok else None), ok

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(review, i, d) for i, d in enumerate(docs)]
        for done, future in enumerate(as_completed(futures), 1):
            i, pred, ok = future.result()
            results[i] = score_doc(docs[i], pred, checklist_keys)
            failures += not ok
            if done % 10 == 0:
                print(f"  {model_id}: {done}/{len(docs)}")

    rows = [row for doc_rows in results for row in doc_rows]
    return aggregate(rows, failures), rows

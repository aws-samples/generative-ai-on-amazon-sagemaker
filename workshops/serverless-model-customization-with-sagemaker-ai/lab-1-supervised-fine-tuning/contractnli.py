"""
ContractNLI data access and prompt construction.

This module deliberately contains only the boring parts: downloading the dataset,
loading the splits, turning a document into numbered spans, and rendering the prompt.

Everything that constitutes the *lesson* — calling a model, parsing its answer and
scoring it — lives in the notebooks, where you can read it.

  1. the data     ensure_dataset, load, doc_spans, gold_for
  2. the prompt   INSTRUCTION, build_prompt
"""

import io
import json
import os
import pathlib
import re
import urllib.request
import zipfile

LABELS = ["Entailment", "Contradiction", "NotMentioned"]
DATA = os.environ.get("CONTRACTNLI_DIR", "./data/contract-nli")
DATASET_URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"


# ---------------------------------------------------------------- 1. the data

def ensure_dataset(target="./data"):
    """Download and unpack ContractNLI once. Released under CC-BY-4.0."""
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
    """Render the prompt for one contract. Identical at training and inference time."""
    spans = "\n".join(f"[{i}] {t}" for i, t in doc_spans(doc))
    checklist = "\n".join(f'{k}: {v["hypothesis"]} ({v["short_description"]})'
                          for k, v in labels.items())
    return INSTRUCTION.format(n=len(labels), spans=spans, checklist=checklist)

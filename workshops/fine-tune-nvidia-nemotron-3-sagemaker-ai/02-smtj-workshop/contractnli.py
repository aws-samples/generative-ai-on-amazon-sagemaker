"""
ContractNLI data access and prompt construction.

This module deliberately contains only the boring parts: downloading the dataset,
loading the splits, turning a document into numbered spans, and rendering the prompt.

Everything that constitutes the *lesson* — calling a model, parsing its answer and
scoring it — lives in the notebooks, where you can read it.

  1. the data       ensure_dataset, load, doc_spans, gold_for
  2. chat turns     SYSTEM, USER, build_system, build_user, build_messages
  3. inspection     INSTRUCTION, build_prompt

Plus CHAT_TEMPLATE_KWARGS, the switch that turns reasoning off.

`build_messages` is the one rendering that matters. Every caller uses it:

  training      notebook 1 writes `prompt` (system + user turns) and `completion`
                (the assistant turn) into each record, and TRL applies the model's own
                chat template to them inside the training job.
  serving       notebook 3 sends the same turns to the vLLM endpoint, which applies the
                same chat template.
  baseline      notebook 4 sends the same turns to Bedrock Converse.

So the string the model trains on and the string it is asked at inference time are
produced by one function and rendered by one template. There is no train/serve skew to
reason about, which is the main thing the messages format buys here.

`build_prompt` renders the same content as a single flat string. Nothing trains or
serves through it — it exists so notebook 1 can print the whole request and measure its
length in one piece.

TURNING REASONING OFF. Nemotron 3.5 Bolt E4B is a reasoning model: asked to deliberate
over a 17-item checklist with reasoning on, it spends its generation budget in a thought
channel before the JSON. The switch is NOT a magic string in the prompt — this model's
(Gemma 4) chat template takes an `enable_thinking` flag:

    {%- set enable_thinking = enable_thinking | default(false) -%}
    ...
    {%- if enable_thinking or tools or (messages and messages[0]['role'] in ['system', ...]) -%}
        {{- '<|turn>system\\n' -}}
        {%- if enable_thinking -%}
            {{- '<|think|>\\n' -}}

With the flag true the template injects a `<|think|>` token at the top of the system
turn and the model opens a `<|channel>thought ... <channel|>` block before answering.
With it false there is no `<|think|>` token and the model answers directly. The template
already defaults to false, but the flag is passed explicitly anyway so the record, the
request and this module all state the same thing. It reaches the template three ways,
all carrying the same dict: as a per-record `chat_template_kwargs` column that TRL
forwards (training), as `chat_template_kwargs` in the request body (vLLM), and not at
all for Bedrock, whose models do not share this template.
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


# Reasoning models can spend their whole generation budget reasoning on a 17-item
# checklist and be cut off before the JSON. This model's chat template takes a flag
# rather than a magic string in the prompt — see the module docstring for the three
# places it is passed.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


# -------------------------------------------------------------- Single-string form (inspection only)

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
    """Render the whole request for one contract as one flat string.

    Nothing trains or serves through this. It is here so notebook 1 can print a complete
    request and measure its length in one piece — the same content `build_messages`
    splits across two turns, in the order a person would read it.
    """
    spans = "\n".join(f"[{i}] {t}" for i, t in doc_spans(doc))
    checklist = "\n".join(
        f'{k}: {v["hypothesis"]} ({v["short_description"]})' for k, v in labels.items()
    )
    return INSTRUCTION.format(n=len(labels), spans=spans, checklist=checklist)


# -------------------------------------------------------------- Messages format

SYSTEM = """You are a contract review assistant. You review a non-disclosure agreement (NDA) against a fixed checklist of {n} legal hypotheses.

For EACH hypothesis, decide:
- "Entailment": the contract states or implies the hypothesis is true.
- "Contradiction": the contract states something that conflicts with the hypothesis.
- "NotMentioned": the contract does not address it.

Also cite the span numbers that justify the decision (the exact spans a lawyer would point to). Cite spans only for Entailment or Contradiction; use an empty list for NotMentioned. Read exceptions and carve-outs carefully: a clause with an exception may contradict a hypothesis stated absolutely.

CHECKLIST:
{checklist}

Respond with JSON only, no other text:
{{"nda-1": {{"label": "Entailment|Contradiction|NotMentioned", "evidence": [span numbers]}}, ...}}
Include an entry for every hypothesis key listed above."""

USER = """CONTRACT (numbered spans):
{spans}"""


def build_system(labels):
    """The standing instruction. Same string for every contract."""
    checklist = "\n".join(
        f'{k}: {v["hypothesis"]} ({v["short_description"]})' for k, v in labels.items()
    )
    return SYSTEM.format(n=len(labels), checklist=checklist)


def build_user(doc):
    """The one contract under review, as numbered spans."""
    return USER.format(spans="\n".join(f"[{i}] {t}" for i, t in doc_spans(doc)))


def build_messages(doc, labels, completion=None):
    """The chat turns for one contract. The one rendering every caller uses.

    Omit `completion` for an inference request (system + user) — what notebook 3 sends to
    the endpoint and notebook 4 sends to Bedrock. Pass it to get the assistant turn
    appended, which is how notebook 1 builds the `completion` half of a training record.

    Because training, serving and the baseline all go through here, the turns cannot
    drift between them: the same two dicts are rendered by the same chat template at
    training time (by TRL, inside the job) and at inference time (by vLLM, in the
    container). Reword the instruction here and every caller picks it up.
    """
    messages = [
        {"role": "system", "content": build_system(labels)},
        {"role": "user", "content": build_user(doc)},
    ]
    if completion is not None:
        messages.append({"role": "assistant", "content": completion})
    return messages

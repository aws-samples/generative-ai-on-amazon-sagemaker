"""
ContractNLI data access and prompt construction.

This module deliberately contains only the boring parts: downloading the dataset,
loading the splits, turning a document into numbered spans, and rendering the prompt.

Everything that constitutes the *lesson* — calling a model, parsing its answer and
scoring it — lives in the notebooks, where you can read it.

  1. the data       ensure_dataset, load, doc_spans, gold_for
  2. one string     INSTRUCTION, build_prompt
  3. chat turns     SYSTEM, USER, build_system, build_user, build_messages

Plus NO_THINK, the switch both formats append.

The prompt is rendered two ways because the callers need two shapes, and they are
deliberately not interchangeable:

  build_prompt    one string, contract then checklist. Notebook 1 writes it into every
                  training record (`prompt`/`completion`) and every test record
                  (`query`/`response`), so the trained prompt has exactly one source.

  build_messages  system + user turns, checklist in the system turn and the contract in
                  the user turn. For APIs that take roles rather than a single string:
                  the Bedrock Converse baseline in notebook 3, and serving in 4 and 4a.

Same instruction, same checklist, same `/no_think` — but a different order, so the two
are not byte-identical and neither is a drop-in for the other. Change the wording in
both or in neither.
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


# Reasoning models spend their whole generation budget inside <think> on a 17-item
# checklist and can be cut off before the JSON. Both formats append this switch.
NO_THINK = "/no_think"


# -------------------------------------------------------------- Chat completion format

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


def build_prompt(doc, labels, no_think=True):
    """Render the single-string prompt for one contract.

    The only prompt notebook 1 stores, for all three splits, so what the model trains on
    and what it is evaluated on are the same string. Note the order: contract first, then
    checklist, so the last thing the model reads before answering is the question. The
    two-turn `build_messages` below orders them the other way round.
    """
    spans = "\n".join(f"[{i}] {t}" for i, t in doc_spans(doc))
    checklist = "\n".join(
        f'{k}: {v["hypothesis"]} ({v["short_description"]})' for k, v in labels.items()
    )
    body = INSTRUCTION.format(n=len(labels), spans=spans, checklist=checklist)
    return f"{body}\n\n{NO_THINK}" if no_think else body


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


def build_user(doc, no_think=True):
    """The one contract under review, as numbered spans."""
    body = USER.format(spans="\n".join(f"[{i}] {t}" for i, t in doc_spans(doc)))
    return f"{body}\n\n{NO_THINK}" if no_think else body


def build_messages(doc, labels, completion=None, no_think=True):
    """The chat turns for one contract, for callers that take roles rather than a string.

    Pass `completion` to get a full training record (system + user + assistant); omit it
    to get an inference request (system + user). Every caller that needs turns builds
    them here, so the turns cannot drift between them.

    Not the same string as `build_prompt`: the checklist goes in the system turn, ahead
    of the contract. Notebook 1 trains on `build_prompt`, so a model served through here
    is being asked in a different order than it was trained in — which is fine for an
    instruction this explicit, and is what the serving checks in notebooks 4 and 4a
    verify rather than assume.
    """
    messages = [
        {"role": "system", "content": build_system(labels)},
        {"role": "user", "content": build_user(doc, no_think=no_think)},
    ]
    if completion is not None:
        messages.append({"role": "assistant", "content": completion})
    return messages

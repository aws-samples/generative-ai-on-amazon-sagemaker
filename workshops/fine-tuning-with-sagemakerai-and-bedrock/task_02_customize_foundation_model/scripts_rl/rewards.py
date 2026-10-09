"""Reward signals for the RL pass.

The ROUGE benchmark from Lab 1 measures n-gram overlap with a reference answer.
It cannot tell whether an answer is clinically safe, whether the reasoning
actually supports the conclusion, or whether the model padded its output.
This module scores those dimensions in two ways:

* RLVR (verifiable): deterministic checks that anyone can re-run and audit.
  Grounding recall of the reference answer's key terms, answer structure,
  and penalties for over-length and degenerate repetition.
* RLAIF (AI feedback): an Amazon Bedrock judge model scores correctness,
  safety and reasoning quality against the reference with a fixed rubric.

``hybrid`` mode blends both so that neither signal can be gamed on its own.
Only the standard library is needed for RLVR; boto3 is needed for RLAIF.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had has
    have having he her here hers herself him himself his how i if in into is it its itself just let me
    more most my myself no nor not now of off on once only or other our ours ourselves out over own same
    she should so some such than that the their theirs them themselves then there these they this those
    through to too under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves patient patients likely condition case given based may
    might however therefore thus diagnosis treatment answer question response""".split()
)


def default_judge_model_ids(region: str) -> Dict[str, str]:
    """Return default Bedrock model ids for the training judge and the independent eval judge.

    The training judge (Amazon Nova Pro) produces the reward the policy optimizes.
    The eval judge comes from a different model family (Anthropic Claude) so that
    an RL model that learned to please the training judge cannot also grade itself.
    """
    geo = "us"
    if region.startswith("eu-"):
        geo = "eu"
    elif region.startswith("ap-"):
        geo = "apac"
    return {
        "train_judge": f"{geo}.amazon.nova-pro-v1:0",
        # Claude Sonnet 4 (20250514) is marked Legacy and is blocked for accounts that have not used it recently.
        "eval_judge": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    }


@dataclass
class RewardConfig:
    """Configuration of the reward signal. Serialized as JSON into the REWARD_CONFIG env var."""

    mode: str = "hybrid"  # "rlvr", "rlaif" or "hybrid"
    judge_model_id: str = ""  # Bedrock model or inference profile id for RLAIF
    judge_weight: float = 0.7  # hybrid only: weight of the judge score
    region: str = ""  # Bedrock region; empty means the boto3 default
    grounding_weight: float = 0.7  # RLVR: weight of key-term grounding
    structure_weight: float = 0.3  # RLVR: weight of reasoning-then-answer structure
    max_words: int = 450  # RLVR: answers longer than this are penalized
    min_reasoning_words: int = 40  # RLVR: minimum words of reasoning before the answer
    judge_max_chars: int = 6000  # completion is truncated before it is sent to the judge

    @classmethod
    def from_json(cls, text: Optional[str]) -> "RewardConfig":
        if not text:
            return cls()
        data = json.loads(text)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        if cfg.mode not in ("rlvr", "rlaif", "hybrid"):
            raise ValueError(f"Unknown reward mode {cfg.mode!r}; use rlvr, rlaif or hybrid")
        if cfg.mode != "rlvr" and not cfg.judge_model_id:
            raise ValueError(f"Reward mode {cfg.mode!r} needs judge_model_id")
        return cfg

    def to_json(self) -> str:
        return json.dumps(asdict(self))


# ---------------------------------------------------------------------------
# RLVR: verifiable reward
# ---------------------------------------------------------------------------

def content_terms(text: str) -> List[str]:
    """Lower-cased content words (length > 2, not a stopword)."""
    return [t for t in re.findall(r"[a-z][a-z0-9\-]+", text.lower()) if len(t) > 2 and t not in STOPWORDS]


def split_reasoning_and_answer(text: str):
    """Split a completion into (reasoning, final answer).

    The SFT data is formatted as ``<chain of thought>\\n\\n<final response>``, so the
    last paragraph is treated as the answer. An explicit "final answer" marker wins.
    """
    text = text.strip()
    marker = re.search(r"(?im)^\s*(\*\*)?\s*final answer\s*(\*\*)?\s*:?", text)
    if marker:
        return text[: marker.start()].strip(), text[marker.end():].strip()
    parts = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) < 2:
        return "", text
    return "\n\n".join(parts[:-1]).strip(), parts[-1].strip()


def distinct_ngram_ratio(words: List[str], n: int = 4) -> float:
    """Share of distinct n-grams. Degenerate, looping text scores close to 0."""
    if len(words) < n + 1:
        return 1.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def verifiable_reward(completion: str, reference_answer: str, cfg: RewardConfig) -> Dict[str, float]:
    """Deterministic reward in [0, 1] plus its components."""
    words = completion.split()
    n_words = len(words)
    reasoning, answer = split_reasoning_and_answer(completion)

    ref_terms = set(content_terms(reference_answer))
    out_terms = set(content_terms(completion))
    grounding = len(ref_terms & out_terms) / len(ref_terms) if ref_terms else 0.0

    has_reasoning = len(reasoning.split()) >= cfg.min_reasoning_words
    has_answer = len(answer.split()) >= 5
    structure = (0.5 if has_reasoning else 0.0) + (0.5 if has_answer else 0.0)

    if n_words <= cfg.max_words:
        length_factor = 1.0
    else:
        length_factor = max(0.0, 1.0 - (n_words - cfg.max_words) / cfg.max_words)
    repetition_factor = min(1.0, distinct_ngram_ratio([w.lower() for w in words]) / 0.9)

    base = cfg.grounding_weight * grounding + cfg.structure_weight * structure
    score = base * length_factor * repetition_factor if n_words else 0.0
    return {
        "score": round(score, 6),
        "grounding": round(grounding, 6),
        "structure": structure,
        "length_factor": round(length_factor, 6),
        "repetition_factor": round(repetition_factor, 6),
        "words": n_words,
    }


# ---------------------------------------------------------------------------
# RLAIF: Bedrock judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a board-certified physician grading answers written by a medical AI assistant. "
    "You grade strictly and consistently, and you reply with JSON only."
)

JUDGE_TEMPLATE = """Grade the ASSISTANT ANSWER to the QUESTION. Use the REFERENCE ANSWER as the ground truth for the correct conclusion.

Score each criterion from 1 (poor) to 5 (excellent):
- correctness: does the final conclusion match the reference answer's diagnosis or recommendation?
- safety: does the answer avoid harmful or overconfident advice, mention red flags or the need for clinical confirmation when appropriate, and avoid inventing drug doses?
- reasoning: does the step-by-step reasoning actually support the conclusion, without contradictions or filler?

Do not reward length. A concise, correct and safe answer deserves 5s. Padding, repetition or restating the question lowers reasoning.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

ASSISTANT ANSWER:
{completion}

Reply with exactly one JSON object and nothing else:
{{"correctness": <1-5>, "safety": <1-5>, "reasoning": <1-5>, "rationale": "<one sentence>"}}"""

JUDGE_WEIGHTS = {"correctness": 0.5, "safety": 0.3, "reasoning": 0.2}


def parse_judge_json(text: str) -> Optional[Dict[str, float]]:
    """Extract the rubric scores from the judge's reply. Returns None if unparseable."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        scores = {k: float(data[k]) for k in JUDGE_WEIGHTS}
    except (ValueError, KeyError, TypeError):
        return None
    if not all(1.0 <= v <= 5.0 for v in scores.values()):
        return None
    weighted = sum(JUDGE_WEIGHTS[k] * scores[k] for k in JUDGE_WEIGHTS)
    scores["score"] = round((weighted - 1.0) / 4.0, 6)  # map 1..5 to 0..1
    scores["rationale"] = str(data.get("rationale", ""))[:300]
    return scores


class BedrockJudge:
    """Scores one completion with the Amazon Bedrock Converse API."""

    def __init__(self, model_id: str, region: str = "", max_tokens: int = 300):
        import boto3
        from botocore.config import Config

        kwargs = {"config": Config(retries={"max_attempts": 10, "mode": "adaptive"}, read_timeout=120)}
        if region:
            kwargs["region_name"] = region
        self.client = boto3.client("bedrock-runtime", **kwargs)
        self.model_id = model_id
        self.max_tokens = max_tokens

    def score(self, question: str, completion: str, reference: str, max_chars: int = 6000) -> Optional[Dict[str, float]]:
        prompt = JUDGE_TEMPLATE.format(
            question=question.strip(), reference=reference.strip(), completion=completion.strip()[:max_chars]
        )
        try:
            response = self.client.converse(
                modelId=self.model_id,
                system=[{"text": JUDGE_SYSTEM}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.0},
            )
            text = response["output"]["message"]["content"][0]["text"]
        except Exception as exc:  # a single failed call must not stop a batch
            print(f"[judge] {self.model_id} call failed: {exc}")
            return None
        return parse_judge_json(text)


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------

@dataclass
class RewardInput:
    question: str
    completion: str
    reference: str  # full reference (reasoning + answer), shown to the judge
    reference_answer: str  # final answer only, used for grounding
    extra: dict = field(default_factory=dict)


def score_one(item: RewardInput, cfg: RewardConfig, judge: Optional[BedrockJudge]) -> Dict:
    rlvr = verifiable_reward(item.completion, item.reference_answer, cfg)
    result = {"rlvr": rlvr["score"], "rlvr_detail": rlvr, "judge": None, "judge_detail": None}
    if cfg.mode in ("rlaif", "hybrid") and judge is not None:
        detail = judge.score(item.question, item.completion, item.reference, cfg.judge_max_chars)
        result["judge_detail"] = detail
        result["judge"] = detail["score"] if detail else None

    if cfg.mode == "rlvr":
        result["reward"] = result["rlvr"]
    elif cfg.mode == "rlaif":
        result["reward"] = result["judge"]
    else:
        result["reward"] = (
            None if result["judge"] is None
            else round(cfg.judge_weight * result["judge"] + (1 - cfg.judge_weight) * result["rlvr"], 6)
        )
    return result


def score_many(items: List[RewardInput], cfg: RewardConfig, max_workers: int = 4) -> List[Dict]:
    """Score many completions. Judge calls run in a small thread pool."""
    judge = BedrockJudge(cfg.judge_model_id, cfg.region) if cfg.mode in ("rlaif", "hybrid") else None
    if judge is None or max_workers <= 1:
        return [score_one(it, cfg, judge) for it in items]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda it: score_one(it, cfg, judge), items))

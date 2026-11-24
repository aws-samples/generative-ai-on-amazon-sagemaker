"""Make LLMAsJudgeEvaluator's inference steps use our generation settings.

sagemaker-train 1.27.0's LLM-as-a-Judge pipeline templates pass only {name, task} to the
inference (BenchmarkEvaluation) jobs, so service defaults apply: max_new_tokens=2048 and greedy
decoding (temperature 0). Both hurt a thinking model: answers get truncated, and greedy decoding
makes Qwen's thinking mode loop (Qwen recommends temperature 0.6, top_p 0.95 - which is also how
we serve it). The SDK's benchmark templates already pass these hyperparameters to the same job
type, so we add them to the LLMAJ templates' inference steps.

Call `apply(...)` before constructing LLMAsJudgeEvaluator. Verify in the inference job's
CloudWatch log: "Provided recipe config ... 'inference': {... 'max_new_tokens': 4096,
'temperature': 0.6, ... 'top_p': 0.95}".
"""
from __future__ import annotations

_MARK = '"task": "inference_only"'
_TEMPLATES = ("LLMAJ_TEMPLATE", "LLMAJ_TEMPLATE_BASE_MODEL_ONLY", "LLMAJ_INSPECTAI_TEMPLATE")
_ORIGINALS: dict[str, str] = {}


def apply(max_new_tokens: int, temperature: float | None = None, top_p: float | None = None) -> int:
    """(Re)patch the LLMAJ templates in place from the pristine originals. Returns #steps patched."""
    from sagemaker.train.evaluate import pipeline_templates as pt

    extra = f', "max_new_tokens": "{max_new_tokens}"'
    if temperature is not None:
        extra += f', "temperature": "{temperature}"'
    if top_p is not None:
        extra += f', "top_p": "{top_p}"'
    patched = 0
    for name in _TEMPLATES:
        tpl = _ORIGINALS.setdefault(name, getattr(pt, name, "") or "")
        if _MARK not in tpl:
            continue
        patched += tpl.count(_MARK)
        setattr(pt, name, tpl.replace(_MARK, _MARK + extra))
    return patched


def apply_default() -> int:
    from .config import EVAL_MAX_NEW_TOKENS, EVAL_TEMPERATURE, EVAL_TOP_P
    return apply(EVAL_MAX_NEW_TOKENS, EVAL_TEMPERATURE, EVAL_TOP_P)

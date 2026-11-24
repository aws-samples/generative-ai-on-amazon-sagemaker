"""Thin Bedrock Converse wrapper: retries, parallel map, robust JSON extraction, reasoning capture."""
from __future__ import annotations

import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import boto3
from botocore.config import Config

_local = threading.local()


def _client(region: str | None = None):
    key = f"br_{region}"
    c = getattr(_local, key, None)
    if c is None:
        c = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(retries={"max_attempts": 2, "mode": "standard"},
                          read_timeout=300, connect_timeout=10, max_pool_connections=64),
        )
        setattr(_local, key, c)
    return c


@dataclass
class LLMResult:
    text: str
    reasoning: str
    input_tokens: int
    output_tokens: int
    stop_reason: str


RETRYABLE = ("ThrottlingException", "ServiceUnavailableException", "ModelErrorException",
             "InternalServerException", "ModelTimeoutException", "ReadTimeoutError",
             "EndpointConnectionError", "ConnectionClosedError")


def converse(model_id: str, user: str, system: str | None = None, *, max_tokens: int = 2048,
             temperature: float = 0.7, top_p: float | None = None,
             reasoning_effort: str | None = None, region: str | None = None,
             retries: int = 6) -> LLMResult:
    """One Converse call with exponential backoff on throttling/transient errors."""
    args: dict[str, Any] = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "inferenceConfig": {"maxTokens": max_tokens},
    }
    if temperature is not None:          # some newer models reject `temperature`
        args["inferenceConfig"]["temperature"] = temperature
    if top_p is not None:
        args["inferenceConfig"]["topP"] = top_p
    if system:
        args["system"] = [{"text": system}]
    if reasoning_effort:
        args["additionalModelRequestFields"] = {"reasoning_effort": reasoning_effort}

    delay = 1.0
    for attempt in range(retries + 1):
        try:
            r = _client(region).converse(**args)
            text, reasoning = [], []
            for c in r["output"]["message"]["content"]:
                if "text" in c:
                    text.append(c["text"])
                rc = c.get("reasoningContent", {}).get("reasoningText", {}).get("text")
                if rc:
                    reasoning.append(rc)
            u = r.get("usage", {})
            return LLMResult("".join(text).strip(), "\n".join(reasoning).strip(),
                             u.get("inputTokens", 0), u.get("outputTokens", 0),
                             r.get("stopReason", ""))
        except Exception as e:  # noqa: BLE001
            name = type(e).__name__ + " " + str(e)
            if attempt >= retries or not any(k in name for k in RETRYABLE):
                raise
            time.sleep(delay + random.random())
            delay = min(delay * 2, 30)
    raise RuntimeError("unreachable")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Parse JSON from model text: strips ``` fences, else takes the outermost {...} or [...]."""
    if text is None:
        raise ValueError("empty response")
    m = _FENCE.search(text)
    cand = m.group(1) if m else text
    cand = cand.strip()
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (cand.find("{"), cand.find("[")) if i >= 0]
    if not starts:
        raise ValueError(f"no JSON found in: {text[:200]!r}")
    start = min(starts)
    closer = "}" if cand[start] == "{" else "]"
    end = cand.rfind(closer)
    return json.loads(cand[start:end + 1])


def converse_json(model_id: str, user: str, system: str | None = None, *, json_retries: int = 2,
                  **kw) -> tuple[Any, LLMResult]:
    """Converse + JSON parse, re-asking on malformed output."""
    last_err = None
    for _ in range(json_retries + 1):
        res = converse(model_id, user, system, **kw)
        try:
            return extract_json(res.text), res
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
    raise ValueError(f"model did not return valid JSON: {last_err}")


def pmap(fn: Callable[[Any], Any], items: Iterable[Any], *, workers: int = 32,
         desc: str = "", on_result: Callable[[Any, Any], None] | None = None,
         show_progress: bool = True) -> list[tuple[Any, Any, Exception | None]]:
    """Parallel map over items. Returns (item, result, error) triples; never raises per-item."""
    items = list(items)
    out: list[tuple[Any, Any, Exception | None]] = []
    bar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=len(items), desc=desc)
        except Exception:
            bar = None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for f in as_completed(futs):
            it = futs[f]
            try:
                res, err = f.result(), None
            except Exception as e:  # noqa: BLE001
                res, err = None, e
            out.append((it, res, err))
            if on_result is not None:
                on_result(it, res if err is None else err)
            if bar is not None:
                bar.update(1)
    if bar is not None:
        bar.close()
    return out

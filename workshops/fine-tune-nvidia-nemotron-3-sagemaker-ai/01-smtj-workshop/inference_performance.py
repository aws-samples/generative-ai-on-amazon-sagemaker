"""Optional client-side vLLM streaming measurements, independent of quality scoring."""

import codecs
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor


def iter_sse(events):
    """Yield (event type, data) across arbitrary SageMaker PayloadPart boundaries."""
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    line, data, event_type, after_cr = "", [], "message", False

    def payloads():
        for event in events:
            for error in ("ModelStreamError", "InternalStreamFailure"):
                if error in event:
                    raise RuntimeError(f"{error}: {event[error]}")
            if "PayloadPart" not in event:
                raise RuntimeError(f"Unexpected stream event: {event}")
            yield decoder.decode(event["PayloadPart"]["Bytes"])
        yield decoder.decode(b"", final=True)

    for text in payloads():
        for char in text:
            if after_cr and char == "\n":
                after_cr = False
                continue
            after_cr = char == "\r"
            if char not in "\r\n":
                line += char
                continue
            if not line:
                if data or event_type == "error":
                    yield event_type, "\n".join(data)
                data, event_type = [], "message"
            elif not line.startswith(":"):
                field, _, value = line.partition(":")
                value = value[1:] if value.startswith(" ") else value
                if field == "data":
                    data.append(value)
                elif field == "event":
                    event_type = value
            line = ""
    if line or data or event_type != "message":
        raise ValueError("Truncated SSE event at end of stream")


def stream_request(runtime, endpoint_name, payload, *, clock=time.perf_counter):
    """Measure one n=1 request; tokens/s means final completion_tokens / elapsed_s."""
    if payload.get("n", 1) != 1 or payload.get("echo", False):
        raise ValueError("Benchmark requires n=1 and echo disabled")
    body = json.dumps({**payload, "stream": True, "n": 1,
                       "stream_options": {"include_usage": True}})
    started = clock()
    response = runtime.invoke_endpoint_with_response_stream(
        EndpointName=endpoint_name, ContentType="application/json", Body=body)
    stream = response["Body"]
    first_content, usage, finish_reason, done = None, None, None, False
    pieces = []
    try:
        for event_type, data in iter_sse(stream):
            if event_type == "error":
                raise RuntimeError(f"SSE error: {data}")
            if data.strip() == "[DONE]":
                done = True
                continue
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(f"vLLM stream error: {chunk['error']}")
            if done:
                raise ValueError("Received data after [DONE]")
            choices = chunk.get("choices", [])
            if choices == [] and isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            for choice in choices:
                if choice.get("index", 0) != 0:
                    raise ValueError("Expected only choice index 0")
                if choice.get("finish_reason") == "error":
                    raise RuntimeError("vLLM returned finish_reason=error")
                finish_reason = choice.get("finish_reason") or finish_reason
                content = (choice.get("delta") or {}).get("content")
                if content is not None and not isinstance(content, str):
                    raise ValueError("Expected a text content delta")
                if content:
                    if first_content is None:
                        first_content = clock()
                    pieces.append(content)
        elapsed = clock() - started
    finally:
        stream.close()
    if not done:
        raise ValueError("Stream ended without [DONE]")
    if first_content is None:
        raise ValueError("Stream ended without generated content")
    ttft = first_content - started
    if not (math.isfinite(elapsed) and math.isfinite(ttft) and 0 < ttft <= elapsed):
        raise ValueError("Invalid elapsed or first-content time")
    tokens = (usage or {}).get("completion_tokens")
    if tokens is not None and (type(tokens) is not int or tokens <= 0):
        raise ValueError("Invalid final usage.completion_tokens")
    return {"text": "".join(pieces), "elapsed_s": elapsed, "ttft_s": ttft,
            "completion_tokens": tokens, "usage": usage, "finish_reason": finish_reason,
            "output_tokens_per_s": tokens / elapsed if tokens is not None else None}


def percentile(values, percent):
    """Linear interpolation at (n - 1) * percent / 100, with no unit conversion."""
    values = sorted(values)
    if not 0 <= percent <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    if not values:
        return None
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Percentiles require finite values")
    rank = (len(values) - 1) * percent / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def summarise(samples):
    """Report metric coverage and p50/p95 in seconds or output tokens/second."""
    result = {"sample_count": len(samples)}
    for metric in ("elapsed_s", "ttft_s", "output_tokens_per_s"):
        values = [sample[metric] for sample in samples if sample[metric] is not None]
        result[metric] = {"sample_count": len(values), "p50": percentile(values, 50),
                          "p95": percentile(values, 95)}
    return result


def benchmark(runtime, endpoint_name, payloads, *, warmup_count=1, concurrency=1,
              max_request_bytes=32768, max_output_tokens=512):
    """Run 1-20 measured requests and 0-5 serial warmups, failing on stream errors."""
    payloads = list(payloads)
    if not 1 <= len(payloads) <= 20:
        raise ValueError("Use 1-20 measured requests")
    for value, lower, upper, name in (
        (warmup_count, 0, 5, "warmup_count"), (concurrency, 1, 4, "concurrency"),
        (max_request_bytes, 1, 131072, "max_request_bytes"),
        (max_output_tokens, 1, 1024, "max_output_tokens"),
    ):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    for payload in payloads:
        tokens = payload.get("max_tokens")
        if type(tokens) is not int or not 1 <= tokens <= max_output_tokens:
            raise ValueError("max_tokens exceeds the benchmark output cap")
        if payload.get("n", 1) != 1 or payload.get("echo", False):
            raise ValueError("Benchmark requires n=1 and echo disabled")
        wire = {**payload, "stream": True, "n": 1,
                "stream_options": {"include_usage": True}}
        if len(json.dumps(wire).encode("utf-8")) > max_request_bytes:
            raise ValueError("Request exceeds byte cap; choose a shorter benchmark input")
    started = time.perf_counter()
    warmup = [stream_request(runtime, endpoint_name, payloads[i % len(payloads)])
              for i in range(warmup_count)]
    warmup_wall_s = time.perf_counter() - started
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        samples = list(pool.map(lambda p: stream_request(runtime, endpoint_name, p), payloads))
    measured_wall_s = time.perf_counter() - started
    return {"samples": samples, "summary": summarise(samples), "warmup_samples": warmup,
            "warmup_summary": summarise(warmup), "warmup_wall_s": warmup_wall_s,
            "measured_wall_s": measured_wall_s, "concurrency": concurrency}


def compute_cost(seconds, hourly_usd, instance_count=1):
    """Compute allocation only, not an invoice; caller supplies the regional rate."""
    if (not math.isfinite(seconds) or seconds < 0 or not math.isfinite(hourly_usd)
            or hourly_usd < 0 or type(instance_count) is not int or instance_count < 1):
        raise ValueError("Use finite nonnegative time/rate and a positive instance count")
    return seconds / 3600 * hourly_usd * instance_count

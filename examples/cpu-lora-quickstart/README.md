# LoRA fine-tuning on CPU in a few minutes

A minimal, end-to-end SageMaker fine-tuning example that needs **no GPU**: a CPU training job fine-tunes
`Qwen/Qwen3-0.6B` with LoRA, evaluates base vs fine-tuned in the same job, and an optional CPU endpoint serves the
result. It is meant to show the mechanics (ModelTrainer, adapters, metrics, endpoint) in any account, including
when GPU capacity is unavailable.

| Path | What |
|---|---|
| `quickstart.ipynb` | the whole flow: data, training job, results, optional endpoint, cleanup |
| `scripts/booking_data.py` | synthetic task: restaurant booking request → strict JSON (generated in code, no download) |
| `scripts/train.py` | training job entry point: base eval → ~20 LoRA steps → fine-tuned eval → `adapter/` + `metrics.json` |
| `serving/inference.py` | CPU endpoint handler: base model from the Hugging Face Hub + adapter, merged at startup |

## Measured (us-east-1)

Training, three jobs in parallel on `ml.c7i.4xlarge` (about 4 minutes each, submit to completed):

| capacity | to start | training phase | billed | SageMaker-reported spot savings |
|---|---|---|---|---|
| on-demand | 78 s | 131 s | 174 s | |
| spot | 82 s | 116 s | 56 s | 65% |
| fallback (`instance_preferences`) | 81 s | 136 s | 184 s | |

All three: exact JSON match on 30 held-out requests, base 0% → fine-tuned 100% (20 steps, batch 8, LoRA r=8).

Serving, three endpoints created in parallel (in service after 3.5 min), same request twice:

| endpoint | first request | second request |
|---|---|---|
| real-time `ml.m5.xlarge` | 4.9 s | 4.5 s |
| real-time with `InstancePools` | 4.6 s | 4.5 s |
| serverless (6 GB) | 46 s (cold start) | 3.6 s |

The base model gets the same instruction and mostly extracts the right values, but not in the required form: it
wraps the JSON in a markdown code block, keeps "sunday" lowercase and writes "5:15pm". After 20 steps the output is
exactly `{"name": "Priya", "party_size": 12, "day": "Sunday", "time": "17:15"}`.

## Requirements

* A SageMaker execution role (Studio's role works) and a training quota for one of the CPU types in
  `INSTANCE_TYPES` (types without quota are dropped automatically if the role can read Service Quotas).
* Internet access from the job and the endpoint to download `Qwen/Qwen3-0.6B` (public, Apache-2.0).
* Endpoint quotas for `ml.m5.xlarge` (and the pool types `ml.m7i.xlarge`, `ml.c7i.2xlarge`), a Spot training quota
  for `ml.c7i.4xlarge`, and serverless endpoints enabled. The last cell deletes the endpoints.

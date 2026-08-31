# Option 4 — Group Relative Policy Optimization (GRPO) with veRL

Post-train `Qwen/Qwen3-4B` with GRPO on GSM8K using [veRL](https://github.com/volcengine/verl),
vLLM rollouts, and Ray, on SageMaker Training Jobs.

The other options in this solution teach the model from text we give it: a corpus, an
instruction-response pair, a chosen-and-rejected pair. This one teaches it from text the
model writes itself. For every prompt GRPO samples a group of completions, scores each
one against a checkable ground truth, and pushes the policy toward the ones that beat
their group's average. There is no reward model and no critic network, which is what
makes it practical at this size.

| Step | Notebook | Description |
| ---- | -------- | ----------- |
| 1 | `1-prepare-data.ipynb` | Map GSM8K into veRL's reinforcement-learning parquet schema and upload to S3 |
| 2 | `2-group-relative-policy-optimization.ipynb` | Run GRPO training with veRL, vLLM rollouts, and Ray on SageMaker |
| 3 | `3-deployment.ipynb` | Deploy the exported model with vLLM, using an instance-pool capacity fallback |
| 4 | `4-evaluation.ipynb` | Measure GSM8K accuracy before and after training, and confirm the endpoint agrees |

**Use case:** verifiable mathematical reasoning

## Prerequisites

Beyond the workshop-level prerequisites, this option needs two things the others do not.

### A prebuilt training container

veRL, vLLM, Ray, FSDP, and CUDA all have to agree on versions, so this option cannot use
a SageMaker Deep Learning Container. It uses an image built from veRL's own published
base image with the SageMaker `/opt/ml` contract added on top.

**For the workshop, that image is prebuilt and shared.** Set `GRPO_TRAINING_IMAGE_URI` to
the Amazon ECR URI the workshop provides before running Lab 2, or edit
`TRAINING_IMAGE_URI` in the notebook directly.

To build it yourself:

```bash
docker build -f container/Dockerfile \
  --build-arg BASE_DIGEST=$(python -c "import json;print(json.load(open('container/base-image.lock.json'))['digest'])") \
  -t verl-grpo-training .
```

Push the result to ECR in your own account and use that URI. Note the build context is
this directory, not `container/`, because the image needs both `container/` and
`scripts/`. `container/base-image.lock.json` records the base image digest and the
versions it carries; `container/verify_stack.py` runs during the build and fails it if
veRL, Ray, vLLM, or torch cannot be imported or their versions drift.

### Quota

| Purpose | Instance | Quota needed |
| ------- | -------- | ------------ |
| Training | `ml.g6e.12xlarge` | 1 |
| Serving | `ml.g6e.2xlarge` (or `4xlarge`/`8xlarge`) | 1 |

Training needs four L40S cards, so `ml.g5.12xlarge` will not do. Serving needs one.

Quota is not capacity: a request can be within quota and still fail with
`InsufficientInstanceCapacity` because the region has no instance of that size free.
Lab 3 handles that for the endpoint by listing several instance types in priority order,
and section 8.3 of that notebook explains the failure modes. For training there is no
equivalent fallback — a job that cannot get capacity sits in `Pending`, which is not
billed, until it can.

## Time and cost

| Lab | Wall clock | Notes |
| --- | ---------- | ----- |
| 1 | ~5 minutes | No GPU |
| 2 | ~2 hours | Instance acquisition, image pull, model download, 10 GRPO steps, 2 validation passes |
| 3 | ~10 minutes | Longer if the first instance pool has no capacity |
| 4 | ~10 minutes | Scales with how many questions you score |

`ml.g6e.12xlarge` is roughly $10.49/hour on demand, so Lab 2 is about $20. The endpoint
bills from the moment it reaches `InService` until it is deleted, whether or not it is
invoked. **Both Lab 3 and Lab 4 end with teardown cells. Run them.**

## What to expect

Ten GRPO steps over 1280 prompts is 10,240 sampled completions, and it moves GSM8K
accuracy substantially — runs of this configuration have gone from roughly 40% to around
76% on held-out data, a gain of more than thirty points. Two validated runs measured
39.45% → 76.56% and 44.14% → 76.17%. The baseline is itself measured by sampling, so it
moves a few points between runs; the trained figure is consistently near 76%.

That is a real measurement, and it is not a converged model. Ten steps over a sixth of
the training set was chosen so the lab finishes, not so the model stops improving. The
full split is 7473 rows, which at this batch size is 58 steps.

## Layout

```
option-4-grpo/
├── 1-prepare-data.ipynb
├── 2-group-relative-policy-optimization.ipynb
├── 3-deployment.ipynb
├── 4-evaluation.ipynb
├── requirements.txt          notebook kernel dependencies
├── container/                the training image
│   ├── Dockerfile
│   ├── base-image.lock.json  pinned base image digest and versions
│   ├── requirements-container.txt
│   └── verify_stack.py       build-time check of the GPU stack
└── scripts/                  runs inside the container
    ├── entrypoint.py         reads hyperparameters.json, orchestrates the run
    ├── start_ray.py          forms the Ray cluster
    ├── run_grpo.py           builds veRL overrides and invokes the trainer
    └── export_checkpoint.py  merges FSDP shards into a Hugging Face model
```

`requirements.txt` is for the notebook kernel only. `container/requirements-container.txt`
is the image's extension layer, installed with `--no-deps` against the veRL base image, and
must not be installed into a Studio kernel — it would try to pull veRL and its CUDA stack
onto the notebook instance.

`scripts/` is passed to the training job as `SourceCode`, so the image supplies the
environment and the notebook supplies the code. You can edit `run_grpo.py` and resubmit
without rebuilding the image.

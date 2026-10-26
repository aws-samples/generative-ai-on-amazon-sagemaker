# LoRA fine-tuning of a VLA (OpenVLA-7B) on SageMaker Training Jobs

End-to-end lab: robot images → LoRA fine-tune on a SageMaker Training Job → closed-form evaluation →
model registry → real-time endpoint.

It combines two sources:

* **Training approach:** the AWS blog post
  [Fine-tuning OpenVLA on Amazon SageMaker AI with LoRA](https://aws.amazon.com/blogs/physical-ai/fine-tuning-openvla-on-amazon-sagemaker-ai-with-lora/):
  OpenVLA-7B, BridgeData V2 scripted images, `ModelTrainer`, PyTorch 2.7 DLC, Accelerate DDP on `ml.g6e.48xlarge`.
* **Workflow and environment:** the *Serverless Model Customization with Amazon SageMaker AI* workshop,
  [Lab 1 SFT](https://catalog.us-east-1.prod.workshops.aws/workshops/548b5be9-2da8-4c93-82f7-b0b474108ab3/en-US/02-lab-1-sft).
  It reuses that lab's Studio domain/role and MLflow App, its notebook layout, model-package registration,
  base-vs-tuned evaluation and deploy/cleanup steps.

Serverless customization (`SFTTrainer`) only covers JumpStart text models, so the training step uses
`ModelTrainer` with our own script instead.

## Layout

| Path | What |
|---|---|
| `1-prepare-data.ipynb` | BridgeData V2 → pseudo-labelled 7-DoF actions → HF `DatasetDict` on S3 |
| `2-fine-tune-vla.ipynb` | `ModelTrainer` job (smoke or full), then register a model package |
| `3-evaluation.ipynb` | single-GPU eval job: base vs fine-tuned vs mean-action baseline; approve/reject the package |
| `4-deployment.ipynb` | endpoint from the approved package (PyTorch inference DLC + custom handler), invoke |
| `5-cleanup.ipynb` | delete endpoints/configs/models; optional packages and S3 |
| `config.py` | run mode, instance types, images, names, MLflow App resolver |
| `bridge_data.py` | dataset build, optical-flow pseudo-labels, OpenVLA-style action normalization |
| `scripts/` | training job source: `run_finetuning.sh` → `train_openvla_lora.py`, recipes, eval job |
| `serving/` | endpoint handler (`inference.py`) and its requirements |
| `sample_data/bridge-300/` | 300-episode Parquet sample (19 MB, CC BY 4.0) so notebook 1 runs without Hugging Face access |
| `job_utils.py` | find, follow and time training jobs; reconnect after a dropped connection or kernel restart |
| `workshop_mlflow.py` | copied unchanged from the serverless workshop |

## Prerequisites

1. **Environment:** the serverless workshop's CloudFormation stack (Studio domain, execution role,
   MLflow App), or any Studio domain whose role can run training jobs, read/write the default bucket,
   create model packages/endpoints and call the MLflow App. Run the notebooks in a JupyterLab space;
   a CPU instance is enough.
2. **Quotas** (Service Quotas → SageMaker), in the region you use. Training and eval use managed Spot
   by default (`USE_SPOT = True`):
   * `ml.g6e.xlarge for spot training job usage`: smoke run and eval job
   * `ml.g6e.12xlarge for spot training job usage`: full run (the blog used `ml.g6e.48xlarge`, which works
     too with the same recipe)
   * endpoint: an `... for endpoint usage` quota for at least one type in `config.ENDPOINT_INSTANCE_TYPES`
     (`ml.g5.2xlarge`, `ml.g6e.xlarge`, `ml.g6.2xlarge`, ...); the endpoint falls back through them in order
   * with `USE_SPOT = False`: the on-demand `... for training job usage` quotas instead
3. **No Hugging Face token** is needed: `openvla/openvla-7b` is public.

## Run

1. In `config.py`, keep `RUN_MODE = "smoke"` for the first pass. It runs 20 steps on one L40S plus the
   real adapter save, merge, registration, eval and deployment, so every integration point gets exercised.
2. Run notebooks 1 → 5.
3. Set `RUN_MODE = "full"` and rerun notebooks 2–4 (the data from notebook 1 is reused).

The full recipe (`scripts/recipes/openvla_lora.yaml`) follows OpenVLA's reference `finetune.py`:

* LoRA r=32, alpha=16, dropout 0, `all-linear` target modules (110.8 M trainable parameters, 1.45 %)
* AdamW, lr 5e-4, cosine schedule with 5 % warmup, bf16
* global batch 32 = 4 samples/GPU × GPUs × gradient accumulation (derived at startup), 10 epochs
* Supervision on the 7 action tokens + EOS only

Each run produces:

* **Metrics:** stdout `[step N] train/loss … val/l1 …` lines and the MLflow App experiment `openvla-lora-bridge`
* **Model:** `training-output/<job>/output/model/{adapter,hf_merged}/`, stored uncompressed
  (`compression_type="NONE"`) so the endpoint can load `hf_merged/` directly
* **Checkpoints:** `checkpoints/<run>/step-*` (adapter + optimizer state). A restarted or Spot-interrupted
  job resumes from the latest one. Spot jobs can wait in `Starting` for capacity for up to `SPOT_MAX_WAIT_H`.

## Differences from the blog's code (and why)

1. **Action tokenization.** The blog's `ActionTokenizer` maps a bin `b` to token `len(tokenizer) - 256 + b`.
   That runs in the opposite direction from the checkpoint, and the top bin becomes `<PAD>` (id 32000).
   OpenVLA's own `predict_action` (`modeling_prismatic.py`) decodes `32000 - digitize(a)` and puts token
   29871 before the action tokens. `scripts/openvla_utils.py` uses that convention. With the blog's
   mapping, training first has to overwrite the pretrained action head.
2. **Inference decoding.** The blog's evaluation takes `argmax` over the last 7 logits of a single forward
   pass over the prompt, so it reads predictions at prompt positions. Here the 7 action tokens are generated
   greedily, as `predict_action` does.
3. **Action scale.** The blog labels actions as `flow / 100`, which gives a mean |a| of ~0.008, about one
   bin (2/255). Nearly every target lands in the "no motion" bin. Here each axis is normalized with
   OpenVLA's `bounds_q99` scheme (train-split 1st/99th percentiles → [-1, 1]). The stats are shipped with
   the model, and the endpoint also returns un-normalized actions.
4. **Merged output.** The blog saves only the adapter. Here the job writes `adapter/` and a merged bf16
   `hf_merged/`, including the remote-code `*.py` files and `action_stats.json`.
5. **Configuration actually applies.** The blog's `run_finetuning.sh` ignores the recipe and accelerate
   YAMLs (it runs a fixed 5,000 steps). Here the recipe drives the run and `EXTRA_ARGS` can override any
   flag.
6. **Checkpoints and validation.** Checkpoints go to `/opt/ml/checkpoints`, so they are synced to S3 and
   resumable (the blog wrote them under the model dir). Validation loss, action-token accuracy and L1 are
   logged during training (the blog disabled validation).
7. **Library versions.** `transformers==4.40.1`, `tokenizers==0.19.1`, `timm==0.9.10`, `peft==0.11.1`:
   the versions OpenVLA's remote code targets. The blog's transformers 4.57.6 is not used.

## What was verified, and what wasn't

These were verified locally on an Apple-silicon Mac (MPS), using the real `openvla/openvla-7b` weights
and a 6-episode dataset built by `bridge_data.py`:

* the prompt/action token layout matches the tokenizer and OpenVLA's `predict_action` convention
* training runs and the loss falls (12.8 → 0.2 within 6 steps on 8 samples)
* validation metrics are computed
* checkpoints save, and resume continues at the right step
* the adapter merges and the merged checkpoint reloads with `trust_remote_code`
* `evaluate_openvla.py` and `serving/inference.py` (`model_fn` → `predict_fn`) run on that merged checkpoint

**Not yet run on AWS:**

* the SageMaker jobs themselves: CUDA, multi-GPU DDP, the DLC's package install, MLflow App logging from
  the job role
* model-package registration with an uncompressed `ModelDataSource`
* the PyTorch inference DLC loading `/opt/ml/model/code` from that package
* run times and costs

Run the smoke mode first.

## Caveats

* The pseudo-labels (mean optical flow, one fixed instruction) only demo the pipeline; they are not real
  robot actions. For a real use case, put your robot's actions and instructions in `bridge_data.py`.
  Check `mean_action_baseline` in lab 3: a model that doesn't beat it hasn't learned beyond the label prior.
* z, rotation and gripper are constant (0) in these labels. The base model's prior for the gripper is
  "open" (+1), which is why its initial loss is high.

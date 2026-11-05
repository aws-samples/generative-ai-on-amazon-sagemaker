# LoRA fine-tuning of a VLA (OpenVLA-7B) on SageMaker Training Jobs

End-to-end lab: robot demonstrations → LoRA fine-tune on a SageMaker Training Job → closed-form evaluation →
model registry → real-time endpoint. Training, evaluation and the endpoint each use one GPU (L40S, L4 or A10G
class); notebook 2 explains the capacity options when that GPU is scarce.

The default data is **LIBERO** (simulated tabletop manipulation, 40 tasks, real 7-DoF actions), which OpenVLA was
not pretrained on, so fine-tuning has something to learn. The blog's BridgeData V2 pseudo-label setup is kept as an
option (`config.DATASET = "bridge"`).

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
| `1-prepare-data.ipynb` | LIBERO demos (or BridgeData V2 pseudo-labels) → normalized 7-DoF actions → HF `DatasetDict` on S3 |
| `2-fine-tune-vla.ipynb` | `ModelTrainer` job (smoke or full), then register a model package |
| `3-evaluation.ipynb` | single-GPU eval job: base vs fine-tuned vs mean-action baseline; approve/reject the package |
| `4-deployment.ipynb` | endpoint from the approved package (PyTorch inference DLC + custom handler), invoke |
| `5-cleanup.ipynb` | delete endpoints/configs/models; optional packages and S3 |
| `config.py` | run mode, instance types, images, names, MLflow App resolver |
| `libero_data.py` | LIBERO dataset build (default): episode split per task, OpenVLA-style action normalization |
| `bridge_data.py` | BridgeData V2 build: optical-flow pseudo-labels (the AWS blog's setup) |
| `scripts/` | training job source: `run_finetuning.sh` → `train_openvla_lora.py`, recipes, eval job |
| `serving/` | endpoint handler (`inference.py`) and its requirements |
| `sample_data/libero-120/`, `sample_data/bridge-300/` | Parquet samples (18 / 19 MB, CC BY 4.0) so notebook 1 needs no Hugging Face access |
| `job_utils.py` | capacity modes; find, follow and time training jobs; reconnect after a dropped connection |
| `artifacts.py` | register a merged checkpoint or a LoRA adapter as a model package; copy artifacts between S3 locations |
| `workshop_mlflow.py` | copied unchanged from the serverless workshop |

## Prerequisites

1. **Environment:** the serverless workshop's CloudFormation stack (Studio domain, execution role,
   MLflow App), or any Studio domain whose role can run training jobs, read/write the default bucket,
   create model packages/endpoints and call the MLflow App. Run the notebooks in a JupyterLab space;
   a CPU instance is enough.
2. **Quotas** (Service Quotas → SageMaker), in the region you use. Training and eval jobs use
   `CAPACITY = "fallback"` by default: on-demand, on the first instance type in the list that has capacity.
   SageMaker rejects a fallback list containing any type with a quota of 0; the notebooks check Service
   Quotas and drop those types first (needs `servicequotas:ListServiceQuotas`; otherwise the full list is sent).
   * `ml.g6e.xlarge for training job usage` (and ideally `2xlarge`, `4xlarge`, `8xlarge`, each its own
     capacity pool): smoke run and eval job
   * `ml.g6e.12xlarge` / `24xlarge for training job usage`: full run (falls back to the single-GPU sizes,
     slower but same recipe)
   * endpoint: an `... for endpoint usage` quota for at least one type in `config.ENDPOINT_INSTANCE_TYPES`
   * `CAPACITY = "spot"` uses the `... for spot training job usage` quota instead. Spot is cheaper but can
     queue for hours (5.5 h for `ml.g6e.xlarge` in us-west-2 in our tests), so keep it for self-paced runs.
3. **No Hugging Face token** is needed: `openvla/openvla-7b` is public.

## Run

1. In `config.py`, keep `RUN_MODE = "smoke"` for the first pass. It runs 20 steps on one L40S plus the
   real adapter save, merge, registration, eval and deployment, so every integration point gets exercised.
2. Run notebooks 1 → 5.
3. Set `RUN_MODE = "full"` and rerun notebooks 2–4 (the data from notebook 1 is reused).

The full recipe (`scripts/recipes/openvla_lora.yaml`) follows OpenVLA's reference `finetune.py`:

* LoRA r=8, alpha=8, dropout 0, on the language model's attention and MLP projections (q/k/v/o/gate/up/down;
  ~20 M trainable parameters, ~0.3 %, a ~40 MB adapter): lighter than OpenVLA's reference (r=32 on all linear
  layers)
* AdamW, lr 5e-4, cosine schedule with 5 % warmup, bf16
* global batch 32 = 4 samples/GPU × GPUs × gradient accumulation (derived at startup), 3 epochs
* Supervision on the 7 action tokens + EOS only

Each run produces:

* **Metrics:** stdout `[step N] train/loss … val/l1 …` lines and the MLflow App experiment `openvla-lora-<dataset>` (`openvla-lora-libero` by default)
* **Model:** `training-output/<job>/output/model/{adapter,hf_merged}/`, stored uncompressed
  (`compression_type="NONE"`) so the endpoint can load `hf_merged/` directly
* **Checkpoints:** `checkpoints/<run>/step-*` (adapter + optimizer state). A restarted or Spot-interrupted
  job resumes from the latest one. Spot jobs can wait in `Starting` for capacity for up to `SPOT_MAX_WAIT_H`.

## Results (LIBERO sample)

400 evenly spaced frames from 40 held-out demonstrations (one per task), actions generated by the model (greedy, 7
action tokens) and compared with the recorded action in normalized units:

| | L1, 7 axes | same 1/256 bin | gripper L1 |
|---|---|---|---|
| base `openvla/openvla-7b` (not trained on LIBERO) | 0.480 | 0.7% | 1.28 |
| mean-action baseline (always predict the average) | 0.359 | 6.5% | 1.00 |
| **fine-tuned, LoRA r=8, 3 epochs** | **0.221** | **18.6%** | **0.28** |

Trained on one RTX PRO 6000 (EC2 g7e.2xlarge Spot) in ~11 min of training + merge; same script and recipe as the
SageMaker job. 10 epochs gave 0.227: validation stops improving after ~3 epochs on 120 demonstrations, so the
`full` recipe trains 3.

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

## What was verified

* Locally (Apple-silicon Mac, real `openvla/openvla-7b` weights, a 6-episode dataset): prompt/action token layout
  vs OpenVLA's `predict_action`, training loss falling, validation metrics, checkpoint save/resume, adapter merge
  and reload with `trust_remote_code`, `evaluate_openvla.py` and `serving/inference.py` on the merged checkpoint.
* On SageMaker (us-east-1 / us-west-2, 2026-10): the smoke training job on a single L40S (~20 min end to end,
  most of it startup), model-package registration with an uncompressed
  `ModelDataSource`, the evaluation job (LIBERO: L1 0.220 fine-tuned vs 0.478 base vs 0.359 mean action,
  Approved), and the endpoint from the approved package (in service in ~8 min, ~0.45 s per request after
  warm-up).
* The `full` recipe was run with the same training script on one RTX PRO 6000 (EC2 g7e) for the Results table;
  the multi-GPU `full` job on SageMaker (DDP across 4 L40S) has not been run end to end.

Run the smoke mode first.

## Caveats

These apply to `DATASET = "bridge"` (the AWS blog's setup); the LIBERO default uses recorded robot actions.

* The pseudo-labels (mean optical flow, one fixed instruction) only demo the pipeline; they are not real
  robot actions. For a real use case, put your robot's actions and instructions in `bridge_data.py`.
  Check `mean_action_baseline` in lab 3: a model that doesn't beat it hasn't learned beyond the label prior.
  Measured on the 300-episode sample with the full recipe (1x RTX PRO 6000, 170 steps): L1 on x/y of
  0.33 fine-tuned vs 0.40 base vs 0.27 mean-action baseline. The fine-tuned model clearly beats the base model
  but not the prior, so lab 3 approves on "beats the base model" and reports the baseline as context.
* z, rotation and gripper are constant (0) in these labels. The base model's prior for the gripper is
  "open" (+1), which is why its initial loss is high.

## Licenses

* **Model:** `openvla/openvla-7b` is built on Llama 2; its weights, and adapters or merged checkpoints derived
  from them, are subject to the Llama 2 Community License (see the model card on the Hugging Face Hub). This lab
  downloads the base model from the Hub and stores your fine-tuned artifacts in your own S3 bucket; it does not
  redistribute weights.
* **Data:** the LIBERO and BridgeData V2 samples in `sample_data/` are CC BY 4.0 (see `sample_data/README.md`).

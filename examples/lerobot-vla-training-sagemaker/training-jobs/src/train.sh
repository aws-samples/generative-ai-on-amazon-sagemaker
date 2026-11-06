#!/usr/bin/env bash
# Launcher executed inside the SageMaker training container (LeRobot 0.6.1).
#
#   1. Read the SageMaker topology (hosts, current host) and derive accelerate flags.
#   2. Report whether HF_TOKEN is present (gated repos: Cosmos-Reason2-2B for groot, PaliGemma for pi05).
#   3. Run lerobot-train with the policy preset selected by --model, resuming from
#      /opt/ml/checkpoints if a previous run left a checkpoint there.
#   4. Copy the final policy to /opt/ml/model so SageMaker uploads it as model.tar.gz.
#
# Multi-GPU in LeRobot 0.6.x goes through `accelerate launch` (LeRobot detects it and
# handles DDP/FSDP itself). Default is DDP, which fits both 3B models on 80/96 GB GPUs.
# Set DISTRIBUTED=fsdp to shard instead (needs FSDP_WRAP_CLS in the preset).
# Everything model-specific lives in presets/<model>.env.
set -euo pipefail

MODEL="groot"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET="${HERE}/presets/${MODEL}.env"
[[ -f "$PRESET" ]] || { echo "Unknown model preset: ${MODEL} (expected ${PRESET})"; exit 1; }
# shellcheck disable=SC1090
source "$PRESET"

# ---------------------------------------------------------------------------
# 1. Topology from SageMaker
# ---------------------------------------------------------------------------
# Three environments, one script:
#   SageMaker Training Jobs : /opt/ml/input/config/resourceconfig.json (hosts, current_host)
#   HyperPod EKS PyTorchJob : Kubeflow injects MASTER_ADDR / RANK (pod index) / WORLD_SIZE (pod count)
#   anything else           : single node
RESOURCE_CONFIG=/opt/ml/input/config/resourceconfig.json
if [[ -f "$RESOURCE_CONFIG" ]]; then
  CURRENT_HOST=$(jq -r '.current_host' "$RESOURCE_CONFIG")
  HOSTS=$(jq -r '.hosts | sort | .[]' "$RESOURCE_CONFIG")
  MASTER_HOST=$(echo "$HOSTS" | head -n1)
  NNODES=$(echo "$HOSTS" | wc -l | tr -d ' ')
  NODE_RANK=$(echo "$HOSTS" | grep -n "^${CURRENT_HOST}$" | cut -d: -f1)
  NODE_RANK=$((NODE_RANK - 1))
  MASTER_PORT=29500
elif [[ -n "${MASTER_ADDR:-}" && -n "${WORLD_SIZE:-}" ]]; then
  CURRENT_HOST=$(hostname); MASTER_HOST="$MASTER_ADDR"; NNODES="$WORLD_SIZE"; NODE_RANK="${RANK:-0}"
  MASTER_PORT="${MASTER_PORT:-29500}"
  # These are pod-level values; accelerate will set the real per-process RANK/WORLD_SIZE.
  unset RANK WORLD_SIZE LOCAL_RANK
else
  CURRENT_HOST=localhost; MASTER_HOST=localhost; NNODES=1; NODE_RANK=0; MASTER_PORT=29500
fi
NPROC_PER_NODE=$(nvidia-smi -L | wc -l | tr -d ' ')
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

echo "=== SageMaker LeRobot training ==="
echo "model=${MODEL} host=${CURRENT_HOST} master=${MASTER_HOST} nnodes=${NNODES} node_rank=${NODE_RANK} gpus/node=${NPROC_PER_NODE} world=${WORLD_SIZE}"
nvidia-smi -L

# EFA / NCCL: harmless on single node, required for multi-node p5/g7e.48xlarge.
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_FORK_SAFE=1 NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker0}

# ---------------------------------------------------------------------------
# 2. Hugging Face token: plain HF_TOKEN env var, set by the launcher (--hf-token) or the pod spec.
# ---------------------------------------------------------------------------
[[ -n "${HF_TOKEN:-}" ]] && echo "HF_TOKEN set (len ${#HF_TOKEN})" || echo "HF_TOKEN not set (fine for ACT; GR00T/pi0.5 need it)"

# ---------------------------------------------------------------------------
# 3. Dataset: S3 channel if provided, otherwise stream from the Hub.
# ---------------------------------------------------------------------------
DATASET_REPO_ID="${DATASET_REPO_ID:-sreetz-nv/so101_teleop_vials_rack_left}"
# Local copy of the dataset: SageMaker input channel "dataset" by default, or any path via DATASET_ROOT
# (e.g. a directory on FSx on HyperPod). If absent, LeRobot downloads repo_id from the Hub into HF_LEROBOT_HOME.
DATASET_ROOT="${DATASET_ROOT:-/opt/ml/input/data/dataset}"
DATASET_ARGS=(--dataset.repo_id="${DATASET_REPO_ID}")
if [[ -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "Using local dataset at ${DATASET_ROOT}"
  DATASET_ARGS+=(--dataset.root="${DATASET_ROOT}")
fi

# ---------------------------------------------------------------------------
# 4. Checkpoints / resume. SageMaker syncs /opt/ml/checkpoints <-> S3 continuously.
# ---------------------------------------------------------------------------
CKPT_ROOT="${CKPT_ROOT:-/opt/ml/checkpoints}"
OUTPUT_DIR="${CKPT_ROOT}/${JOB_NAME:-${MODEL}}"
RESUME_ARGS=()
CKPT_DIR="${OUTPUT_DIR}/checkpoints"
if [[ -d "$CKPT_DIR" ]]; then
  # LeRobot writes checkpoints/<step>/ and a 'last' symlink. SageMaker's S3 checkpoint sync does not
  # preserve symlinks, so after a restore only the numbered dirs exist: recreate 'last' from the
  # highest step that has a complete train_config.json.
  LATEST=$(ls -1 "$CKPT_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | while read -r d; do
             [[ -f "$CKPT_DIR/$d/pretrained_model/train_config.json" ]] && echo "$d"; done | tail -n1)
  if [[ -n "$LATEST" ]]; then
    ln -sfn "$LATEST" "$CKPT_DIR/last"
    echo "Found checkpoint ${CKPT_DIR}/${LATEST} in ${OUTPUT_DIR}, resuming."
    # --resume loads train_config.json; CLI flags given afterwards (e.g. --steps) still override it.
    RESUME_ARGS=(--resume=true --config_path="${CKPT_DIR}/last/pretrained_model/train_config.json")
  fi
fi

# ---------------------------------------------------------------------------
# 5. Launcher: plain python on 1 GPU, accelerate launch otherwise.
# ---------------------------------------------------------------------------
LAUNCHER=()
if [[ $WORLD_SIZE -gt 1 ]]; then
  LAUNCHER=(accelerate launch
            --num_processes="${WORLD_SIZE}" --num_machines="${NNODES}" --machine_rank="${NODE_RANK}"
            --main_process_ip="${MASTER_HOST}" --main_process_port="${MASTER_PORT}"
            --mixed_precision="${MIXED_PRECISION:-bf16}")
  if [[ "${DISTRIBUTED:-ddp}" == "fsdp" ]]; then
    [[ -n "${FSDP_WRAP_CLS:-}" ]] || { echo "DISTRIBUTED=fsdp needs FSDP_WRAP_CLS in the preset"; exit 1; }
    # FSDP1 FULL_SHARD, per LeRobot 0.6.1 multi_gpu_training docs.
    LAUNCHER+=(--use_fsdp --fsdp_version=1 --fsdp_sharding_strategy=FULL_SHARD
               --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP
               --fsdp_transformer_layer_cls_to_wrap="${FSDP_WRAP_CLS}"
               --fsdp_use_orig_params=true --fsdp_state_dict_type=FULL_STATE_DICT)
  else
    LAUNCHER+=(--multi_gpu)
  fi
fi

# ---------------------------------------------------------------------------
# 5b. PEFT / LoRA (for 24-48 GB GPUs where a full fine-tune of a 3B VLA does not fit).
#     LeRobot 0.6.1: --peft.* (generic HF PEFT); LR ~10x the full-FT value.
# ---------------------------------------------------------------------------
PEFT_ARGS=()
if [[ "${PEFT:-0}" == "1" ]]; then
  [[ "${PEFT_SUPPORTED:-1}" == "1" ]] || { echo "Preset ${MODEL} does not support PEFT. ${PEFT_UNSUPPORTED_HINT:-}"; exit 1; }
  if [[ -n "${PEFT_ARGS_STR:-}" ]]; then
    # Preset-specific mechanism, if a preset defines one.
    # shellcheck disable=SC2206
    PEFT_ARGS=(${PEFT_ARGS_STR})
  else
    # Generic HF PEFT LoRA (pi0/pi05/smolvla/molmoact2 define default target modules).
    # NOTE: do NOT pass --policy.use_peft=true here; that flag means "load an existing adapter".
    PEFT_ARGS=(--peft.method_type=LORA --peft.r="${PEFT_R:-32}" --peft.lora_alpha="${PEFT_ALPHA:-32}")
  fi
  [[ -n "${PEFT_LR:-}" ]] && PEFT_ARGS+=(--optimizer.lr="${PEFT_LR}")
fi

# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------
# Peak-VRAM sampler (per GPU, MiB) so CloudWatch logs carry a measurable number.
GPU_MEM_LOG=/tmp/gpu_mem.csv
( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> "$GPU_MEM_LOG" 2>/dev/null || true; sleep 5; done ) &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null || true' EXIT

# shellcheck disable=SC2206
POLICY_ARGS=(${POLICY_ARGS_STR})
START_TS=$(date +%s)
set -x
"${LAUNCHER[@]}" "$(which lerobot-train)" \
    "${DATASET_ARGS[@]}" \
    "${POLICY_ARGS[@]}" \
    "${PEFT_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME:-${MODEL}}" \
    --steps="${STEPS:-20000}" \
    --batch_size="${BATCH_SIZE:-16}" \
    --save_checkpoint=true --save_freq="${SAVE_FREQ:-2500}" \
    --num_workers="${NUM_WORKERS:-8}" --prefetch_factor=2 --persistent_workers=true \
    --log_freq=10 --env_eval_freq=0 --eval_steps=0 \
    --wandb.enable="${WANDB_ENABLE:-false}" \
    --policy.push_to_hub=false \
    --seed=42 \
    "${EXTRA_ARGS[@]}"
set +x
END_TS=$(date +%s)
kill $SAMPLER_PID 2>/dev/null || true
echo "=== RESULT model=${MODEL} peft=${PEFT:-0} world=${WORLD_SIZE} batch=${BATCH_SIZE:-16} steps=${STEPS:-20000} train_wall_s=$((END_TS-START_TS))"
if [[ -s "$GPU_MEM_LOG" ]]; then
  awk -F', *' '{ if ($2+0 > m[$1]) m[$1]=$2+0 } END { for (g in m) printf "=== PEAK_VRAM gpu=%s used_mib=%d\n", g, m[g] }' "$GPU_MEM_LOG" | sort
fi

# ---------------------------------------------------------------------------
# 7. Publish the final policy as the SageMaker model artifact (rank 0 only).
# ---------------------------------------------------------------------------
if [[ "$NODE_RANK" -eq 0 && -d /opt/ml/model ]]; then
  FINAL="${OUTPUT_DIR}/checkpoints/last/pretrained_model"
  if [[ -d "$FINAL" ]]; then
    echo "Copying ${FINAL} -> /opt/ml/model"
    # SageMaker's checkpoint uploader creates transient *.sagemaker-uploading files in
    # /opt/ml/checkpoints; skip them and never let the copy fail a successful run.
    ( cd "$FINAL" && find . -type f ! -name '*.sagemaker-uploading' ! -name '*.sagemaker-uploaded' -print0 \
        | xargs -0 -I{} sh -c 'mkdir -p "/opt/ml/model/$(dirname "{}")" && cp "{}" "/opt/ml/model/{}"' ) \
      || echo "WARNING: partial copy to /opt/ml/model (checkpoint is complete in ${OUTPUT_DIR})"
    ls -la /opt/ml/model | head -20
  fi
fi
echo "=== done ==="

#!/bin/bash
# SageMaker entry point: launches train_openvla_lora.py on every GPU of the instance with Accelerate.
# ModelTrainer has already pip-installed requirements.txt. RECIPE and EXTRA_ARGS come from the job environment.
set -euo pipefail
cd "$(dirname "$0")"

NUM_GPUS=${SM_NUM_GPUS:-$(nvidia-smi -L | wc -l)}
RECIPE=${RECIPE:-recipes/openvla_lora.yaml}
MULTI_GPU=$([ "$NUM_GPUS" -gt 1 ] && echo "--multi_gpu" || true)

echo "GPUs=${NUM_GPUS} recipe=${RECIPE} extra='${EXTRA_ARGS:-}'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

accelerate launch ${MULTI_GPU} --num_processes "${NUM_GPUS}" --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no train_openvla_lora.py --config "${RECIPE}" ${EXTRA_ARGS:-}

ls -la /opt/ml/model /opt/ml/model/hf_merged 2>/dev/null || true

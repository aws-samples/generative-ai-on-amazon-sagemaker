#!/usr/bin/env bash
# Render and (optionally) apply a PyTorchJob for LeRobot VLA training on HyperPod EKS.
#
#   ./launch.sh --model groot --instance-type ml.g7e.2xlarge                            # 1 node, 1 GPU
#   ./launch.sh --model act   --instance-type ml.g6.12xlarge --gpus 4 --efa 1 --nodes 2  # 2 nodes x 4 GPUs, DDP over EFA
#
# --gpus: GPUs per node of the instance type (default 1). --efa: EFA interfaces to request per pod on
# multi-node jobs (default 0 = no EFA). Both values come from the EC2 instance type specs.
#   ./launch.sh ... --steps 300 --dry-run                                     # print YAML only
#
# Requires: kubectl context pointing at the HyperPod EKS cluster, python3, IMAGE env var.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=groot; INSTANCE_TYPE=ml.g7e.2xlarge; NODES=1; GPUS=1; EFA=0; STEPS=20000; SAVE_FREQ=2500; BATCH_SIZE=""
PEFT=0; DISTRIBUTED=ddp; NAMESPACE=default; PVC=fsx-claim; DRY=0; JOB_NAME=""; EXTRA_ARGS=""
DATASET_REPO_ID=sreetz-nv/so101_teleop_vials_rack_left; DATASET_ROOT=""; HF_TOKEN="${HF_TOKEN:-}"
IMAGE="${IMAGE:?set IMAGE to your ECR image URI, e.g. <acct>.dkr.ecr.<region>.amazonaws.com/lerobot-vla:0.6.1}"
while [[ $# -gt 0 ]]; do case "$1" in
  --model) MODEL=$2; shift 2;; --instance-type) INSTANCE_TYPE=$2; shift 2;; --nodes) NODES=$2; shift 2;; --gpus) GPUS=$2; shift 2;; --efa) EFA=$2; shift 2;;
  --steps) STEPS=$2; shift 2;; --save-freq) SAVE_FREQ=$2; shift 2;; --batch-size) BATCH_SIZE=$2; shift 2;;
  --peft) PEFT=1; shift;; --distributed) DISTRIBUTED=$2; shift 2;; --namespace) NAMESPACE=$2; shift 2;;
  --pvc) PVC=$2; shift 2;; --job-name) JOB_NAME=$2; shift 2;; --dataset) DATASET_REPO_ID=$2; shift 2;; --dataset-root) DATASET_ROOT=$2; shift 2;; --hf-token) HF_TOKEN=$2; shift 2;;
  --image) IMAGE=$2; shift 2;; --extra-args) EXTRA_ARGS=$2; shift 2;; --dry-run) DRY=1; shift;;
  *) echo "unknown arg $1"; exit 1;; esac; done

# EFA is only useful across nodes; request adapters only for multi-node jobs.
EFA_LIMIT=""; [[ $NODES -gt 1 && $EFA -gt 0 ]] && EFA_LIMIT="vpc.amazonaws.com/efa: ${EFA}"
SHM_SIZE=$(( GPUS * 16 ))Gi
WORKER_REPLICAS=$(( NODES - 1 ))
short=$(echo "$INSTANCE_TYPE" | sed 's/ml\.//; s/\./-/g')
LORA_SUFFIX=""; [[ "$PEFT" == 1 ]] && LORA_SUFFIX="-lora"
JOB_NAME="${JOB_NAME:-hp-${MODEL}${LORA_SUFFIX}-${short}-x${NODES}}"
JOB_NAME=$(echo "$JOB_NAME" | tr '[:upper:]_' '[:lower:]-' | cut -c1-63)

export EXTRA_ARGS JOB_NAME NAMESPACE INSTANCE_TYPE IMAGE MODEL DATASET_REPO_ID DATASET_ROOT HF_TOKEN STEPS SAVE_FREQ BATCH_SIZE PEFT DISTRIBUTED PVC SHM_SIZE EFA_LIMIT WORKER_REPLICAS
export GPUS_PER_NODE=$GPUS
OUT="${HERE}/generated/${JOB_NAME}.yaml"; mkdir -p "${HERE}/generated"
python3 - "${HERE}/pytorchjob.yaml.template" > "$OUT" <<'PY'
import os, re, sys
t = open(sys.argv[1]).read()
print(re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), t), end="")
PY
echo "rendered $OUT  (nodes=${NODES} gpus/node=${GPUS} efa=${EFA_LIMIT:-none})"
[[ $DRY == 1 ]] && { cat "$OUT"; exit 0; }
kubectl apply -f "$OUT"
echo "watch: kubectl get pytorchjob ${JOB_NAME} -n ${NAMESPACE}; kubectl logs -f ${JOB_NAME}-master-0 -n ${NAMESPACE}"

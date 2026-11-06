#!/usr/bin/env bash
# Render and (optionally) apply a PyTorchJob for LeRobot VLA training on HyperPod EKS.
#
#   ./launch.sh --model groot --instance-type ml.g7e.8xlarge                 # 1 node, 1 GPU
#   ./launch.sh --model act   --instance-type ml.g6e.12xlarge --nodes 2      # 2 nodes x 4 GPUs, DDP
#   ./launch.sh ... --steps 300 --dry-run                                     # print YAML only
#
# Requires: kubectl context pointing at the HyperPod EKS cluster, python3, IMAGE env var.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=groot; INSTANCE_TYPE=ml.g7e.8xlarge; NODES=1; STEPS=20000; SAVE_FREQ=2500; BATCH_SIZE=""
PEFT=0; DISTRIBUTED=ddp; NAMESPACE=default; PVC=fsx-claim; DRY=0; JOB_NAME=""; EXTRA_ARGS=""
DATASET_REPO_ID=sreetz-nv/so101_teleop_vials_rack_left; DATASET_ROOT=""; HF_TOKEN="${HF_TOKEN:-}"
IMAGE="${IMAGE:?set IMAGE to your ECR image URI, e.g. <acct>.dkr.ecr.<region>.amazonaws.com/lerobot-vla:0.6.1}"
while [[ $# -gt 0 ]]; do case "$1" in
  --model) MODEL=$2; shift 2;; --instance-type) INSTANCE_TYPE=$2; shift 2;; --nodes) NODES=$2; shift 2;;
  --steps) STEPS=$2; shift 2;; --save-freq) SAVE_FREQ=$2; shift 2;; --batch-size) BATCH_SIZE=$2; shift 2;;
  --peft) PEFT=1; shift;; --distributed) DISTRIBUTED=$2; shift 2;; --namespace) NAMESPACE=$2; shift 2;;
  --pvc) PVC=$2; shift 2;; --job-name) JOB_NAME=$2; shift 2;; --dataset) DATASET_REPO_ID=$2; shift 2;; --dataset-root) DATASET_ROOT=$2; shift 2;; --hf-token) HF_TOKEN=$2; shift 2;;
  --image) IMAGE=$2; shift 2;; --extra-args) EXTRA_ARGS=$2; shift 2;; --dry-run) DRY=1; shift;;
  *) echo "unknown arg $1"; exit 1;; esac; done

# GPUs per node and EFA interfaces per instance type, from `aws ec2 describe-instance-types`
# (GpuInfo.Gpus[0].Count, NetworkInfo.EfaInfo.MaximumEfaInterfaces). All G-family 8xlarge+ have EFA.
case "$INSTANCE_TYPE" in
  ml.g7e.2xlarge|ml.g7e.4xlarge|ml.g6e.xlarge|ml.g6e.2xlarge|ml.g6e.4xlarge|ml.g6.xlarge|ml.g6.2xlarge|ml.g6.4xlarge|ml.g5.xlarge|ml.g5.2xlarge|ml.g5.4xlarge) GPUS=1; EFA=0;;
  ml.g7e.8xlarge|ml.g6e.8xlarge|ml.g6e.16xlarge|ml.g6.8xlarge|ml.g6.16xlarge|ml.g5.8xlarge|ml.g5.16xlarge) GPUS=1; EFA=1;;
  ml.g7e.12xlarge) GPUS=2; EFA=1;;
  ml.g6e.12xlarge|ml.g6.12xlarge|ml.g6.24xlarge|ml.g5.12xlarge|ml.g5.24xlarge) GPUS=4; EFA=1;;
  ml.g7e.24xlarge|ml.g6e.24xlarge) GPUS=4; EFA=2;;
  ml.g6.48xlarge|ml.g5.48xlarge) GPUS=8; EFA=1;;
  ml.g7e.48xlarge|ml.g6e.48xlarge|ml.p4d.24xlarge|ml.p4de.24xlarge) GPUS=8; EFA=4;;
  ml.p5.48xlarge|ml.p5e.48xlarge|ml.p5en.48xlarge) GPUS=8; EFA=32;;
  *) echo "unknown instance type $INSTANCE_TYPE"; exit 1;;
esac
# EFA is only useful across nodes; request adapters only for multi-node jobs.
EFA_LIMIT=""; [[ $NODES -gt 1 && $EFA -gt 0 ]] && EFA_LIMIT="vpc.amazonaws.com/efa: ${EFA}"
SHM_SIZE=$(( GPUS * 16 ))Gi
NUM_WORKERS=$(( GPUS >= 4 ? 8 : 4 ))
WORKER_REPLICAS=$(( NODES - 1 ))
short=$(echo "$INSTANCE_TYPE" | sed 's/ml\.//; s/\./-/g')
LORA_SUFFIX=""; [[ "$PEFT" == 1 ]] && LORA_SUFFIX="-lora"
JOB_NAME="${JOB_NAME:-hp-${MODEL}${LORA_SUFFIX}-${short}-x${NODES}}"
JOB_NAME=$(echo "$JOB_NAME" | tr '[:upper:]_' '[:lower:]-' | cut -c1-63)

export EXTRA_ARGS JOB_NAME NAMESPACE INSTANCE_TYPE IMAGE MODEL DATASET_REPO_ID DATASET_ROOT HF_TOKEN STEPS SAVE_FREQ BATCH_SIZE NUM_WORKERS PEFT DISTRIBUTED PVC SHM_SIZE EFA_LIMIT WORKER_REPLICAS
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

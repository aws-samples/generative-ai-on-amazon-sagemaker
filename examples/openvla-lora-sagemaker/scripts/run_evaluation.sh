#!/bin/bash
# SageMaker entry point for the evaluation job (single GPU).
set -euo pipefail
cd "$(dirname "$0")"
python evaluate_openvla.py ${EXTRA_ARGS:-}

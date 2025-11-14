#!/bin/bash

huggingface-cli login --token $HF_token

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected ${NUM_GPUS} GPUs on the machine"


# # Install Python dependencies
# echo "Installing Python packages from requirements.txt..."
# python3 -m pip install -r ./requirements.txt

# Launch fine-tuning with Accelerate + DeepSpeed (Zero3)
accelerate launch \
  --config_file $ACCELERATE_CONFIG_PATH \
  --num_processes ${NUM_GPUS} \
  train.py \
  --config $CONFIG_PATH
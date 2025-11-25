# RIV 2025 AIM 412 || Mastering foundation model customization with Amazon SageMaker AI

## Setup

Model Choice: Qwen/Qwen3-VL-2B-Instruct or Qwen/Qwen3-VL-4B-Instruct

Dataset: [pranavvmurthy26/DoclingMatix_5K](https://huggingface.co/datasets/pranavvmurthy26/DoclingMatix_5K)

## Prep Dataset

Run `finetune--Qwen--Qwen3-VL-2B-Instruct.ipynb` or `finetune--Qwen--Qwen3-VL-4B-Instruct.ipynb` Notebook (Data prep stage).

> [NOTE]
> If you're running this inside an EC2 move the JSONL file to /opt/ml/input/data/training/HuggingFaceM4--DoclingMatix.jsonl before running ft task.


## How to Run

If you're running this inside EC2 (using capacity blocks)

```bash

sudo apt-get update -y

sudo apt-get install python3-pip

sudo pip install uv

uv venv py312 --python 3.12

source py312/bin/activate 

uv pip install riv2025-aim412-mastering-fm-fine-tuning/sagemaker_code/requirements.txt

accelerate launch --config_file configs/accelerate/ds_zero3.yaml --num_processes 1 sft.py --config hf_recipes/Qwen/Qwen3-VL-2B-Instruct-vanilla-peft-qlora.yaml

```


## Use Docker

```

aws configure

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

docker run -it --gpus all -v /home/ubuntu/pranavvm/riv2025-aim412-mastering-fm-fine-tuning/:/app 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker /bin/bash

cd /app/riv2025-aim412-mastering-fm-fine-tuning/sagemaker_code

./sm_accelerate_train.sh --config hf_recipes/Qwen/Qwen3-VL-2B-Instruct-vanilla-peft-qlora.yaml

```


## SageMaker Workflow

WIP





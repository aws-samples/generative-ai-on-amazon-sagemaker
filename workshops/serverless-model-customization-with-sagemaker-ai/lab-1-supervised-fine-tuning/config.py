# Base model for this lab.
#
# Must be a JumpStart model that (a) supports serverless customization and
# (b) can be served by the SageMaker LMI/DJL container and imported into
# Amazon Bedrock Custom Model Import.
#
# Verified end-to-end in this lab: huggingface-reasoning-qwen3-4b
#
# Known NOT to work end-to-end: huggingface-vlm-qwen3-5-4b (Qwen3.5). Training
# succeeds, but no current LMI container recognises model type `qwen3_5`, so
# notebooks 4 and 4a fail.
BASE_MODEL_ID = "huggingface-reasoning-qwen3-4b"

# Fixed dataset / resource names used across the notebooks
DATASET_PREFIX = "contractnli-nda-review"

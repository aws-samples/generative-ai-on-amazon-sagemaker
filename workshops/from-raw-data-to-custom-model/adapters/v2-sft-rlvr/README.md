---
library_name: peft
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
tags:
- lora
- sagemaker
- workshop
---
# Octank Gift Concierge LoRA adapter: SFT then RLVR (20K)

Starts from the SFT adapter and continues with reinforcement learning with verifiable rewards (RLVR, GRPO). The reward is a hybrid: hard rules checked in code plus an LLM taste score, with anti-gaming penalties (`judge/reward_lambda.py`).

This adapter is part of the workshop **From raw data to custom model: dataset prep with Amazon
SageMaker AI**. See the workshop `README.md` for the full scenario.

| | |
|---|---|
| Base model | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), thinking on (Apache 2.0) |
| Method | LoRA, rank 32, alpha 64, bf16 |
| Task | Given a shopper request and 40 to 80 candidate products, reason and return JSON picks, a short reason per pick, the total and a gift-card message |
| Training data | Synthetic only: shopper requests, catalog and teacher answers generated with `gpt-oss-120b` on Amazon Bedrock (20,000 raw requests, curated with the workshop pipeline) |
| Trained with | Amazon SageMaker AI serverless model customization |

## Use

Attach the adapter to the base model as a LoRA adapter, for example as a SageMaker inference
component on a vLLM (DJL LMI) endpoint. `serving.upload_adapter` in the workshop code does this
for Lab 4.

The tokenizer and chat template files are copied from the base model.

## Limitations

The model is trained on a fictional shop's catalog (Octank) and on synthetic requests. It is a
teaching artifact, not a production recommender. Output should be checked with the rule checks
in `judge/reward_lambda.py` before it is used for anything else.

## License

The adapter and the copied tokenizer and chat-template files follow the base model's license,
[Apache 2.0](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE).

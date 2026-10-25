"""CPU-only SFT training script for SmolLM with LoRA on Amazon SageMaker."""

from dataclasses import dataclass, field
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import glob as _glob
from huggingface_hub import snapshot_download
import json
import logging
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
import transformers
from transformers.trainer_utils import get_last_checkpoint
from trl import TrlParser
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from typing import Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """Training script arguments."""

    model_id: str = field(default=None, metadata={"help": "HuggingFace model ID"})
    train_dataset_path: Optional[str] = field(default=None)
    val_dataset_path: Optional[str] = field(default=None)
    checkpoint_dir: Optional[str] = field(default=None)
    use_checkpoints: bool = field(default=False)

    # LoRA
    use_peft: bool = field(default=True)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    target_modules: Optional[str] = field(default=None)

    # Model config
    merge_weights: bool = field(default=False)
    attn_implementation: Optional[str] = field(default="sdpa")
    torch_dtype: Optional[str] = field(default="float32")
    text_field: str = field(default="text")
    dataset_format: str = field(default="auto")
    messages_field: str = field(default="messages")
    apply_truncation: bool = field(default=False)
    max_length: Optional[int] = field(default=None)

    # Unused but accepted from yaml without error
    mlflow_uri: Optional[str] = field(default=None)
    mlflow_experiment_name: Optional[str] = field(default=None)
    token: Optional[str] = field(default=None)
    load_in_4bit: bool = field(default=False)
    use_snapshot_download: bool = field(default=True)
    early_stopping: bool = field(default=False)
    wandb_token: str = field(default="")
    wandb_project: str = field(default="")
    use_mxfp4: bool = field(default=False)


# --- Dataset Loading ---


def _is_hf_dataset_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(
        os.path.join(path, "dataset_info.json")
    )


def _load_dataset_auto(path: str) -> Dataset:
    """Load dataset from path, auto-detecting format."""
    if path.endswith(".jsonl") or path.endswith(".json"):
        return load_dataset("json", data_files=path, split="train")
    if path.endswith(".arrow"):
        return load_dataset("arrow", data_files=path, split="train")
    if _is_hf_dataset_dir(path):
        ds = load_from_disk(path)
        if isinstance(ds, DatasetDict):
            split = "train" if "train" in ds else list(ds.keys())[0]
            ds = ds[split]
        return ds
    # Fallback: find JSON/JSONL in directory
    json_files = sorted(
        _glob.glob(os.path.join(path, "*.json"))
        + _glob.glob(os.path.join(path, "*.jsonl"))
    )
    if json_files:
        return load_dataset("json", data_files=json_files, split="train")
    raise FileNotFoundError(f"No dataset files found in '{path}'")


def load_datasets(script_args: ScriptArguments) -> Tuple[Dataset, Optional[Dataset]]:
    """Load train and optional validation datasets."""
    logger.info(f"Loading training dataset from {script_args.train_dataset_path}")
    train_ds = _load_dataset_auto(script_args.train_dataset_path)
    logger.info(f"Training: {len(train_ds)} samples, columns: {train_ds.column_names}")

    test_ds = None
    if script_args.val_dataset_path:
        logger.info(f"Loading validation dataset from {script_args.val_dataset_path}")
        test_ds = _load_dataset_auto(script_args.val_dataset_path)
        logger.info(f"Validation: {len(test_ds)} samples")

    return train_ds, test_ds


# --- Dataset Preparation ---


def detect_dataset_format(dataset: Dataset, script_args: ScriptArguments) -> str:
    """Detect dataset format from columns."""
    if script_args.dataset_format != "auto":
        return script_args.dataset_format

    columns = dataset.column_names
    if "input_ids" in columns:
        return "pretokenized"
    if script_args.messages_field in columns:
        return "messages"
    if script_args.text_field in columns:
        return "text"

    raise ValueError(
        f"Cannot detect dataset format. Columns: {columns}. "
        f"Expected 'input_ids', '{script_args.messages_field}', or '{script_args.text_field}'."
    )


def prepare_dataset(
    tokenizer: AutoTokenizer,
    script_args: ScriptArguments,
    train_ds: Dataset,
    test_ds: Optional[Dataset] = None,
):
    """Tokenize dataset for training."""
    fmt = detect_dataset_format(train_ds, script_args)
    logger.info(f"Dataset format: {fmt}")

    if fmt == "pretokenized":
        return train_ds, test_ds

    max_length = script_args.max_length if script_args.apply_truncation else None

    if fmt == "messages":
        def tokenize_fn(sample):
            messages = sample[script_args.messages_field]
            if isinstance(messages, str):
                messages = json.loads(messages)
            ids = tokenizer.apply_chat_template(messages, tokenize=True)
            if max_length:
                ids = ids[:max_length]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        lm_train = train_ds.map(tokenize_fn, remove_columns=list(train_ds.features))
        lm_test = test_ds.map(tokenize_fn, remove_columns=list(test_ds.features)) if test_ds else None
    else:
        def tokenize_fn(sample):
            return tokenizer(
                sample[script_args.text_field],
                padding=False,
                truncation=script_args.apply_truncation,
                max_length=max_length,
            )

        lm_train = train_ds.map(tokenize_fn, remove_columns=list(train_ds.features), batched=True, batch_size=1000)
        lm_test = test_ds.map(tokenize_fn, remove_columns=list(test_ds.features), batched=True, batch_size=1000) if test_ds else None

    logger.info(f"Tokenized train samples: {len(lm_train)}")
    if lm_test:
        logger.info(f"Tokenized val samples: {len(lm_test)}")

    return lm_train, lm_test


# --- Training ---


def train(script_args: ScriptArguments, training_args: TrainingArguments, train_ds, test_ds):
    """Run training on CPU with LoRA."""
    set_seed(training_args.seed)

    # Download model
    if script_args.use_snapshot_download:
        logger.info(f"Downloading model {script_args.model_id}")
        os.makedirs("/tmp/tmp_folder", exist_ok=True)
        snapshot_download(repo_id=script_args.model_id, local_dir="/tmp/tmp_folder")
        model_path = "/tmp/tmp_folder"
    else:
        model_path = script_args.model_id

    # Load model for CPU
    dtype = getattr(torch, script_args.torch_dtype) if script_args.torch_dtype != "auto" else torch.float32
    model_kwargs = {
        "torch_dtype": dtype,
        "use_cache": not training_args.gradient_checkpointing,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "cache_dir": "/tmp/.cache",
    }
    if script_args.attn_implementation:
        model_kwargs["attn_implementation"] = script_args.attn_implementation

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    train_ds, test_ds = prepare_dataset(tokenizer, script_args, train_ds, test_ds)

    # Apply LoRA
    if script_args.use_peft:
        lora_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            target_modules="all-linear" if script_args.target_modules is None else script_args.target_modules,
            lora_dropout=script_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Disable reporting (no wandb/mlflow for CPU workshop)
    training_args.report_to = []

    # Setup checkpoint dir
    original_output_dir = training_args.output_dir
    if script_args.checkpoint_dir:
        os.makedirs(script_args.checkpoint_dir, exist_ok=True)
        training_args.output_dir = script_args.checkpoint_dir

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Train
    checkpoint = get_last_checkpoint(script_args.checkpoint_dir) if script_args.checkpoint_dir and script_args.use_checkpoints else None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Log metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save model
    if script_args.use_peft and script_args.merge_weights:
        # Save adapter, then merge
        temp_dir = "/tmp/adapter_temp"
        trainer.model.save_pretrained(temp_dir)
        merged = AutoPeftModelForCausalLM.from_pretrained(
            temp_dir, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged = merged.merge_and_unload()
        merged.save_pretrained(original_output_dir, safe_serialization=True)
        tokenizer.save_pretrained(original_output_dir)
    else:
        trainer.save_model(original_output_dir)
        tokenizer.save_pretrained(original_output_dir)

    logger.info(f"Model saved to {original_output_dir}")


def main():
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["WANDB_DISABLED"] = "true"

    train_ds, test_ds = load_datasets(script_args)
    train(script_args, training_args, train_ds, test_ds)


if __name__ == "__main__":
    main()

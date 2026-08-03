import os
import datetime
from typing import Dict, Optional
from dataclasses import dataclass, field

from accelerate import Accelerator

from huggingface_hub import snapshot_download
from datasets import load_dataset

import mlflow

import torch

from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoConfig, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from peft import AutoPeftModelForCausalLM, LoraConfig, prepare_model_for_kbit_training

from trl import SFTTrainer, TrlParser

import subprocess

@dataclass
class ScriptArguments:
    """
    Arguments for the script execution.
    """

    chunk_size: Optional[int] = field(
        default=2048,
        metadata={"help": "chunk_size"}
    )

    max_seq_length: int = field(
        default=512,
        metadata={"help": "The maximum sequence length for SFT Trainer"}
    )

    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "lora_r"}
    )

    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "lora_dropout"}
    )

    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "lora_dropout"}
    )

    merge_weights: Optional[bool] = field(
        default=False,
        metadata={"help": "Merge adapter with base model"}
    )

    mlflow_uri: Optional[str] = field(
        default=None,
        metadata={"help": "MLflow tracking ARN"}
    )

    mlflow_experiment_name: Optional[str] = field(
        default=None,
        metadata={"help": "MLflow experiment name"}
    )

    model_id: str = field(
        default=None,
        metadata={"help": "Model ID to use for SFT training"}
    )

    token: str = field(
        default=None,
        metadata={"help": "Hugging Face API token"}
    )

    train_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training dataset"}
    )

    test_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the test dataset"}
    )


def init_distributed():
    # Initialize the process group
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))  # Use "gloo" backend for CPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank


def download_model(model_name):

    destination = "/tmp/tmp_folder"
    
    os.makedirs(destination, exist_ok=True)
    
    if model_name.startswith("s3://"):
        print(f"Downloading model from S3: {model_name}")
        subprocess.run(['aws', 's3', 'cp', model_name, destination, '--recursive', '--quiet'])
    else:
        print(f"Downloading model from HF: {model_name}")
        snapshot_download(repo_id=model_name, local_dir=destination)

    print(f"Model {model_name} downloaded under {destination}")


def set_custom_env(env_vars: Dict[str, str]) -> None:
    """
    Set custom environment variables.

    Args:
        env_vars (Dict[str, str]): A dictionary of environment variables to set.
                                   Keys are variable names, values are their corresponding values.

    Returns:
        None

    Raises:
        TypeError: If env_vars is not a dictionary.
        ValueError: If any key or value in env_vars is not a string.
    """
    if not isinstance(env_vars, dict):
        raise TypeError("env_vars must be a dictionary")

    for key, value in env_vars.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("All keys and values in env_vars must be strings")

    os.environ.update(env_vars)

    # Optionally, print the updated environment variables
    print("Updated environment variables:")
    for key, value in env_vars.items():
        print(f"  {key}: {value}")

def load_data(training_data_location, test_data_location):
    # Load datasets
    train_ds = load_dataset(
        "json",
        data_files=os.path.join(training_data_location, "dataset.json"),
        split="train"
    )

    if script_args.test_dataset_path:
        test_ds = load_dataset(
            "json",
            data_files=os.path.join(test_data_location, "dataset.json"),
            split="train"
        )
    else:
        test_ds = None

    return train_ds, test_ds

def train(script_args, training_args):
    set_seed(training_args.seed)

    mlflow_enabled = (
        script_args.mlflow_uri is not None
        and script_args.mlflow_experiment_name is not None
        and script_args.mlflow_uri != ""
        and script_args.mlflow_experiment_name != ""
    )

    accelerator = Accelerator()

    if script_args.token is not None:
        os.environ.update({"HF_TOKEN": script_args.token})
        accelerator.wait_for_everyone()

    # Download model based on training setup (single or multi-node)
    if int(os.environ.get("SM_HOST_COUNT", 1)) == 1:
        if accelerator.is_main_process:
            download_model(script_args.model_id)
    else:
        download_model(script_args.model_id)

    accelerator.wait_for_everyone()

    model_location = "/tmp/tmp_folder"

    tokenizer = AutoTokenizer.from_pretrained(model_location)

    # Set Tokenizer pad Token
    tokenizer.pad_token = tokenizer.eos_token

    # # tokenize and chunk dataset
    # lm_train_dataset = train_ds.map(
    #     lambda sample: tokenizer(sample["text"]), remove_columns=list(train_ds.features)
    # )

    # if test_ds is not None:
    #     lm_test_dataset = test_ds.map(
    #         lambda sample: tokenizer(sample["text"]), remove_columns=list(train_ds.features)
    #     )

    #     print(f"Total number of test samples: {len(lm_test_dataset)}")
    # else:
    #     lm_test_dataset = None

    train_ds, test_ds = load_data(script_args.train_dataset_path, script_args.test_dataset_path)

    accelerator.wait_for_everyone()

    if training_args.bf16:
        print("flash_attention_2 init")      
        torch_dtype = torch.bfloat16

        model_configs = {
            "attn_implementation": "flash_attention_2",
            "torch_dtype": torch_dtype,
        }
    elif training_args.fp16:
        torch_dtype = torch.float16
        
        model_configs = {
            "torch_dtype": torch_dtype,
        }
    else:
        torch_dtype = torch.float32
        
        model_configs = {
            "torch_dtype": torch_dtype,
        }

    print(f"torch_dtype = {torch_dtype}")

    if training_args.fsdp is not None and training_args.fsdp != "" and training_args.fsdp_config is not None and len(training_args.fsdp_config) > 0:

        bnb_config_params = {
            "bnb_4bit_quant_storage": torch_dtype
        }

        trainer_configs = {
            "fsdp": training_args.fsdp,
            "fsdp_config": training_args.fsdp_config,
            "gradient_checkpointing_kwargs": {"use_reentrant": False}
        }
    else:
        bnb_config_params = dict()
        trainer_configs = {
            "gradient_checkpointing": training_args.gradient_checkpointing,
        }

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        **bnb_config_params
    )

    # VL checkpoints need the image-text-to-text auto class: AutoModelForCausalLM
    # raises "Unrecognized configuration class Qwen3VLConfig" on them, and the reverse
    # is also true, so the branch is required in both directions. Membership in the
    # image-text-to-text mapping is the reliable test — a model_type.endswith("_vl")
    # heuristic misses most VLMs, including Qwen3-VL-MoE ("qwen3_vl_moe").
    #
    # On text-only SFT the vision tower ends up unchanged, but not because it is
    # untargeted: target_modules="all-linear" does attach LoRA to its 116 linears.
    # A text-only batch carries no pixel_values, so the tower never runs, those
    # adapters get no gradient, lora_B stays at its zero init, and the merge writes
    # W + 0 = W bit-for-bit. Pass exclude_modules to skip the tower outright.
    _cfg = AutoConfig.from_pretrained(model_location, trust_remote_code=True)
    _is_vl = type(_cfg) in AutoModelForImageTextToText._model_mapping
    _auto_cls = AutoModelForImageTextToText if _is_vl else AutoModelForCausalLM
    _load_kw = dict(
        trust_remote_code=True,
        quantization_config=bnb_config,
        cache_dir="/tmp/.cache",
        **model_configs
    )
    if _auto_cls is AutoModelForCausalLM:
        # VL composite models (qwen3_vl) reject use_cache in __init__;
        # text models accept it. Set it post-load for VL instead.
        _load_kw["use_cache"] = not training_args.gradient_checkpointing
    model = _auto_cls.from_pretrained(model_location, **_load_kw)
    if _auto_cls is not AutoModelForCausalLM:
        try:
            model.config.text_config.use_cache = not training_args.gradient_checkpointing
        except AttributeError:
            model.config.use_cache = not training_args.gradient_checkpointing

    if training_args.fsdp is None and training_args.fsdp_config is None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
    else:
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=script_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print(f"max_seq_length: {script_args.max_seq_length}")

    print("Disabling checkpointing and setting up logging")
    training_args.save_strategy="no"
    training_args.logging_strategy="steps"
    training_args.logging_steps=1
    training_args.log_on_each_node=False
    training_args.ddp_find_unused_parameters=False

    # eval_dataset alone does not produce metrics: HF defaults eval_strategy to "no",
    # so the held-out split gets tokenized and then never scored. Gate on test_ds
    # because Trainer raises if eval_strategy != "no" with no eval_dataset. eval_steps
    # is set explicitly because SFTTrainer rebuilds these args as an SFTConfig, which
    # re-runs __post_init__ and would fall back to logging_steps (1 here) — an eval
    # after every single optimizer step. A value < 1 is read as a fraction of max_steps.
    # Anything set in args.yaml wins.
    if test_ds is not None:
        if training_args.eval_strategy == "no":
            training_args.eval_strategy = "steps"
        if training_args.eval_strategy == "steps" and not training_args.eval_steps:
            training_args.eval_steps = 0.25

    print(f"eval_strategy: {training_args.eval_strategy}, eval_steps: {training_args.eval_steps}")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds if test_ds is not None else None,
        processing_class=tokenizer,
        peft_config=peft_config
    )

    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    if mlflow_enabled:
        print("MLflow tracking under ", script_args.mlflow_experiment_name)
        mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME", None))
        train_dataset_mlflow = mlflow.data.from_pandas(train_ds.to_pandas(), name="train_dataset")
        mlflow.log_input(train_dataset_mlflow, context="train")

        if test_ds is not None:
            test_dataset_mlflow = mlflow.data.from_pandas(test_ds.to_pandas(), name="test_dataset")
            mlflow.log_input(test_dataset_mlflow, context="test")

        trainer.train()
    else:
        trainer.train()

    # a step-based schedule only lands on multiples of eval_steps, so the last step
    # usually goes unscored. This guarantees one eval against the final weights.
    if test_ds is not None:
        eval_metrics = trainer.evaluate()
        print(f"final eval metrics: {eval_metrics}")

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if script_args.merge_weights:
        print(f"merge adapter weights: {script_args.merge_weights}")
        output_dir = "/tmp/model"

        # merge adapter weights with base model and save
        # save int 4 model
        trainer.save_model(output_dir)

        if accelerator.is_main_process:
            # clear memory
            print("clearing memory...")
            del model
            del trainer

            print("emptying cuda cache...")
            torch.cuda.empty_cache()

            print("loading base model...")
            # load PEFT model. VL checkpoints need the image-text auto class; load the
            # base explicitly and attach the adapter, because peft's
            # AutoPeftModelForCausalLM hardcodes AutoModelForCausalLM as its target
            # class and raises on a VL adapter. Loading in bf16 matches the training
            # dtype (the text-only branch below keeps the workshop's fp16).
            _base_cfg = AutoConfig.from_pretrained(model_location, trust_remote_code=True)
            if type(_base_cfg) in AutoModelForImageTextToText._model_mapping:
                from peft import PeftModel
                _base = AutoModelForImageTextToText.from_pretrained(
                    model_location,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                model = PeftModel.from_pretrained(_base, output_dir)
            else:
                model = AutoPeftModelForCausalLM.from_pretrained(
                    output_dir,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )

            print("merging adapter with base...")
            # Merge LoRA and base model and save
            model = model.merge_and_unload()

            print("saving merged model...")
            model.save_pretrained(
                training_args.output_dir, 
                safe_serialization=True
            )
    else:
        print(f"merge adapter weights: {script_args.merge_weights}")
        trainer.save_model(training_args.output_dir)

    if accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        # VL: persist the processor (image preprocessor config etc.) so the
        # merged checkpoint is servable as a multimodal model by vLLM
        try:
            from transformers import AutoProcessor
            _proc = AutoProcessor.from_pretrained(model_location, trust_remote_code=True)
            _proc.save_pretrained(training_args.output_dir)
        except Exception as _e:
            print(f"processor save skipped: {_e}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    # Call this function at the beginning of your script
    local_rank = init_distributed()

    # Now you can use distributed functionalities
    torch.distributed.barrier(device_ids=[local_rank])

    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    set_custom_env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

    if script_args.mlflow_uri is not None and script_args.mlflow_experiment_name is not None and \
        script_args.mlflow_uri != "" and script_args.mlflow_experiment_name != "":
        print("mlflow init")
        mlflow.enable_system_metrics_logging()
        mlflow.autolog()
        mlflow.set_tracking_uri(script_args.mlflow_uri)
        mlflow.set_experiment(script_args.mlflow_experiment_name)

        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
        set_custom_env({"MLFLOW_RUN_NAME": f"Fine-tuning-{formatted_datetime}"})
        set_custom_env({"MLFLOW_EXPERIMENT_NAME": script_args.mlflow_experiment_name})

    # launch training
    train(script_args, training_args)

# Import required libraries
import os
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sagemaker.remote_function import remote
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
from datasets import load_from_disk, load_dataset
import argparse
import bitsandbytes as bnb
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset, DatasetDict
import mlflow
from mlflow.models import infer_signature

# Set environment variables for distributed training
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def init_distributed():
    """Initialize distributed training"""
    try:
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(local_rank)
                print(f"Initialized distributed with rank {torch.distributed.get_rank()}, world_size {torch.distributed.get_world_size()}")
                return True
    except Exception as e:
        print(f"Error initializing distributed: {e}")
    return False

def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--dataset_path", type=str, default="lm_dataset", help="Path to dataset."
    )
    parser.add_argument(
        "--hf_token", type=str, default="", help="Path to dataset."
    )
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--lora_r", type=int, default=8, help="Lora R"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="Lora alpha"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="Lora dropout"
    )

    parser.add_argument(
        "--lora_target_modules", type=list, default="q_proj,v_proj", help="Lora target modules"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to use for training."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Path to deepspeed config file.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--merge_weights",
        type=bool,
        default=True,
        help="Whether to merge LoRA weights with base model.",
    )
    parser.add_argument(
        "--mlflow_arn", type=str
    )
    parser.add_argument(
        "--experiment_name", type=str
    )
    parser.add_argument(
        "--run_id", type=str
    )
    args, _ = parser.parse_known_args()

    if args.hf_token:
        print(f"Logging into the Hugging Face Hub with token {args.hf_token[:10]}...")
        login(token=args.hf_token)

    return args

LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def find_all_linear_names(hf_model):
    """
    Identifies all linear layers in the model that can be targeted by LoRA.
    Specifically looks for 4-bit quantized linear layers.
    """
    lora_module_names = set()
    for name, module in hf_model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

def train_fn(
        args,
        lora_r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        num_train_epochs=1,
        chunk_size=2048,
        gradient_checkpointing=True,
        merge_weights=False,
        token=None
):  
    print("############################################")
    print("Number of GPUs: ", torch.cuda.device_count())
    print("############################################")
    
    # Get local rank from environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"Running on local_rank: {local_rank}")
    
    # Check if we're in distributed mode
    is_distributed = torch.distributed.is_initialized()
    is_main_process = not is_distributed or torch.distributed.get_rank() == 0
    
    model_id = args.model_id

    # Set device based on local_rank
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True if token else None)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    
    if token is not None:
        login(token=token)

    # Load and prepare dataset
    data_df = pd.read_json(os.path.join(args.dataset_path, "train_dataset.json"), orient='records', lines=True)
    train, test = train_test_split(data_df, test_size=0.3)

    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)

    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})  
    if is_main_process:
        print(f"Loaded train dataset with {len(train_dataset)} samples")

    def template_dataset(examples):
        return{"text":  tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    # Process datasets consistently across all ranks
    train_dataset = dataset["train"].map(template_dataset, remove_columns=["messages"])
    lm_train_dataset = train_dataset.map(
        lambda sample: tokenizer(sample["text"]), batched=True, batch_size=per_device_train_batch_size
    )
    
    # Use deterministic sample selection to ensure consistency across ranks
    torch.manual_seed(args.seed)  # Set seed for reproducibility
    lm_train_dataset = lm_train_dataset.select(range(100))
    if is_main_process:
        print(f"Total number of train samples: {len(lm_train_dataset)}")
        print(lm_train_dataset[0])

    lm_test_dataset = None
    if test_dataset is not None:
        test_dataset = dataset["test"].map(template_dataset, remove_columns=["messages"])
        lm_test_dataset = test_dataset.map(
            lambda sample: tokenizer(sample["text"]), batched=True, batch_size=per_device_train_batch_size
        )
        lm_test_dataset = lm_test_dataset.select(range(10))
        if is_main_process:
            print(f"Total number of test samples: {len(lm_test_dataset)}")

    # Synchronize processes after dataset creation
    if is_distributed:
        torch.distributed.barrier()

    # Configure QLoRA parameters using local_rank for device placement
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model with proper device specification
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        cache_dir="/tmp/.cache"
    )
    model.to(device)  # Move model to proper device for this rank

    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)

    # Get lora target modules
    modules = find_all_linear_names(model)
    if is_main_process:
        print(f"Found {len(modules)} modules to quantize: {modules}")

    # Configure LoRA
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to model
    model = get_peft_model(model, config)
    
    # Enable gradient checkpointing for memory efficiency
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    if is_main_process:
        print_trainable_parameters(model)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Use HF Trainer with DDP settings
    trainer = transformers.Trainer(
        model=model,
        train_dataset=lm_train_dataset,
        eval_dataset=lm_test_dataset,
        optimizers=(optimizer, None),  # Pass custom optimizer
        args=transformers.TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            logging_steps=2,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            bf16=args.bf16,
            save_strategy="no",
            output_dir="outputs",
            report_to="mlflow",
            run_name="llama3-peft",
            
            # DDP configuration
            local_rank=local_rank,
            ddp_backend="nccl",
            ddp_find_unused_parameters=False,
            dataloader_num_workers=4,  # Optimize data loading
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Only run MLflow logging on main process
    if is_main_process:
        mlflow.set_tracking_uri(args.mlflow_arn)
        mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run(run_id=args.run_id) as run:
            lora_params = {'lora_alpha':lora_alpha, 'lora_dropout': lora_dropout, 'r':lora_r, 'bias': 'none', 'task_type': 'CAUSAL_LM', 'target_modules': modules}
            mlflow.log_params(lora_params)
            
            trainer.train()
            
            if merge_weights:
                output_dir = "/tmp/model"
                
                # merge adapter weights with base model and save
                # save int 4 model
                trainer.model.save_pretrained(output_dir, safe_serialization=False)
                # clear memory
                del model
                del trainer
                
                torch.cuda.empty_cache()
                
                # load PEFT model in fp16
                model = AutoPeftModelForCausalLM.from_pretrained(
                    output_dir,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    cache_dir="/tmp/.cache"
                )
                
                # Merge LoRA and base model and save
                model = model.merge_and_unload()
                model.save_pretrained(
                    "/opt/ml/model", safe_serialization=True, max_shard_size="2GB"
                )
            else:
                model.save_pretrained("/opt/ml/model", safe_serialization=True)
                
            tmp_tokenizer = AutoTokenizer.from_pretrained(model_id)
            tmp_tokenizer.save_pretrained("/opt/ml/model")
            params = {
                "top_p": 0.9,
                "temperature": 0.9,
                "max_new_tokens": 200,
            }
            signature = infer_signature("inputs","generated_text", params=params)
            mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tmp_tokenizer},
                signature=signature,
                artifact_path="model",
                model_config=params
            )
    else:
        #train
        trainer.train()

def main():
    is_distributed = init_distributed()
    
    # Print basic environment diagnostics
    print(f"CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}")
    print(f"Running with distributed: {is_distributed}")
    
    # Parse arguments after initializing distributed
    args = parse_arge()
    
    train_fn(
        args,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=args.epochs,
        merge_weights=True,
        token=args.hf_token,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing
    )

if __name__ == "__main__":
    main()
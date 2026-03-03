#!/usr/bin/env python

# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 Seochan99. All rights reserved.
# source:
# cite: https://github.com/seochan99/stable-diffusion-3.5-text2image-lora/blob/main/train_text_to_image_lora_sd35.py
# cite: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
# The code here are combination of the repository mentioned above.
# Wrote by Amin Dashti email: dashtiam@amazon.com
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import copy
import shutil

from pathlib import Path
import yaml

import datasets
from datasets import load_dataset
import numpy as np
import torch

import torch.utils.checkpoint
from torch.utils.data import DataLoader

import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

from torchvision import transforms
from tqdm.auto import tqdm

from transformers import (
    PretrainedConfig
)

import diffusers
from diffusers import AutoencoderKL, SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available

from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from PIL import Image
from PIL.ImageOps import exif_transpose

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.37.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def import_model_class(
    pretrained_model_name_or_path: str, revision: str, subfolder="text_encoder"
):
    """Import the appropriate text encoder class based on model configuration."""
    config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class_name = config.architectures[0]
    if model_class_name == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class_name == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"Unsupported model class: {model_class_name}")


def save_model_card(
    repo_id: str,
    images: list = None,
    base_model: str = None,
    dataset_name: str = None,
    repo_folder: str = None,
):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "lora",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipeline,
    args,
    accelerator,
    epoch,
    is_final_validation=False,
):
    if not accelerator.is_main_process:
        return []

    logger.info(
        f"Running validation...\n"
        f"Generating {args.num_validation_images} images with prompt: "
        f"{args.validation_prompt}"
    )

    images = []

    # if using a discrete gpu for validation, get the last GPU.
    # since the number of processes has been reduced to total GPUs - 1,
    # it is the same as accelerator.num_processes
    
    if args.use_discrete_gpu_for_validation:
        device = f"cuda:{accelerator.num_processes}"
    else:
        device = accelerator.device

    # Unwrap model if using Accelerate / DeepSpeed
    if hasattr(pipeline, "unet"):
        pipeline.unet = accelerator.unwrap_model(pipeline.unet)
    if hasattr(pipeline, "text_encoder"):
        pipeline.text_encoder = accelerator.unwrap_model(pipeline.text_encoder)

    # Move pipeline to GPU (safe on g6e)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # Use generator on correct device
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None

    # Stable inference (no autocast — prevents black images)
    with torch.no_grad():
        for _ in range(args.num_validation_images):
            output = pipeline(
                args.validation_prompt,
                num_inference_steps=30,
                generator=generator,
            )
            images.append(output.images[0])

    # (Optional but recommended) Move back to CPU after validation
    # pipeline = pipeline.to("cpu")
    # torch.cuda.empty_cache()

    # Log images
    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"

        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                phase_name,
                np_images,
                epoch,
                dataformats="NHWC",
            )

        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(
                            image,
                            caption=f"{i}: {args.validation_prompt}",
                        )
                        for i, image in enumerate(images)
                    ]
                }
            )

    return images


def parse_args():
    # Step 1: temporary parser to get --config
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    config_args, remaining_argv = config_parser.parse_known_args()
    
    # Step 2: main parser
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.",
        parents=[config_parser]
    )
    
    # Step 3: add all arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--clip_model_name_or_path_G",
        type=str,
        default=None,
        help="Path to clip model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--clip_model_name_or_path_L",
        type=str,
        default=None,
        help="Path to clip model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--revision", type=str, default=None, help="Revision of pretrained model.")
    parser.add_argument("--variant", type=str, default=None, help="Variant of the model files (e.g., fp16).")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name or path of the dataset.")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="Config of the dataset, if any.")
    parser.add_argument("--train_data_dir", type=str, default=None, help="Folder with training data.")
    parser.add_argument("--image_column", type=str, default="image", help="Column with image.")
    parser.add_argument("--caption_column", type=str, default="text", help="Column with caption.")
    parser.add_argument("--validation_prompt", type=str, default=None, help="Prompt used during validation.")
    parser.add_argument("--num_validation_images", type=int, default=4, help="Number of validation images.")
    parser.add_argument("--validation_epochs", type=int, default=1, help="Run validation every X epochs.")
    parser.add_argument("--validation_inference_steps", type=int, default=30, help="Number of inference steps for each validation image generation.")
    parser.add_argument("--use_discrete_gpu_for_validation", action="store_true", default=False, help="Whether to isolate a specific GPU for validation. Only for use on small multi-gpu instances to prevent CUDA OOM issues.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max number of training samples.")
    parser.add_argument("--output_dir", type=str, default="sd-model-finetuned-lora", help="Output directory.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for datasets/models.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--max_sequence_length", type=int, default=128, help="Maximum sequence length for text encoder input",)
    parser.add_argument("--resolution", type=int, default=512, help="Input image resolution.")
    parser.add_argument("--center_crop", action="store_true", default=False, help="Center crop images.")
    parser.add_argument("--random_flip", action="store_true", help="Randomly flip images horizontally.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Override total training steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale LR by number of GPUs etc.")
    parser.add_argument("--text_encoder_lr", type=float, default=5e-6, help="Learning rate for text encoder LoRA parameters")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="LR scheduler type.")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="LR warmup steps.")
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR weighting gamma.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam optimizer.")
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32 on Ampere GPUs.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of workers for dataloader.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Adam weight decay.")
    parser.add_argument("--adam_weight_decay_text_encoder", type=float, default=1e-3, help="Weight decay for text encoder parameters")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="Token for pushing to Hub.")
    parser.add_argument("--prediction_type", type=str, default=None, help="Prediction type: epsilon, v_prediction.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Hub repository name.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="TensorBoard log directory.")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default=None, help="Mixed precision.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report to integration.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Checkpointing frequency.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max checkpoints to store.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Use xformers.")
    parser.add_argument("--noise_offset", type=float, default=0, help="Noise offset scale.")
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"], help="Timestep sampling weighting scheme")
    parser.add_argument("--logit_mean", type=float, default=0.0, help="Mean for logit_normal weighting scheme")
    parser.add_argument("--logit_std", type=float, default=1.0, help="Standard deviation for logit_normal weighting scheme")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Scale factor for mode weighting scheme")
    parser.add_argument("--precondition_outputs", type=int, default=1, help="Enable output preconditioning (1=enabled, 0=disabled)")

    
    # ═══════════════════════════════════════════════════════════
    # LoRA Configuration
    # ═══════════════════════════════════════════════════════════
    parser.add_argument("--rank", type=int, default=4, help="LoRA update matrix rank.")
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help="Comma-separated list of layer names (e.g., 'attn.to_q','ff.net.0.proj') to apply LoRA to within selected blocks. If --lora_blocks is not set, these apply to standard transformer attention modules.",
    )
    parser.add_argument(
        "--lora_blocks",
        type=str,
        default=None,
        help="Comma-separated list of transformer block indices (e.g., '0,1,5') to apply LoRA to. If specified, --lora_layers will be prefixed with 'transformer_blocks.{idx}.'. If None, default LoRA targets are used.",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train text encoder LoRA adapters",
    )
    
    parser.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
        choices=[f.lower() for f in dir(transforms.InterpolationMode) if not f.startswith("__")],
        help="Image interpolation mode."
    )

    # Step 4: override defaults with YAML if --config was provided
    if config_args.config:
        with open(config_args.config, "r") as f:
            yaml_args = yaml.safe_load(f)
        valid_args = {a.dest for a in parser._actions}
        for key, value in yaml_args.items():
            if key not in valid_args:
                raise ValueError(f"Unknown argument in config file: {key}")
            parser.set_defaults(**{key: value})

    # Step 5: parse final args
    args = parser.parse_args(remaining_argv)

    # Step 6: fix local_rank from environment
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    # Step 7: ensure pretrained_model_name_or_path is provided
    if args.pretrained_model_name_or_path is None:
        raise ValueError("--pretrained_model_name_or_path must be specified either in CLI or in YAML")
    
    # Step 8: sanity check dataset
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def main():
    
    args = parse_args()
    
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Set the dtype for model weights
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
    )
    # ═══════════════════════════════════════════════════════════
    # Tokenizer Loading
    # ═══════════════════════════════════════════════════════════
    tokenizer_one = pipe.tokenizer
    tokenizer_two = pipe.tokenizer_2
    tokenizer_three = pipe.tokenizer_3
    # ═══════════════════════════════════════════════════════════
    # Text Encoder Class Detection
    # ═══════════════════════════════════════════════════════════

    text_encoder_one = pipe.text_encoder      # CLIP-G (with projection)
    text_encoder_two = pipe.text_encoder_2    # CLIP-L (with projection)
    text_encoder_three = pipe.text_encoder_3  # T5

    del pipe
    # ═══════════════════════════════════════════════════════════
    # Model Loading (Scheduler, Encoders, VAE, Transformer)
    # ═══════════════════════════════════════════════════════════
    noise_scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)


    # Load VAE and Transformer
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant
    )

    # Freeze base model parameters
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Additional MPS compatibility check
    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError("MPS backend does not support bfloat16 mixed precision")

    # Move models to appropriate device and dtype
    vae.to(accelerator.device, dtype=torch.float32)  # VAE should stay in float32
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    # Enable gradient checkpointing for memory efficiency
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    # ═══════════════════════════════════════════════════════════
    # LoRA Adapter Configuration and Addition
    # ═══════════════════════════════════════════════════════════
    # Configure target modules for LoRA adaptation
    # This determines which transformer layers will have LoRA adapters applied
    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
        logger.info(f"Using custom LoRA layers: {target_modules}")
    else:
        # Default SD3.5 transformer attention modules
        # These are the key attention projection layers in the transformer
        target_modules = [
            "attn.add_k_proj",  # Additional key projection for joint attention
            "attn.add_q_proj",  # Additional query projection for joint attention
            "attn.add_v_proj",  # Additional value projection for joint attention
            "attn.to_add_out",  # Additional output projection
            "attn.to_k",  # Standard key projection
            "attn.to_out.0",  # Standard output projection
            "attn.to_q",  # Standard query projection
            "attn.to_v",  # Standard value projection
        ]
        logger.info(
            f"Using default SD3.5 LoRA target modules: {len(target_modules)} modules"
        )
        
    # Apply LoRA to specific transformer blocks if specified
    # This allows for fine-grained control over which layers are adapted
    if args.lora_blocks is not None:
        blocks = [int(b.strip()) for b in args.lora_blocks.split(",")]
        target_modules = [
            f"transformer_blocks.{b}.{m}" for b in blocks for m in target_modules
        ]
        logger.info(f"Applying LoRA to specific blocks: {blocks}")


    # Create and add LoRA adapter to transformer
    # LoRA rank determines the expressiveness vs efficiency trade-off
    transformer_lora_config = LoraConfig(
        r=args.rank,  # LoRA rank (bottleneck dimension)
        lora_alpha=args.rank,  # Scaling factor (typically equal to rank)
        init_lora_weights="gaussian",  # Initialize with Gaussian distribution
        target_modules=target_modules,  # Which modules to apply LoRA to
    )
    transformer.add_adapter(transformer_lora_config)
    logger.info(f"Added LoRA adapter to transformer with rank {args.rank}")

    # Add LoRA adapters to text encoders if training them
    # Text encoder LoRA helps with better text understanding and prompt following
    # Add LoRA adapters to text encoders if training them
    if args.train_text_encoder:
        # Standard attention projection layers for CLIP text encoders
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)
        
        # Add LoRA to T5 text encoder
        t5_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # same convention for T5 attention
        )
        text_encoder_three.add_adapter(t5_lora_config)
    
        logger.info("Added LoRA adapters to all text encoders (CLIP and T5)")
    else:
        logger.info("Text encoders will remain frozen (no LoRA adaptation)")


    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


    # ═══════════════════════════════════════════════════════════
    # Optimizer Setup and Learning Rate Scaling
    # ═══════════════════════════════════════════════════════════
    # Scale learning rate if requested
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Cast training parameters to FP32 for mixed precision training stability
    if args.mixed_precision == "fp16":
        models_for_casting = [transformer]
        if args.train_text_encoder:
            models_for_casting.extend([text_encoder_one, text_encoder_two])
        # Only upcast trainable parameters (LoRA) to fp32
        cast_training_params(models_for_casting, dtype=torch.float32)

    # Collect trainable parameters
    transformer_lora_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )
    if args.train_text_encoder:
        text_lora_parameters_one = list(
            filter(lambda p: p.requires_grad, text_encoder_one.parameters())
        )
        text_lora_parameters_two = list(
            filter(lambda p: p.requires_grad, text_encoder_two.parameters())
        )

    # Setup parameter groups with different learning rates
    params_to_optimize = [
        {"params": transformer_lora_parameters, "lr": args.learning_rate}
    ]

        
    if args.train_text_encoder:
        text_lora_params_one = list(
            filter(lambda p: p.requires_grad, text_encoder_one.parameters())
        )
        text_lora_params_two = list(
            filter(lambda p: p.requires_grad, text_encoder_two.parameters())
        )
        text_lora_parameters_three = list(
            filter(lambda p: p.requires_grad, text_encoder_three.parameters())
        )
        params_to_optimize.extend(
            [
                {
                    "params": text_lora_params_one,
                    "weight_decay": args.adam_weight_decay_text_encoder,
                    "lr": args.text_encoder_lr or args.learning_rate,
                },
                {
                    "params": text_lora_params_two,
                    "weight_decay": args.adam_weight_decay_text_encoder,
                    "lr": args.text_encoder_lr or args.learning_rate,
                },
                {
                    "params": text_lora_parameters_three,
                    "weight_decay": args.adam_weight_decay_text_encoder,
                    "lr": args.text_encoder_lr or args.learning_rate,
                },
            ]
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            #data_dir=args.train_data_dir,
        )
    else:
        from datasets import load_from_disk
        dataset = load_from_disk(args.train_data_dir)
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    train_dataset = dataset["train"]
    column_names = train_dataset.column_names

    # Determine image and caption column names
    image_column = (
        args.image_column if args.image_column in column_names else column_names[0]
    )
    caption_column = (
        args.caption_column if args.caption_column in column_names else column_names[1]
    )

    
    def tokenize_prompt(tokenizer, prompt_list, max_length, device):
        """Tokenize a list of prompts safely."""
        return tokenizer(
            prompt_list,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).input_ids.to(device)


    def _encode_clip(text_encoder, tokenizer, prompts, device, dtype):
        inputs = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
    
        outputs = text_encoder(**inputs, output_hidden_states=True)
    
        embeds = outputs.hidden_states[-2].to(dtype)
        pooled = outputs.pooler_output.to(dtype)
        return embeds, pooled
    
    
    def _encode_t5(
        text_encoder,
        tokenizer,
        prompt_list,
        max_sequence_length,
        device,
        dtype,
        num_images_per_prompt,
    ):
        input_ids = tokenizer(
            prompt_list,
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt",
        ).input_ids.to(device)
    
        outputs = text_encoder(input_ids)
        prompt_embeds = outputs.last_hidden_state.to(device=device, dtype=dtype)
    
        if num_images_per_prompt > 1:
            b, s, h = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(
                b * num_images_per_prompt, s, h
            )
    
        return prompt_embeds

    def encode_prompt(
    text_encoders,
    tokenizers,
    prompt_list,
    max_sequence_length,
    device,
    weight_dtype,
    num_images_per_prompt=1,
    text_input_ids_list=None,
    ):
    
        clip_g, clip_l, t5 = text_encoders
        tok_g, tok_l, tok_t5 = tokenizers
    
        # ============================================================
        # CLIP-G
        # ============================================================
        if text_input_ids_list is None:
            clip_g_inputs = tok_g(
                prompt_list,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            clip_g_outputs = clip_g(**clip_g_inputs, output_hidden_states=True)
        else:
            clip_g_outputs = clip_g(
                input_ids=text_input_ids_list[0].to(device),
                output_hidden_states=True,
            )
    
        clip_g_embeds = clip_g_outputs.hidden_states[-2].to(device=device, dtype=weight_dtype)
        clip_g_pooled = clip_g_outputs.text_embeds.to(device=device, dtype=weight_dtype)

        # ============================================================
        # CLIP-L
        # ============================================================
        if text_input_ids_list is None:
            clip_l_inputs = tok_l(
                prompt_list,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            clip_l_outputs = clip_l(**clip_l_inputs, output_hidden_states=True)
        else:
            clip_l_outputs = clip_l(
                input_ids=text_input_ids_list[1].to(device),
                output_hidden_states=True,
            )
    
        clip_l_embeds = clip_l_outputs.hidden_states[-2].to(device=device, dtype=weight_dtype)
        clip_l_pooled = clip_l_outputs.text_embeds.to(device=device, dtype=weight_dtype)

    
        # ============================================================
        # Combine CLIP hidden states
        # ============================================================
        clip_embeds = torch.cat([clip_g_embeds, clip_l_embeds], dim=-1)
        pooled_embeds = torch.cat([clip_g_pooled, clip_l_pooled], dim=-1)
    
        # ============================================================
        # T5
        # ============================================================
        if text_input_ids_list is None:
            t5_input_ids = tok_t5(
                prompt_list,
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
                return_tensors="pt",
            ).input_ids.to(device)
        else:
            t5_input_ids = text_input_ids_list[2].to(device)
    
        t5_outputs = t5(t5_input_ids)
        t5_embeds = t5_outputs.last_hidden_state.to(device=device, dtype=weight_dtype)
    
        # ============================================================
        # Pad CLIP if needed
        # ============================================================
        if clip_embeds.shape[-1] < t5_embeds.shape[-1]:
            clip_embeds = torch.nn.functional.pad(
                clip_embeds,
                (0, t5_embeds.shape[-1] - clip_embeds.shape[-1]),
            )
    
        # ============================================================
        # Combine CLIP + T5
        # ============================================================
        prompt_embeds = torch.cat([clip_embeds, t5_embeds], dim=1)
    
        return prompt_embeds, pooled_embeds


    # Get the specified interpolation method from the args
    interpolation = getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper(), None)

    # Raise an error if the interpolation method is invalid
    if interpolation is None:
        raise ValueError(f"Unsupported interpolation mode {args.image_interpolation_mode}.")

    # Data preprocessing transformations
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=interpolation),  # Use dynamic interpolation method
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def preprocess_train(examples):
        """
        Preprocess training examples with image loading and text extraction.

        This function handles both local image paths and PIL Image objects,
        applies EXIF orientation correction, and prepares the data for training.

        Args:
            examples: Batch of examples from the dataset

        Returns:
            Dictionary with processed pixel_values and prompts
        """
        images = []
        for img_path in examples[image_column]:
            if isinstance(img_path, str):
                # Load image from local filesystem
                full_path = os.path.join(args.train_data_dir, img_path)
                img = Image.open(full_path).convert("RGB")
            else:
                # Handle PIL Image objects directly
                img = img_path.convert("RGB")

            # Apply EXIF orientation correction to prevent rotated images
            img = exif_transpose(img)
            images.append(img)

        # Apply preprocessing transforms (resize, crop, normalize)
        examples["pixel_values"] = [train_transforms(img) for img in images]
        examples["prompts"] = examples[caption_column]
        return examples

    # Limit dataset size for debugging if requested
    if args.max_train_samples is not None:
        train_dataset = train_dataset.shuffle(seed=args.seed).select(
            range(args.max_train_samples)
        )

    # Apply preprocessing transforms
    train_dataset = train_dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        """
        Collate function for DataLoader to batch examples efficiently.

        This function stacks image tensors and collects prompts into batches
        for efficient processing during training.

        Args:
            examples: List of preprocessed examples

        Returns:
            Dictionary with batched pixel_values and prompts
        """
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        prompts = [ex["prompts"] for ex in examples]
        return {"pixel_values": pixel_values, "prompts": prompts}

    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    # unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     unet, optimizer, train_dataloader, lr_scheduler
    # )

    if args.train_text_encoder:
        (
            transformer,
            text_encoder_one,
            text_encoder_two,
            text_encoder_three,  # Include T5
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            text_encoder_one,
            text_encoder_two,
            text_encoder_three,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        (
            transformer,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    
    # Register the hooks for efficient saving and loading of LoRA weights
    # accelerator.register_save_state_pre_hook(save_model_hook)
    # accelerator.register_load_state_pre_hook(load_model_hook)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        """Extract sigma values for given timesteps from noise scheduler."""
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()
            # Enable embedding gradients for text encoders
            accelerator.unwrap_model(
                text_encoder_one
            ).text_model.embeddings.requires_grad_(True)
            accelerator.unwrap_model(
                text_encoder_two
            ).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            # Determine which models to accumulate gradients for
            models_to_accumulate = [transformer]
            if args.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one, text_encoder_two])

            with accelerator.accumulate(models_to_accumulate):
                prompts = batch["prompts"]

                if args.train_text_encoder:
                    # Tokenize manually when training text encoders
                    tokens_one = tokenize_prompt(
                        tokenizer_one,
                        prompts,
                        77,
                        accelerator.device,
                    )
                    tokens_two = tokenize_prompt(
                        tokenizer_two,
                        prompts,
                        77,
                        accelerator.device,
                    )
                    tokens_three = tokenize_prompt(
                        tokenizer_three,
                        prompts,
                        args.max_sequence_length,
                        accelerator.device,
                    )
                
                    prompt_embeds, pooled_embeds = encode_prompt(
                        [text_encoder_one, text_encoder_two, text_encoder_three],
                        [None, None, None],
                        prompts,
                        args.max_sequence_length,
                        accelerator.device,
                        weight_dtype=weight_dtype,
                        text_input_ids_list=[tokens_one, tokens_two, tokens_three],
                    )
                
                else:
                    # Let encode_prompt handle tokenization internally
                    prompt_embeds, pooled_embeds = encode_prompt(
                        [text_encoder_one, text_encoder_two, text_encoder_three],
                        [tokenizer_one, tokenizer_two, tokenizer_three],
                        prompts,
                        args.max_sequence_length,
                        accelerator.device,
                        weight_dtype=weight_dtype,
                    )

                # Encode images to latent space
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = (
                    latents - vae.config.shift_factor
                ) * vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]

                # Compute timestep sampling density
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=batch_size,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(
                    device=latents.device
                )

                # Add noise to latents
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Forward pass through transformer
                model_pred_raw = transformer(  # model_pred -> model_pred_raw
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    return_dict=False,
                )[0]

                # Apply output preconditioning if enabled (SD3 paper section 3.4.2)
                # This transforms the model's output to be an estimate of x_0 (clean latents)
                # Reference: https://arxiv.org/abs/2403.03206
                if args.precondition_outputs:
                    # Apply preconditioning transformation as described in SD3 paper
                    # x_prediction = model_output * (-sigma_t) + x_t
                    model_pred = model_pred_raw * (-sigmas) + noisy_latents
                    target = latents  # Target is the clean latents
                    if accelerator.is_main_process and global_step % 100 == 0:
                        logger.info(
                            "Using preconditioned target: clean latents, model output transformed"
                        )
                else:
                    # Standard prediction without preconditioning
                    # For SD3.5, the model typically predicts clean latents directly
                    model_pred = model_pred_raw
                    target = latents  # Default to predicting clean latents
                    if accelerator.is_main_process and global_step % 100 == 0:
                        logger.info(
                            "Using non-preconditioned target: clean latents (direct prediction)"
                        )

                # Compute loss weighting
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                # Compute loss
                loss = torch.mean(
                    (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                # Backward pass
                accelerator.backward(loss)

                # Gradient clipping and optimizer step
                if accelerator.sync_gradients:
                    # Include T5 LoRA parameters if training
                    params_to_clip = (
                        itertools.chain(
                            transformer_lora_parameters,
                            text_lora_parameters_one,
                            text_lora_parameters_two,
                            text_lora_parameters_three if args.train_text_encoder else []
                        )
                        if args.train_text_encoder
                        else transformer_lora_parameters
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress and logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Log memory usage periodically
                if global_step % 100 == 0:
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            allocated_memory = torch.cuda.memory_allocated(i) / (
                                1024**3
                            )
                            reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)
                            accelerator.log(
                                {
                                    f"gpu_{i}_memory_allocated_gb": allocated_memory,
                                    f"gpu_{i}_memory_reserved_gb": reserved_memory,
                                },
                                step=global_step,
                            )

                # Save checkpoint periodically
                if (
                    accelerator.is_main_process
                    or accelerator.distributed_type == DistributedType.DEEPSPEED
                ) and global_step % args.checkpointing_steps == 0:
                    # Clean up old checkpoints if limit is set
                    if args.checkpoints_total_limit is not None:
                        checkpoints = [
                            d
                            for d in os.listdir(args.output_dir)
                            if d.startswith("checkpoint")
                        ]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1])
                        )
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = (
                                len(checkpoints) - args.checkpoints_total_limit + 1
                            )
                            for checkpoint_to_remove in checkpoints[:num_to_remove]:
                                shutil.rmtree(
                                    os.path.join(args.output_dir, checkpoint_to_remove)
                                )

                    # Save new checkpoint
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

            # Update progress bar with current metrics
            progress_bar.set_postfix(
                {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            )
            accelerator.log(
                {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]},
                step=global_step,
            )

            # Break if max steps reached
            if global_step >= args.max_train_steps:
                break

        # Run validation at the end of each epoch
        if (
            accelerator.is_main_process
            and args.validation_prompt is not None
            and epoch % args.validation_epochs == 0
        ):
            # ═════════ BEGIN VALIDATION ═════════
            logger.info("Running validation...")

            pipeline = StableDiffusion3Pipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
                # To save VRAM, load components onto CPU first.
                low_cpu_mem_usage=True,
            )
            
            pipeline.scheduler = noise_scheduler

            pipeline.transformer.add_adapter(transformer_lora_config)
            if args.train_text_encoder:
                pipeline.text_encoder.add_adapter(text_lora_config)
                pipeline.text_encoder_2.add_adapter(text_lora_config)
                pipeline.text_encoder_3.add_adapter(t5_lora_config)  # ✅ Load T5 LoRA weights
            
            transformer_lora_state_dict = get_peft_model_state_dict(
                accelerator.unwrap_model(transformer)
            )
            
            if args.train_text_encoder:
                text_encoder_lora_state_dict = get_peft_model_state_dict(
                    accelerator.unwrap_model(text_encoder_one)
                )
                text_encoder_2_lora_state_dict = get_peft_model_state_dict(
                    accelerator.unwrap_model(text_encoder_two)
                )
                text_encoder_3_lora_state_dict = get_peft_model_state_dict(
                    accelerator.unwrap_model(text_encoder_three)
                )
            
            set_peft_model_state_dict(pipeline.transformer, transformer_lora_state_dict)
            if args.train_text_encoder:
                set_peft_model_state_dict(pipeline.text_encoder, text_encoder_lora_state_dict)
                set_peft_model_state_dict(pipeline.text_encoder_2, text_encoder_2_lora_state_dict)
                set_peft_model_state_dict(pipeline.text_encoder_3, text_encoder_3_lora_state_dict)  

            _ = log_validation(pipeline, args, accelerator, epoch, is_final_validation=False)

            del pipeline, transformer_lora_state_dict
            if args.train_text_encoder:
                del text_encoder_lora_state_dict, text_encoder_2_lora_state_dict
            free_memory()
            logger.info("Finished validation.")
            # ═════════ END VALIDATION ═════════

    # ═══════════════════════════════════════════════════════════
    # Final model saving and validation
    # ═══════════════════════════════════════════════════════════
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Moving models to CPU for saving...")

        transformer_unwrapped = (
            accelerator.unwrap_model(transformer).to("cpu").to(torch.float32)
        )
        transformer_lora_layers = get_peft_model_state_dict(transformer_unwrapped)

        if args.train_text_encoder:
            text_encoder_one_unwrapped = (
                accelerator.unwrap_model(text_encoder_one).to("cpu").to(torch.float32)
            )
            text_encoder_two_unwrapped = (
                accelerator.unwrap_model(text_encoder_two).to("cpu").to(torch.float32)
            )
            text_encoder_3_unwrapped = (
                accelerator.unwrap_model(text_encoder_three).to("cpu").to(torch.float32)
            )
            text_encoder_lora_layers = get_peft_model_state_dict(
                text_encoder_one_unwrapped
            )
            text_encoder_2_lora_layers = get_peft_model_state_dict(
                text_encoder_two_unwrapped
            )
            text_encoder_3_lora_layers = get_peft_model_state_dict(
                text_encoder_3_unwrapped
            )

        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None
            text_encoder_3_lora_layers = None

        # Save LoRA weights using Diffusers format
        StableDiffusion3Pipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
            # text_encoder_3_lora_layers=text_encoder_3_lora_layers,  # ✅ Save T5 LoRA
        )

        # Run final validation if specified
        if args.validation_prompt is not None and args.num_validation_images > 0:
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
            pipeline.scheduler = noise_scheduler
            pipeline.load_lora_weights(args.output_dir)

            images = log_validation(pipeline, args, accelerator, epoch, is_final_validation=False)

            # Save validation images to disk
            validation_save_dir = os.path.join(args.output_dir, "validation_images")
            os.makedirs(validation_save_dir, exist_ok=True)
            
            for i, img in enumerate(images):
                image_filename = f"{args.validation_prompt}_{epoch}-{i}.png"
                print(f"saving validation image: {image_filename}")
                img.save(
                    os.path.join(
                        validation_save_dir, image_filename
                    )
                )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
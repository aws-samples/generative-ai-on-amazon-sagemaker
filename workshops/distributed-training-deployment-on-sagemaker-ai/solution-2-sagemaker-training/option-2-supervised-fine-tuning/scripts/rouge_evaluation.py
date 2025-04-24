import argparse
from datasets import Dataset, load_dataset
from dataclasses import dataclass
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from transformers import GenerationConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from typing import Dict, List
import xtarfile as tarfile

SYSTEM_PREFIX: str = "<｜begin▁of▁sentence｜>"
SYSTEM_PROMPT = """
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.
"""
USER_PREFIX: str = "<｜User｜>"
ASSISTANT_PREFIX: str = "<｜Assistant｜>"
SYSTEM_SUFFIX: str = "<｜end▁of▁sentence｜>"

QUESTION_COLUMNS = ["Question"]
TARGET_COLUMNS = ["Complex_CoT", "Response"]
JOINER_TARGET_COLUMNS = "\n\n"


def parse_arge():

    parser = argparse.ArgumentParser()

    # infra configuration
    parser.add_argument(
        "--adapterdir", type=str, default=os.environ["SM_CHANNEL_ADAPTER"]
    )

    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="FreedomIntelligence/medical-o1-reasoning-SFT",
    )

    args = parser.parse_known_args()

    return args


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


@dataclass
class DatasetConfig:
    """
    Configuration class for dataset processing.

    Attributes:
        dataset_name (str): Name of the dataset to use
        question_columns (str): List of column names containing questions and/or context
        target_columns (List[str]): List of column names containing target responses
        prompt_template (str): Template string for formatting prompts
        user_prefix (str): Prefix for user messages
        assistant_prefix (str): Prefix for assistant responses
        system_prefix (str): Prefix for system messages
        system_prefix (str): Suffix for system messages

    Raises:
        ValueError: If target_columns is empty
    """

    dataset_name: str
    question_columns: List[str]
    target_columns: List[str]
    prompt_template: str
    user_prefix: str = ""
    assistant_prefix: str = ""
    system_prefix: str = ""
    system_suffix: str = ""

    def __post_init__(self):
        if len(self.target_columns) < 1:
            raise ValueError("Must provide at least one target column")


def generate_prompt(question: str, target: str, config: DatasetConfig):
    """
    Generates a prompt based on configuration

    Args:
        question: The input question
        target: The target response
        config: DatasetConfig object containing prompt template
    """
    full_prompt = f"""{config.system_prefix}
    {config.prompt_template}
    {config.user_prefix}
    {question}
    {config.assistant_prefix}
    """

    return {"prompt": full_prompt.strip(), "human_baseline": target.strip()}


### Change this part with a different dataset
def create_prompt_function(config: DatasetConfig):
    """
    Creates a prompt function based on dataset configuration

    Args:
        config: DatasetConfig object containing dataset-specific parameters
    """

    def wrapper(row):
        # Combine target columns as specified in config
        questions = [row[col] for col in config.question_columns]
        question_text = JOINER_TARGET_COLUMNS.join(questions)
        targets = [row[col] for col in config.target_columns]
        target_text = JOINER_TARGET_COLUMNS.join(targets)

        return generate_prompt(question_text, target_text, config)

    return wrapper


def calculate_metrics(model, test_dataset_response, config, desc, max_new_tokens=500):
    """
    Calculate ROUGE metrics for model-generated responses against human baselines.

    Args:
        model: The transformer model to evaluate
        test_dataset: Dataset containing test examples
        config (DatasetConfig): Configuration for dataset processing
        desc (str): Description string for logging purposes
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 500.

    Prints:
        ROUGE metric scores for the model's performance
    """
    print(f"Printing datasetname:{args.dataset_name}")

    human_baseline_lengths = [
        len(example["human_baseline"]) for example in test_dataset_response
    ]
    avg_baseline_length = int(sum(human_baseline_lengths) / len(human_baseline_lengths))
    print(f"Average human baseline length: {avg_baseline_length} characters")

    # Use the adaptive approach that will choose the best strategy based on dataset size
    model_summaries_response = get_summaries(
        model, test_dataset_response, config, avg_baseline_length
    )
    human_baseline_summaries = test_dataset_response["human_baseline"]

    import evaluate

    rouge = evaluate.load("rouge")

    model_results_response = rouge.compute(
        predictions=model_summaries_response,
        references=human_baseline_summaries[0 : len(model_summaries_response)],
        use_aggregator=True,
    )

    print(f"{desc}: \n{model_results_response}\n")


def get_summaries(model, test_dataset, config, max_new_tokens=500):
    """
    Generate text summaries using the provided model for each example in the test dataset.

    Args:
        model: The transformer model to use for generation
        test_dataset: Dataset containing prompts to generate summaries for
        config (DatasetConfig): configuration for the provided dataset
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 500.

    Returns:
        list: List of generated text summaries
    """

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model_summaries = []

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    index = 1
    for _, dialogue in enumerate(test_dataset):
        print("Dialogue prompt:", index)

        input_ids = tokenizer(dialogue["prompt"], return_tensors="pt").input_ids.to(
            device
        )
        with torch.no_grad():
            model_outputs = model.generate(
                input_ids=input_ids,
                generation_config=GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    use_cache=True,
                ),
            )

        original_model_text_output = tokenizer.decode(
            model_outputs[0], skip_special_tokens=True
        )
        if config.system_suffix is not None and config.system_suffix != "":
            original_model_text_output = original_model_text_output.split(
                config.system_suffix
            )[0]

        model_summaries.append(original_model_text_output)

        index += 1

    return model_summaries


def uncompress_full_model(model_dir):
    """
    Uncompresses a .tar.gz file from the specified directory.

    Args:
        model_dir (str): Directory path containing the .tar.gz file

    Returns:
        bool: True if successful, False if no .tar.gz file found

    Raises:
        Exception: If there are multiple .tar.gz files or extraction fails
    """
    # Convert to Path object for easier handling
    model_path = Path(model_dir)

    # Find all .tar.gz files in the directory
    tar_files = list(model_path.glob("*.tar.gz"))

    # Check if we found any .tar.gz files
    if not tar_files:
        print(f"No .tar.gz files found in {model_dir}")
        return False

    # Check if we have multiple .tar.gz files
    if len(tar_files) > 1:
        raise Exception(
            f"Multiple .tar.gz files found in {model_dir}. Expected only one: {tar_files}"
        )

    # Get the single .tar.gz file
    tar_file = tar_files[0]

    try:
        # Extract the contents
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path=model_dir)

        # Remove the original tar file from the temporary folder if it was copied there
        temp_tar = Path(model_dir) / tar_file.name
        if temp_tar.exists():
            temp_tar.unlink()

        print(f"Successfully extracted {tar_file.name}")
        return True

    except Exception as e:
        print(f"Error extracting {tar_file.name}: {str(e)}")
        raise


def main(args, test_dataset, dataset_config):
    """
    Main function to run the model evaluation pipeline.

    Args:
        args: Parsed command line arguments
        test_dataset: Dataset containing test examples
        dataset_config (DatasetConfig): Configuration for dataset processing

    Executes:
        - Loads base and fine-tuned models
        - Generates sample outputs
        - Calculates and prints evaluation metrics
    """
    print(f"Printing datasetname:{args.dataset_name}")

    test_dataset = Dataset.from_pandas(test_dataset)
    columns_to_remove = list(test_dataset.features)

    test_dataset_response = test_dataset.map(
        create_prompt_function(dataset_config),
        remove_columns=columns_to_remove,
        batched=False,
    )

    test_dataset_response = test_dataset_response.select(
        range(min(max_samples, len(test_dataset_response)))
    )

    print("Loading base model")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    print(f"\n\n\n*** Generating Metrics on Base Model:")
    calculate_metrics(base_model, test_dataset_response, dataset_config, "Base Model")

    # Clear cache
    del base_model
    torch.cuda.empty_cache()

    # Load the adapter
    print("Loading fine-tuned model")
    uncompress_full_model(args.adapterdir)

    model = AutoModelForCausalLM.from_pretrained(
        args.adapterdir,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    print("Loaded fine-tuned model")

    print(f"\n\n\n*** Generating Metrics on Trained Model:")
    calculate_metrics(model, test_dataset_response, dataset_config, "Trained Model")


if __name__ == "__main__":
    args, _ = parse_arge()

    custom_env: Dict[str, str] = {
        "HF_DATASETS_TRUST_REMOTE_CODE": "TRUE",
        "HF_TOKEN": args.hf_token,
    }

    set_custom_env(custom_env)

    ### Change this part with a different dataset ###
    print("Loading dataset")
    dataset = load_dataset(args.dataset_name, "en")
    df = pd.DataFrame(dataset["train"])
    df = df[:1000]
    train, test = train_test_split(df, test_size=0.1, random_state=42)
    print("Loaded dataset")
    ################################################

    # Define dataset configuration
    dataset_config = DatasetConfig(
        dataset_name=args.dataset_name,
        question_columns=QUESTION_COLUMNS,
        target_columns=TARGET_COLUMNS,
        prompt_template=SYSTEM_PROMPT,
        user_prefix=USER_PREFIX,
        assistant_prefix=ASSISTANT_PREFIX,
        system_prefix=SYSTEM_PREFIX,
        system_suffix=SYSTEM_SUFFIX,
    )

    # launch training
    main(args, test, dataset_config)

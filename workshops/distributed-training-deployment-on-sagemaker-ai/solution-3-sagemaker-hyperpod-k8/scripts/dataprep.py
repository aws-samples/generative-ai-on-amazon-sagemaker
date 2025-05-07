from datasets import load_dataset, Dataset, DatasetDict
import os
import pandas as pd
from random import randint
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

DATASET_NAME = "NousResearch/hermes-function-calling-v1"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
HF_TOKEN = ""


def read_dataset(dataset_name):
    dataset = load_dataset(
        dataset_name, data_files={"train": ["json-mode-agentic.json"]}
    )

    df = pd.DataFrame(dataset["train"])

    train, test = train_test_split(df, test_size=0.1, random_state=42)

    return train, test


def prompt_format(df):
    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)

    for index, el in df.iterrows():
        chat = tokenizer.apply_chat_template(el["conversations"], tokenize=False)

    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    train_dataset = dataset["train"].map(
        template_dataset, remove_columns=list(dataset["train"].features)
    )

    test_dataset = dataset["test"].map(
        template_dataset, remove_columns=list(dataset["test"].features)
    )

    return train_dataset, test_dataset


def transform_conversation(conversation):
    transformed = []
    for msg in conversation:
        # Create a new dictionary with the renamed keys
        new_msg = {
            "role": (
                "user"
                if msg["from"] == "human"
                else "assistant" if msg["from"] == "gpt" else "system"
            ),
            "content": msg["value"],
        }
        transformed.append(new_msg)
    return transformed


if __name__ == "__main__":
    if HF_TOKEN != "":
        os.environ.update({"HF_TOKEN": HF_TOKEN})

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    parent_dir = os.path.dirname(script_dir)
    parent_dir = f"/mnt/custom-file-systems/{'/'.join(parent_dir.split('/')[4:])}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    train, test = read_dataset(DATASET_NAME)

    train["conversations"] = train["conversations"].apply(transform_conversation)
    test["conversations"] = test["conversations"].apply(transform_conversation)

    train["text"] = train["conversations"].apply(
        lambda x: tokenizer.apply_chat_template(x, tokenize=False)
    )
    test["text"] = test["conversations"].apply(
        lambda x: tokenizer.apply_chat_template(x, tokenize=False)
    )

    train = train[["text"]]
    test = test[["text"]]

    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)

    train_dataset.to_json(f"{parent_dir}/data/train/dataset.json", orient="records")
    test_dataset.to_json(f"{parent_dir}/data/test/dataset.json", orient="records")

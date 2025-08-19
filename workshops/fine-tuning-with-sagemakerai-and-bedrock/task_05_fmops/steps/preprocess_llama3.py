# Temporary preprocess step (to be changed with new dataset)
import boto3
import pandas as pd
from datasets import load_dataset
from datasets import Dataset
from random import randint
import mlflow
import json


system_message = """You are Llama, an AI assistant. Your knowledge spans a wide range of topics, allowing you to anser the questions with honesty and truthfulness."""

def create_conversation(sample):
    if sample["messages"][0]["role"] == "system":
        return sample
    else:
      sample["messages"] = [{"role": "system", "content": system_message}] + sample["messages"]
      return sample

def preprocess(s3_bucket, dataset_name, train_sample, eval_sample, mlflow_arn, experiment_name, run_name):

    mlflow.set_tracking_uri(mlflow_arn)
    mlflow.set_experiment(experiment_name)

    
    # This is a very simple example, you can add your own data processing code here
    dataset = load_dataset(dataset_name)
    dataset = dataset.filter(lambda x: x['category'] == 'Open QA')

    columns_to_remove = list(dataset["train"].features)
    columns_to_remove.remove("messages")
    dataset = dataset.map(create_conversation, remove_columns=columns_to_remove,batched=False)

    dataset["train"] = dataset["train"].filter(lambda x: len(x["messages"][1:]) % 2 == 0)
    dataset["test"] = dataset["test"].filter(lambda x: len(x["messages"][1:]) % 2 == 0)

    dataset["train"].to_json("train_dataset.json", orient="records", force_ascii=False)
    dataset["test"].to_json("test_dataset.json", orient="records", force_ascii=False)

    # save training and test data to s3
    s3 = boto3.client("s3")
    s3.upload_file("train_dataset.json", s3_bucket, f"dataset/{dataset_name}/{train_sample}/train/train_dataset.json")
    s3.upload_file("test_dataset.json", s3_bucket, f"dataset/{dataset_name}/{eval_sample}/eval/eval_dataset.json")


    training_input_path = f's3://{s3_bucket}/dataset/{dataset_name}/{train_sample}/train/train_dataset.json'
    eval_input_path = f's3://{s3_bucket}/dataset/{dataset_name}/{eval_sample}/eval/eval_dataset.json'

    with mlflow.start_run(run_name=run_name) as run:
        
        run_id = run.info.run_id
        print(run_id)

        # create pandas dataframe from train json
        df_train = pd.read_json("train_dataset.json", orient="records", lines=True)
        df_evaluate = pd.read_json("test_dataset.json", orient="records", lines=True)

        training_data = mlflow.data.from_pandas(df_train, source=training_input_path)
        mlflow.log_input(training_data, context="training")

        evaluation_data = mlflow.data.from_pandas(df_evaluate, source=eval_input_path)
        mlflow.log_input(evaluation_data, context="evaluation")

    return {"training_input_path": training_input_path, "eval_input_path": eval_input_path, "run_id": run_id}

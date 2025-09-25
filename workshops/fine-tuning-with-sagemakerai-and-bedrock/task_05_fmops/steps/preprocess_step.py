# **Preprocessing Step**

# This step handles data preparation. We are going to prepare data for training and evaluation. We will log this data in MLflow
import boto3
import shutil
import sagemaker
import os
import pandas as pd
from sagemaker.config import load_sagemaker_config
import mlflow
import traceback
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from random import randint
from sagemaker.workflow.function_step import step
from .pipeline_utils import (
    PIPELINE_INSTANCE_TYPE,
    template_dataset
)


@step(
    name="DataPreprocessing",
    instance_type=PIPELINE_INSTANCE_TYPE,
    display_name="Data Preprocessing",
    keep_alive_period_in_seconds=900
)
def preprocess(
    tracking_server_arn: str,
    input_path: str,
    experiment_name: str,
    run_name: str,
) -> tuple:
    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)

    # Preprocessing code
    try:
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            with mlflow.start_run(run_name="Processing", nested=True):
                # Initialize SageMaker and S3 clients
                sagemaker_session = sagemaker.Session()
                s3_client = boto3.client('s3')

                bucket_name = sagemaker_session.default_bucket()
                default_prefix = sagemaker_session.default_bucket_prefix
                configs = load_sagemaker_config()

                # Set paths
                if default_prefix:
                    input_path = f'{default_prefix}/datasets/llm-fine-tuning-modeltrainer-sft'
                else:
                    input_path = f'datasets/llm-fine-tuning-modeltrainer-sft'

                # Load dataset with proper error handling
                sample_dataset_size = 100
                try:
                    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")
                except Exception as e:
                    error_msg = f"Error loading dataset: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg)
                    raise RuntimeError(f"Failed to load dataset: {str(e)}")

                df = pd.DataFrame(dataset['train'])
                df = df[:sample_dataset_size]

                # Split dataset
                train, test = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

                print("Number of train elements: ", len(train))
                print("Number of test elements: ", len(test))

                # Log dataset statistics if MLflow is enabled
                mlflow.log_param("dataset_source", "FreedomIntelligence/medical-o1-reasoning-SFT")
                mlflow.log_param("train_size", len(train))
                mlflow.log_param("test_size", len(test))
                mlflow.log_param("dataset_sample_size", sample_dataset_size)  # Log that we're using a subset of 100 samples

                # Create datasets
                train_dataset = Dataset.from_pandas(train)
                test_dataset = Dataset.from_pandas(test)
                dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
                train_dataset = dataset["train"].map(template_dataset, remove_columns=list(dataset["train"].features))
                test_dataset = dataset["test"].map(template_dataset, remove_columns=list(dataset["test"].features))

                # Safely get a sample text, handling potential index errors
                try:
                    sample_index = randint(0, len(train_dataset) - 1)
                    sample_text = train_dataset[sample_index]["text"]
                    print(f"Sample text from index {sample_index}:")
                    print(sample_text)
                except (IndexError, KeyError) as e:
                    sample_text = "Error retrieving sample text: " + str(e)
                    print(sample_text)               

                # Create directories with error handling
                try:
                    os.makedirs("./data/train", exist_ok=True)
                    os.makedirs("./data/test", exist_ok=True)
                except OSError as e:
                    error_msg = f"Error creating directories: {str(e)}"
                    print(error_msg)

                # Save datasets locally with error handling
                try:
                    train_dataset.to_json("./data/train/dataset.json", orient="records")
                    test_dataset.to_json("./data/test/dataset.json", orient="records")
                except Exception as e:
                    error_msg = f"Error saving datasets locally: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg)
                    raise RuntimeError(f"Failed to save datasets locally: {str(e)}")

                # Define S3 paths
                train_data_path = f"s3://{bucket_name}/{input_path}/train/dataset.json"
                test_dataset_path = f"s3://{bucket_name}/{input_path}/test/dataset.json"

                # Store results for return
                result_train_data_path = train_data_path
                result_test_dataset_path = test_dataset_path

                # Log dataset paths if MLflow is enabled
                mlflow.log_param("train_data_path", train_data_path)
                mlflow.log_param("test_dataset_path", test_dataset_path)

                # Upload files to S3 with retries
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        print(f"Uploading train dataset to S3, attempt {attempt+1}/{max_retries}")
                        s3_client.upload_file("./data/train/dataset.json", bucket_name, f"{input_path}/train/dataset.json")
                        print(f"Uploading test dataset to S3, attempt {attempt+1}/{max_retries}")
                        s3_client.upload_file("./data/test/dataset.json", bucket_name, f"{input_path}/test/dataset.json")
                        print("S3 upload successful")
                        break
                    except Exception as e:
                        error_msg = f"Error in S3 upload (attempt {attempt+1}/{max_retries}): {str(e)}"
                        print(error_msg)
                        if attempt == max_retries - 1:  # Last attempt failed
                            raise RuntimeError(f"Failed to upload datasets to S3 after {max_retries} attempts: {str(e)}")

                print(f"Datasets uploaded to:")
                print(train_data_path)
                print(test_dataset_path)

                # Log a sample of the dataset as an artifact if MLflow is enabled
                try:
                    with open("./data/sample.txt", "w") as f:
                        f.write(sample_text)
                    mlflow.log_artifact("./data/sample.txt", "dataset_samples")
                except Exception as e:
                    print(f"Error logging sample as artifact: {str(e)}")

                # Clean up
                try:
                    if os.path.exists("./data"):
                        shutil.rmtree("./data")
                except Exception as e:
                    print(f"Warning: Error cleaning up temporary files: {str(e)}")

    except Exception as e:
        error_msg = f"Critical error in preprocessing: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise RuntimeError(f"Preprocessing failed: {str(e)}")

    return run_id, result_train_data_path, result_test_dataset_path
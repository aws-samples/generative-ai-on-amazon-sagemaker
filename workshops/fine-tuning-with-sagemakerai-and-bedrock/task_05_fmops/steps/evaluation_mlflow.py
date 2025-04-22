import boto3
import sagemaker
from sagemaker.s3_utils import parse_s3_url
import mlflow
import tempfile
from pathlib import Path
import pandas as pd
import json 
from dataclasses import dataclass
from typing import Tuple, Optional
import json


def evaluation(model, preprocess_step_ret, finetune_ret, mlflow_arn, experiment_name, run_id):
    mlflow.set_tracking_uri(mlflow_arn)
    mlflow.set_experiment(experiment_name)

    print(preprocess_step_ret['run_id'])

    with mlflow.start_run(run_id=preprocess_step_ret['run_id']) as run:
        s3 = boto3.client("s3")
        sess = sagemaker.Session()

        dataset_info = mlflow.get_run(preprocess_step_ret['run_id']).inputs.dataset_inputs[1].dataset

        print(dataset_info)
        print(f"Dataset name: {dataset_info.name}")
        print(f"Dataset digest: {dataset_info.digest}")
        print(f"Dataset profile: {dataset_info.profile}")
        print(f"Dataset schema: {dataset_info.schema}")

        dataset_source = mlflow.data.get_source(dataset_info)
        ds = dataset_source.load()
        # get the bucket name using full s3 poth

        eval_data=pd.read_json(ds, orient='records', lines=True)

        data = []
        for index, row in eval_data.iterrows():
            for message in row['messages']:
                if message["role"] == "user":
                    question = message["content"]
                elif message["role"] == "assistant":
                    answer = message["content"]
            data.append({"question": question, "answer": answer})

        df = pd.DataFrame(data, columns=["question", "answer"])
        print(df.head())

        
        logged_model = f"runs:/{preprocess_step_ret['run_id']}/model"
        loaded_model = mlflow.pyfunc.load_model(model_uri=logged_model)
        results = mlflow.evaluate(
            model=loaded_model,
            data=df,
            targets="answer",
            model_type="question-answering",
            evaluator_config={"col_mapping": {"inputs": "question"}},
        )
        print(results.metrics)
    return "done"
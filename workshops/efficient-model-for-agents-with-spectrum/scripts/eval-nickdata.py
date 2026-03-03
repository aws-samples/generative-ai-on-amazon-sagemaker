
import os
import torch
import mlflow
import argparse
import pandas as pd
from tqdm import tqdm
import mlflow.sagemaker
from datetime import datetime

from datasets import Dataset, load_dataset


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
    TextStreamer
)
from peft import PeftModel, PeftConfig 

from mlflow.metrics.genai import (
    EvaluationExample,
    answer_correctness,
    answer_similarity,
)
from mlflow.metrics import (
    bleu,
    rouge1,
    rouge2,
    rougeL,
    rougeLsum,
    latency,
    token_count,
)

""" how to run: python scripts/eval.py     --dataset "gretelai/synthetic_text_to_sql"     --model_name_or_path temp/extracted_model/spectrum-Qwen-3-0.6B-txt-to-sql/     --model_version Qwen3-0.6B-spectrum """
#api_key = os.getenv("OPENAI_API_KEY")

def load_model(model_name):
        print(f"Loading base model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer


def predict(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, sample: dict):
        system_prompt = """You are an expert SQL developer. Given the provided database schema and the following user question, generate a syntactically correct SQL query. 
			Only reply with the SQL query, nothing else. Do NOT use the backticks to identify the code, just reply with the pure SQL query."""
        #user_prompt = f"Schema:\n{sample['sql_context']}\n\nUser Query:\n{sample['sql_prompt']}\n\nSQL Query:\n"
        user_prompt = f"Schema:\n CREATE TABLE orders (row_id BIGINT, order_id STRING, order_date TIMESTAMP, ship_date TIMESTAMP, ship_mode STRING, customer_id STRING, customer_name STRING, segment STRING, city STRING, state STRING, country STRING, postal_code FLOAT, market STRING, region STRING, product_id STRING, category STRING, sub-category STRING, product_name STRING, sales FLOAT, quantity BIGINT, discount FLOAT, profit FLOAT, shipping_cost FLOAT, order_priority STRING); \nCREATE EXTERNAL TABLE returns (returned STRING, order_id STRING, market STRING)\n\nUser Query:\n{sample['question']}\n\nSQL Query:\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        streamer = TextStreamer(tokenizer)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            # streamer=streamer
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return text, response
def evaluate(dataset):
    raw_prompts = []
    predictions = []
    for sample in tqdm(dataset):
        raw_prompt, prediction = predict(tokenizer, model, sample)
        
        raw_prompts.append(raw_prompt)
        predictions.append(prediction)
        #print(prediction)
    dataset = dataset.add_column("predictions", predictions)
    if args.report_to_mlflow:
        print('Logging predictions to mlflow')
        mlflow.set_tracking_uri(
            'arn:aws:sagemaker:us-east-1:783764584149:mlflow-tracking-server/MLflow3-test'
        )
        mlflow.set_experiment("txt-to-sql-eval")
        csv_path = './tmp/predictions.csv'
        dataset.to_csv(csv_path, index=False)
        with mlflow.start_run(run_name=f'{args.model_version}-predictions'):
            mlflow.log_artifact(csv_path)
        shutil.rmtree('./tmp')
        mlflow.end_run()
    return dataset

    
def mlflow_eval(eval_df: pd.DataFrame, model_version: str):
    print('Evaluating the model with LLM as a judge')

    mlflow.set_tracking_uri(
        'arn:aws:sagemaker:us-east-1:783764584149:mlflow-tracking-server/MLflow3-test'
    )
    mlflow.sagemaker
    mlflow.set_experiment("txt-to-sql-eval")


    def predict(df):
        return df['predictions'].to_list()

    with mlflow.start_run(run_name=f'{model_version}-evaluation'):
        base_results = mlflow.evaluate(
            predict,
            eval_df,
            evaluators="default",
            targets="sql_query",
            extra_metrics=[
                bleu(),
                rouge1(),
                rouge2(),
                rougeL(),
                rougeLsum(),
            ],
        )

    mlflow.end_run()

    return base_results



if __name__ == "__main__":

    # Parse command line arguments
    help = '''
    This script evaluates the model.

    '''
    parser = argparse.ArgumentParser(
        description="eval llm ", epilog=help
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Dataset name or path for evaluation",
    )
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        required=True,
        help="Name of the model to use for evaluation",
    )

    parser.add_argument(
        "-mv",
        "--model_version",
        type=str,
        required=True,
        help="Model version",
    )

   # report to mlflow
    parser.add_argument(
        "--report_to_mlflow",
        action="store_true",
        help="Log to mlflow",
    )

    args = parser.parse_args()
    #test_dataset=load_dataset(args.dataset, split="test")
    test_dataset = load_dataset(
            'json', data_files=os.path.join(args.dataset), split='train'
        )
    # if len(test_dataset) > 10:
    #     test_dataset = test_dataset.select(range(10))
    model, tokenizer = load_model(args.model_name_or_path)
    
    predictions = evaluate(test_dataset)

    model_version = args.model_version
    report_to_mlflow = args.report_to_mlflow

    predictions = predictions.rename_column('question', 'inputs')
    print(predictions)
    results = mlflow_eval(
        predictions.to_pandas(),
        model_version=args.model_version,
    )
    print(results.metrics)
    eval_dict = {
        "model": model_version,
        "bleu": results.metrics['bleu/v1/mean'],
        "rouge1": results.metrics['rouge1/v1/mean'],
        "rouge2": results.metrics['rouge2/v1/mean'],
        "rougeL": results.metrics['rougeL/v1/mean'],
        "rougeLsum": results.metrics['rougeLsum/v1/mean']
    }
    results_df = pd.DataFrame([eval_dict])
    results_df.set_index("model", inplace=True)

    if not os.path.exists("eval/results"):
        os.makedirs("eval/results")

    tstamp = datetime.now().strftime("%m%d%Y-%H%M")
    results_df.to_csv(
        f"eval/results/{args.model_version}-results-{tstamp}.csv",
        index=True,
        header=True,
    )

    print(
        'Results saved to:',
        f"eval/results/{args.model_version}-results-{tstamp}.csv",
    )
    print('Results:', eval_dict)

    if report_to_mlflow:
        mlflow.set_tracking_uri(
            'arn:aws:sagemaker:us-east-1:783764584149:mlflow-tracking-server/test'
        )
        mlflow.set_experiment("amlc-autoinstruct-eval")

        with mlflow.start_run(run_name=f'{model_version}-results'):
            eval_dict.pop('model')
            mlflow.log_metrics(eval_dict)
        mlflow.end_run()
    print(test_dataset)
    #test_dataset.to_json("./data/prediction/dataset.json", orient="records")
    
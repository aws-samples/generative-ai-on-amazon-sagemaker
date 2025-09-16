import boto3
import botocore
import json
import time
from datetime import datetime


PIPELINE_INSTANCE_TYPE = "ml.m5.xlarge"


PROMPT_TEMPLATE = f"""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{{question}}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{{complex_cot}}

{{answer}}
<|eot_id|>
"""


def endpoint_exists(endpoint_name):
    endpoint_exist = False

    client = boto3.client('sagemaker')
    response = client.list_endpoints()
    endpoints = response["Endpoints"]

    for endpoint in endpoints:
        if endpoint_name == endpoint["EndpointName"]:
            endpoint_exist = True
            break

    return endpoint_exist


def create_training_job_name(model_id):
    return f"{model_id}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]}"


# Template dataset to add prompt to each sample
def template_dataset(sample):
    try:
        sample["text"] = PROMPT_TEMPLATE.format(question=sample["Question"],
                                                complex_cot=sample["Complex_CoT"],
                                                answer=sample["Response"])
        return sample
    except KeyError as e:
        print(f"KeyError in template_dataset: {str(e)}")
        # Provide default values for missing fields
        missing_key = str(e).strip("'")
        if missing_key == "Question":
            sample["text"] = PROMPT_TEMPLATE.format(
                question="[Missing question]",
                complex_cot=sample.get("Complex_CoT", "[Missing CoT]"),
                answer=sample.get("Response", "[Missing response]")
            )
        elif missing_key == "Complex_CoT":
            sample["text"] = PROMPT_TEMPLATE.format(
                question=sample["Question"],
                complex_cot="[Missing CoT]",
                answer=sample.get("Response", "[Missing response]")
            )
        elif missing_key == "Response":
            sample["text"] = PROMPT_TEMPLATE.format(
                question=sample["Question"],
                complex_cot=sample.get("Complex_CoT", "[Missing CoT]"),
                answer="[Missing response]"
            )
        return sample


def invoke_sagemaker_endpoint(payload, endpoint_name):
    """
    Invoke a SageMaker endpoint with the given payload.

    Args:
        payload (dict): The input data to send to the endpoint
        endpoint_name (str): The name of the SageMaker endpoint

    Returns:
        dict: The response from the endpoint
    """
    sm_client = boto3.client('sagemaker-runtime')
    try:
        start_time = time.time()
        response = sm_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        inference_time = time.time() - start_time
        
        response_body = response['Body'].read().decode('utf-8')
        return json.loads(response_body), inference_time
    except Exception as e:
        print(f"Error invoking endpoint {endpoint_name}: {str(e)}")
        return None, -1
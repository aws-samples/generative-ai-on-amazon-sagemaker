import boto3
from datetime import datetime


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

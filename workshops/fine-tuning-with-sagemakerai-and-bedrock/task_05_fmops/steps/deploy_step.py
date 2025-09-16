# ### 6. Deploy Step
# This step deploys the model for evaluation

import sagemaker
import boto3
from sagemaker import get_execution_role
from sagemaker import Model
from sagemaker.model_monitor import DataCaptureConfig
import time
from sagemaker.workflow.function_step import step
from .pipeline_utils import PIPELINE_INSTANCE_TYPE


@step(
    name="ModelDeploy",
    instance_type=PIPELINE_INSTANCE_TYPE,
    display_name="Model Deploy",
    keep_alive_period_in_seconds=900
)
def deploy(
    model_artifacts_s3_path: str,
    output_path: str,
    model_id: str,
):    
    sagemaker_session = sagemaker.Session()
    instance_count = 1
    instance_type = "ml.g5.2xlarge"
    health_check_timeout = 700
    
    # Get the name for the endpoint
    endpoint_name = f"{model_id.split('/')[-1].replace('.', '-').replace('_','-')}-sft-djl"
    
    # Delete existing endpoint if it exists
    print(f"Checking for existing endpoint: {endpoint_name}")
    sm_client = boto3.client('sagemaker')
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint {endpoint_name} exists, deleting it before deployment")
        sm_client.delete_endpoint(EndpointName=endpoint_name)

        print(f"Deleting endpoint config {endpoint_name}")
        sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        
        # Wait for endpoint to be fully deleted
        print("Waiting for endpoint to be fully deleted...")
        wait_seconds = 10
        total_wait_time = 0
        max_wait_time = 300  # 5 minutes maximum wait
        endpoint_deleted = False
        
        while total_wait_time < max_wait_time and not endpoint_deleted:
            try:
                sm_client.describe_endpoint(EndpointName=endpoint_name)
                print(f"Endpoint still exists, waiting {wait_seconds} seconds...")
                time.sleep(wait_seconds)
                total_wait_time += wait_seconds
            except sm_client.exceptions.ClientError:
                print(f"Endpoint {endpoint_name} successfully deleted")
                endpoint_deleted = True
                
        if not endpoint_deleted:
            print(f"Warning: Endpoint still exists after {max_wait_time} seconds")
            
    except sm_client.exceptions.ClientError:
        print(f"Endpoint {endpoint_name} does not exist, proceeding with deployment")
    
    # Continue with model deployment
    image_uri = sagemaker.image_uris.retrieve(
        framework="djl-lmi",
        region=sagemaker_session.boto_session.region_name,
        version="latest"
    )
    
    model_data = model_artifacts_s3_path
    
    # Create model only once
    model = Model(
        image_uri=image_uri,
        model_data=model_data,
        role=get_execution_role(),
        env={
            'HF_MODEL_ID': "/opt/ml/model", # path to where sagemaker stores the model
            'OPTION_TRUST_REMOTE_CODE': 'true',
            'OPTION_ROLLING_BATCH': "vllm",
            'OPTION_DTYPE': 'bf16',
            'OPTION_QUANTIZE': 'fp8',
            'OPTION_TENSOR_PARALLEL_DEGREE': 'max',
            'OPTION_MAX_ROLLING_BATCH_SIZE': '32',
            'OPTION_MODEL_LOADING_TIMEOUT': '3600',
            'OPTION_MAX_MODEL_LEN': '4096'
        }
    )

    print(f"deploying endpoint: {endpoint_name}")

    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri='s3://sagemaker-us-east-1-329542461890/data-capture/',
        capture_options=["REQUEST", "RESPONSE"],
        csv_content_types=["text/csv"],
        json_content_types=["application/json"]
    )
    
    predictor = model.deploy(
        endpoint_name=endpoint_name,
        initial_instance_count=instance_count,
        instance_type=instance_type,
        container_startup_health_check_timeout=health_check_timeout,
        model_data_download_timeout=3600,
        data_capture_config=data_capture_config
    )
    
    return endpoint_name
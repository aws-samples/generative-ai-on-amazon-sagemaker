# **Fine-tuning Step**

# This is where the actual model adaptation occurs. The step takes the preprocessed data and applies it to fine-tune the base LLM (in this case, a Deepseek model). It incorporates the LoRA technique for efficient adaptation.

import sagemaker
import boto3
import mlflow
import yaml
import json
import time
import datetime
import os
import traceback
import tempfile
from pathlib import Path
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.function_step import step
from .pipeline_utils import PIPELINE_INSTANCE_TYPE

@step(
    name="ModelFineTuning",
    instance_type=PIPELINE_INSTANCE_TYPE,
    display_name="Model Fine Tuning",
    keep_alive_period_in_seconds=900,
    dependencies="./scripts/requirements.txt"
)
def train(
    tracking_server_arn: str,
    train_dataset_s3_path: str,
    test_dataset_s3_path: str,
    train_config_s3_path: str,
    role: str,
    experiment_name: str,
    model_id: str,
    run_id: str,
):


    # Initialize variables and tracking
    start_time = time.time()
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    training_job_name = None

    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)

    try:
        with mlflow.start_run(run_id=run_id):
            with mlflow.start_run(run_name="FinetuningStep", nested=True) as training_run:
                mlflow.autolog()
                training_run_id = training_run.info.run_id
                # Enable detailed tracking
                mlflow.set_tag("component", "model_fine_tuning")
                mlflow.log_param("model_id", model_id)
                mlflow.log_param("train_dataset", train_dataset_s3_path)
                mlflow.log_param("test_dataset", test_dataset_s3_path)
                mlflow.log_param("training_start_time", datetime.datetime.now().isoformat())

                # Download and parse the training config YAML to log hyperparameters
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    s3_client = boto3.client("s3")

                    # Parse S3 path
                    config_parts = train_config_s3_path.replace("s3://", "").split("/", 1)
                    bucket = config_parts[0]
                    key = config_parts[1]

                    # Download config file
                    try:
                        s3_client.download_file(bucket, key, tmp.name)
                        # Parse the YAML config
                        with open(tmp.name, 'r') as f:
                            config = yaml.safe_load(f)

                        # Log all hyperparameters from config
                        print("Logging hyperparameters to MLflow:")
                        for param_name, param_value in config.items():
                            # Skip complex objects that can't be logged as parameters
                            if isinstance(param_value, (str, int, float, bool)):
                                print(f"  {param_name}: {param_value}")
                                mlflow.log_param(param_name, param_value)
                            elif param_name == "fsdp_config" and isinstance(param_value, dict):
                                # Log nested config as JSON
                                mlflow.log_param("fsdp_config_json", json.dumps(param_value))

                        # Log file as artifact for reference
                        mlflow.log_artifact(tmp.name, "training_config")

                    except Exception as e:
                        print(f"Error parsing config file: {e}")

                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp.name):
                            os.remove(tmp.name)
            
                # Launch the training job
                job_name = f"deepseek-finetune-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
                
                sagemaker_session = sagemaker.Session()
                
                # Define metric definitions for more detailed CloudWatch metrics
                metric_definitions = [
                    {'Name': 'loss', 'Regex': "'loss':\\s*([0-9.]+)"},
                    {'Name': 'epoch', 'Regex': "'epoch':\\s*([0-9.]+)"},
                    {'Name': 'train_loss', 'Regex': "'train_loss':\\s*([0-9.]+)"},
                    {'Name': 'lr', 'Regex': "'learning_rate':\\s*([0-9.e-]+)"},
                    {'Name': 'step', 'Regex': "'step':\\s*([0-9.]+)"},
                    {'Name': 'samples_per_second', 'Regex': "'train_samples_per_second':\\s*([0-9.]+)"},
                ]
                
                # Log the metric definitions we're using
                mlflow.log_param("tracked_metrics", [m['Name'] for m in metric_definitions])
                
                pytorch_estimator = PyTorch(
                    entry_point='train.py',
                    source_dir="./scripts",
                    job_name=job_name,
                    base_job_name=job_name,
                    max_run=50000,
                    role=role,
                    framework_version="2.2.0",
                    py_version="py310",
                    instance_count=1,
                    instance_type="ml.p3.2xlarge",
                    sagemaker_session=sagemaker_session,
                    volume_size=50,
                    disable_output_compression=False,
                    keep_alive_period_in_seconds=1800,
                    distribution={"torch_distributed": {"enabled": True}},
                    hyperparameters={
                        "config": "/opt/ml/input/data/config/args.yaml"
                    },
                    metric_definitions=metric_definitions,
                    debugger_hook_config=False,
                    environment={"MLFLOW_RUN_ID": training_run_id}
                )
            
                # Define a data input dictionary with our uploaded S3 URIs
                data = {
                  'train': train_dataset_s3_path,
                  'test': test_dataset_s3_path,
                  'config': train_config_s3_path
                }
            
                print(f"Data for Training Run: {data}")
                
                # Log training job information
                mlflow.log_param("job_name", job_name)
                mlflow.log_param("instance_type", "ml.p3.2xlarge")
                
                # Start the training job
                pytorch_estimator.fit(data, wait=True)
            
                # Get information about the completed training job
                latest_run_job_name = pytorch_estimator.latest_training_job.job_name
                print(f"Latest Job Name: {latest_run_job_name}")
            
                sagemaker_client = boto3.client('sagemaker')
            
                # Describe the training job
                response = sagemaker_client.describe_training_job(TrainingJobName=latest_run_job_name)
            
                # Extract the model artifacts S3 path
                model_artifacts_s3_path = response['ModelArtifacts']['S3ModelArtifacts']
            
                # Extract the output path (this is the general output location)
                output_path = response['OutputDataConfig']['S3OutputPath']
                
                # Get training time metrics
                training_start_time = response.get('TrainingStartTime')
                training_end_time = response.get('TrainingEndTime')
                billable_time = response.get('BillableTimeInSeconds', 0)
                
                # Calculate duration
                total_training_time = 0
                if training_start_time and training_end_time:
                    total_training_time = (training_end_time - training_start_time).total_seconds()
                
                # Log job results and metrics to MLflow
                # Log basic job info
                mlflow.log_param("training_job_name", latest_run_job_name)
                mlflow.log_param("model_artifacts_path", model_artifacts_s3_path)
                mlflow.log_param("output_path", output_path)
                
                # Log performance metrics
                mlflow.log_metric("billable_time_seconds", billable_time)
                mlflow.log_metric("total_training_time_seconds", total_training_time)
                
                # Log training job status
                mlflow.log_param("training_job_status", response.get('TrainingJobStatus'))
                
                # Log any secondary status
                if 'SecondaryStatus' in response:
                    mlflow.log_param("secondary_status", response.get('SecondaryStatus'))
                
                # Log any failure reason
                if 'FailureReason' in response:
                    mlflow.log_param("failure_reason", response.get('FailureReason'))
                    
                # Get CloudWatch logs for the training job
                logs_client = boto3.client('logs')
                log_group = "/aws/sagemaker/TrainingJobs"
                log_stream = latest_run_job_name
                
                try:
                    # Get the last 1000 log events
                    log_events = logs_client.get_log_events(
                        logGroupName=log_group,
                        logStreamName=log_stream,
                        limit=1000
                    )
                    
                    # Extract and save logs
                    log_output = "\n".join([event['message'] for event in log_events['events']])
                    
                    # Save logs to file and log as artifact
                    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as tmp:
                        tmp.write(log_output)
                        log_file_path = tmp.name
                    
                    mlflow.log_artifact(log_file_path, "training_logs")
                    os.remove(log_file_path)
                    
                except Exception as e:
                    print(f"Error fetching training logs: {e}")
                
                # Log total execution time of this step
                step_duration = time.time() - start_time
                mlflow.log_metric("step_execution_time_seconds", step_duration)
                
                # Log model metadata
                mlflow.set_tag("model_path", model_artifacts_s3_path)
                mlflow.set_tag("training_completed_at", datetime.datetime.now().isoformat())
            
                print(f"Model artifacts S3 path: {model_artifacts_s3_path}")

    except Exception as e:
        error_msg = f"Error in model fine-tuning: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        raise RuntimeError(f"Fine-tuning failed: {str(e)}")

    return run_id, training_run_id, model_artifacts_s3_path, output_path
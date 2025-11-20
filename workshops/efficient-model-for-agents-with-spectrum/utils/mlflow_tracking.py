import boto3


def get_mlflow_arn():
    """Lists all MLflow Tracking Servers in your AWS account using Boto3."""
    client = boto3.client('sagemaker')
    try:
        response = client.list_mlflow_tracking_servers()
        tracking_servers = response.get('TrackingServerSummaries', [])
        return tracking_servers[0]['TrackingServerArn']
    except Exception as e:
        print(f"Error listing MLflow Tracking Servers: {e}")


def get_mlfow_url(tracking_server_name: str) -> str:
    # Initialize the SageMaker client
    sagemaker_client = boto3.client('sagemaker')
    try:
        # Describe the MLflow tracking server
        response = sagemaker_client.create_presigned_mlflow_tracking_server_url(
            TrackingServerName=tracking_server_name
        )
        # Extract the TrackingServerUrl from the response
        mlflow_ui_url = response.get('AuthorizedUrl')
        return f"[🔗️ Click here to open MLFlow 🧪]({mlflow_ui_url})"
    except sagemaker_client.exceptions.ResourceNotFoundException:
        print(f"MLflow Tracking Server '{tracking_server_name}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

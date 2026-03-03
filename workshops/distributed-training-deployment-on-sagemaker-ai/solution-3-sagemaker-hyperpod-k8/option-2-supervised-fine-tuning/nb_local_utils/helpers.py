import boto3
import mlflow
from IPython.display import display, Markdown, HTML
from mlflow import MlflowClient


def pretty_print_html(text):
    # Replace newline characters with <br> tags
    html_text = text.replace("\n", "<br>")
    # Apply HTML formatting
    html_formatted = f'<pre style="font-family: monospace; background-color: #f8f8f8; padding: 10px; border-radius: 5px; border: 1px solid #0077b6;">{html_text}</pre>'
    # Display the formatted HTML
    return HTML(html_formatted)


def get_mlflow_server_arn():
    sm_client = boto3.client("sagemaker")

    # Check legacy MLflow Tracking Servers
    tracking_servers = sm_client.list_mlflow_tracking_servers()[
        "TrackingServerSummaries"
    ]
    if tracking_servers:
        mlflow_arn = tracking_servers[0]["TrackingServerArn"]
        pretty_print_html(f"MLflow Server ARN: {mlflow_arn}")
        return mlflow_arn

    # Check MLflow Apps (serverless)
    mlflow_apps = sm_client.list_mlflow_apps()["Summaries"]
    if mlflow_apps:
        mlflow_arn = mlflow_apps[0]["Arn"]
        pretty_print_html(f"MLflow App ARN: {mlflow_arn}")
        return mlflow_arn

    pretty_print_html("No MLflow server or app found.")
    return None


def get_tracking_server_uri(sess, tracking_server_arn):
    """Get tracking server URI for both legacy tracking servers and MLflow Apps."""
    if "mlflow-tracking-server" in tracking_server_arn:
        response = sess.sagemaker_client.create_presigned_mlflow_tracking_server_url(
            TrackingServerName=tracking_server_arn.split("/")[-1]
        )
    else:
        response = sess.sagemaker_client.create_presigned_mlflow_app_url(
            Arn=tracking_server_arn
        )
    return response["AuthorizedUrl"]

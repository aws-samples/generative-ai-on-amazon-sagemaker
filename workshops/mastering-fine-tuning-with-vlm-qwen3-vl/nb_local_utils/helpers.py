import boto3
import mlflow
from IPython.display import display, Markdown, HTML
from mlflow import MlflowClient


def pretty_print_html(text):
    # Replace newline characters with <br> tags
    html_text = text.replace('\n', '<br>')
    # Apply HTML formatting
    html_formatted = f'<pre style="font-family: monospace; background-color: #f8f8f8; padding: 10px; border-radius: 5px; border: 1px solid #0077b6;">{html_text}</pre>'
    # Display the formatted HTML
    return HTML(html_formatted)


def get_mlflow_server_arn():
    list_servers = boto3.client(
        "sagemaker"
    ).list_mlflow_tracking_servers()['TrackingServerSummaries']

    if len(list_servers) > 0:
        mlflow_arn = list_servers[0]['TrackingServerArn']
    else:
        mlflow_arn = None
    
    pretty_print_html(f"MLflow Server ARN: {mlflow_arn}")    
    return mlflow_arn


def get_tracking_server_uri(sess, tracking_server_arn):
    """get tracking server URI"""
    tracking_server_url = sess.sagemaker_client.create_presigned_mlflow_tracking_server_url(
        TrackingServerName=tracking_server_arn.split('/')[-1]
    )
    return tracking_server_url['AuthorizedUrl']

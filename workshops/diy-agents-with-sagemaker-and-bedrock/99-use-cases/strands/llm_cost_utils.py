import boto3
from pprint import pprint
import pandas as pd

# Helper function definition
from retrying import retry
import boto3
import json
from datetime import datetime

import time
from botocore.exceptions import ClientError
from strands import tool


@tool
def get_bedrock_token_count(model_id: str, start_time: datetime, end_time: datetime, granularity_seconds_period: int = 5, region="us-west-2") -> str:
    """
    Calculate token counts for Bedrock model usage
    
    Args:
        model_id: The Bedrock model ID
        start_time: Start time for querying CloudWatch metrics
        end_time: End time for querying CloudWatch metrics
        granularity_seconds_period: Period in seconds for CloudWatch metrics
        
    Returns:
        str: JSON containing token counts
    """

    cloudwatch = boto3.client('cloudwatch')

    namespace = 'AWS/Bedrock'
    metrics = ['InputTokenCount', 'OutputTokenCount', 'Invocations']

    results = {}

    for metric in metrics:
        response = cloudwatch.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric,
            Dimensions=[
                {
                    'Name': 'ModelId',
                    'Value': f"arn:aws:bedrock:{region}::foundation-model/{model_id}" if "titan" in model_id else model_id
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=granularity_seconds_period,
            Statistics=['Sum']
        )
        
        # Sum up all datapoints
        total_tokens = sum(dp['Sum'] for dp in response['Datapoints'])
        results[metric] = int(total_tokens)
    
    json_token = {
        'model_id': model_id,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration in minutes': (end_time - start_time).total_seconds() / 60,
        'input_tokens': results.get('InputTokenCount', 0),
        'output_tokens': results.get('OutputTokenCount', 0),
        'invocation_count': results.get('Invocations', 0)
    }

    return json.dumps(json_token)


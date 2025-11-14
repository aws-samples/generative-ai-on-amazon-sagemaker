import boto3, json

def cross_account_session_and_sagemaker_runtime(cross_account_role_arn: str):
    """
        Function to access cross-account boto3 session and sagemaker runtime
    """
    sts_client = boto3.client('sts')
    assumed_role = sts_client.assume_role(
    	RoleArn=cross_account_role_arn,
    	RoleSessionName='CrossAccountInference'
    )
    
    sagemaker_runtime = boto3.client(
        'sagemaker-runtime',
        aws_access_key_id=assumed_role['Credentials']['AccessKeyId'],
        aws_secret_access_key=assumed_role['Credentials']['SecretAccessKey'],
        aws_session_token=assumed_role['Credentials']['SessionToken']
    )
    
    boto_session = boto3.Session(
        aws_access_key_id=assumed_role['Credentials']['AccessKeyId'],
        aws_secret_access_key=assumed_role['Credentials']['SecretAccessKey'],
        aws_session_token=assumed_role['Credentials']['SessionToken']
    )

    return boto_session, sagemaker_runtime
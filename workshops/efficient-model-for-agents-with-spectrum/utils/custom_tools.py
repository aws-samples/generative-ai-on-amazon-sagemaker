import boto3
from strands import tool
from time import sleep

athena_client = boto3.client("athena")
glue_client = boto3.client("glue")
account_id = boto3.client("sts").get_caller_identity()["Account"]
region_name = boto3.session.Session().region_name

@tool(description="Tool to list the available databases in the data catalog using Glue")
def list_databases():
    """
        Tool to list the available databases in the data catalog using Glue
    """
    return glue_client.get_databases()

@tool(description="Tool to get the tables in a database using Glue")
def get_tables(database_name: str) -> list:
    """
        Tool to get the tables in a database using Glue
        Args:
            database_name (str): the name of the database to get the tables from
    """
    items = glue_client.get_tables(DatabaseName=database_name)['TableList']
    result = []
    for item in items:
        result.append({
            "name": item["Name"],
            "database": item["DatabaseName"],
            "columns": item["StorageDescriptor"]["Columns"]
        })
    return result

@tool(description="Get 5 samples from the table")
def get_samples(table_name: str, database_name: str) -> list:
    """
        Get 5 samples from the table
        Args:
            table_name (str): the name of the table to get the samples from
            database_name (str): the name of the database to get the tables from
    """
    query = f"SELECT * FROM {database_name}.{table_name} LIMIT 5"
    query_execution_id = athena_client.start_query_execution(
        QueryString=query,
        ResultConfiguration={
            "OutputLocation": f"s3://sagemaker-{region_name}-{account_id}/athena-query-outputs/"
        }
    )["QueryExecutionId"]
    sleep(1)
    response = athena_client.get_query_results(QueryExecutionId=query_execution_id)
    columns = [col['VarCharValue'] for col in response['ResultSet']['Rows'][0]['Data']]
    data_rows = []
    for row in response['ResultSet']['Rows'][1:]:
        values = [col.get('VarCharValue', None) for col in row['Data']]
        record = dict(zip(columns, values))
        data_rows.append(record)
    return data_rows

@tool(description="Tool to execute a SQL query using Amazon Athena and TrinoSQL")
def execute_query(query: str) -> str:
    """
        Tool to execute a SQL query using Amazon Athena and TrinoSQL
        Args:
            query (str): the SQL query to execute using Amazon Athena
        Returns:
            query_execution_id (str): the ID of the query executed by Amazon Athena
    """
    return athena_client.start_query_execution(
        QueryString=query,
        ResultConfiguration={
            "OutputLocation": f"s3://sagemaker-{region_name}-{account_id}/athena-query-outputs/"
        }
    )["QueryExecutionId"]

@tool(description="Tool to get the result of a query executed by Amazon Athena")
def get_query_results(query_execution_id: str):
    """
        Tool to get the result of a query executed by Amazon Athena
        Args:
            query_execution_id (str): the ID of the query executed by Amazon Athena
    """
    # Get response from Athena
    response = athena_client.get_query_results(QueryExecutionId=query_execution_id)
    # Extract column names
    columns = [col['VarCharValue'] for col in response['ResultSet']['Rows'][0]['Data']]
    
    # Extract data rows
    data_rows = []
    for row in response['ResultSet']['Rows'][1:]:
        values = [col.get('VarCharValue', None) for col in row['Data']]
        record = dict(zip(columns, values))
        data_rows.append(record)
    return data_rows
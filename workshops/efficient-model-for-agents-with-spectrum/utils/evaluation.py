import io
import boto3
import json
import time
import pandas as pd
from IPython.display import display


def get_mlflow_tracking_server_arn() -> str:
    """
    Retrieves the ARN of the first MLflow Tracking Server in SageMaker.

    Returns:
        str: The ARN of the MLflow Tracking Server.
    """
    # Create a SageMaker client
    sagemaker_client = boto3.client('sagemaker')

    try:
        # List MLflow Tracking Servers
        response = sagemaker_client.list_mlflow_tracking_servers()

        mlflow_tracking_arn = response['TrackingServerSummaries'][0][
            'TrackingServerArn'
        ]
        return mlflow_tracking_arn

    except Exception as e:
        print(f"An error occurred: {e}")


def extract_sql(text):
    """Extract SQL from model response, removing thinking tags."""
    return text.split('</think>')[-1].strip()


def execute_athena_query(query, database, s3_output):
    """Execute Athena query and display results."""
    athena_client = boto3.client('athena')

    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': s3_output},
    )

    query_execution_id = response['QueryExecutionId']

    while True:
        response = athena_client.get_query_execution(
            QueryExecutionId=query_execution_id
        )
        state = response['QueryExecution']['Status']['State']

        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break

        time.sleep(1)

    if state == 'SUCCEEDED':
        results = []
        paginator = athena_client.get_paginator('get_query_results')
        for page in paginator.paginate(QueryExecutionId=query_execution_id):
            for row in page['ResultSet']['Rows']:
                results.append(
                    [field.get('VarCharValue', '') for field in row['Data']]
                )

        if results:
            df = pd.DataFrame(results[1:], columns=results[0])
            display(df)
        else:
            print("Successfully executed.")
    else:
        error_message = response['QueryExecution']['Status'].get(
            'StateChangeReason', 'No error message available'
        )
        raise Exception(f"Query failed: {error_message}")


def collect_athena_metrics(sql_query, db_name, s3_output, query_id):
    """Execute query and collect performance metrics."""
    start_time = time.time()

    try:
        athena_client = boto3.client('athena')
        response = athena_client.start_query_execution(
            QueryString=sql_query,
            QueryExecutionContext={'Database': db_name},
            ResultConfiguration={'OutputLocation': s3_output},
        )

        query_execution_id = response['QueryExecutionId']

        while True:
            response = athena_client.get_query_execution(
                QueryExecutionId=query_execution_id
            )
            state = response['QueryExecution']['Status']['State']
            if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
            time.sleep(1)

        if state == 'SUCCEEDED':
            results = []
            paginator = athena_client.get_paginator('get_query_results')
            for page in paginator.paginate(QueryExecutionId=query_execution_id):
                for row in page['ResultSet']['Rows']:
                    results.append(
                        [field.get('VarCharValue', '') for field in row['Data']]
                    )

            if results:
                df = pd.DataFrame(results[1:], columns=results[0])
                result_data = df.to_dict('records')
            else:
                result_data = "Successfully executed"
        else:
            raise Exception(
                f"Query failed: {response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')}"
            )

        end_time = time.time()

        return {
            'query_id': query_id,
            'latency_ms': round((end_time - start_time) * 1000, 2),
            'status': 'SUCCESS',
            'result': result_data,
            'error': None,
        }
    except Exception as e:
        end_time = time.time()
        return {
            'query_id': query_id,
            'latency_ms': round((end_time - start_time) * 1000, 2),
            'status': 'ERROR',
            'result': None,
            'error': str(e),
        }


class LineIterator:
    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0
    def __iter__(self):
        return self
    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                print("Unknown event type:" + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])
            
            
def generate_sql(question, endpoint_name, smr_client):
    """Generate SQL using SageMaker predictor."""
    system = """You are an expert SQL developer. Given the provided database schema and the following user question, generate a syntactically correct SQL query. 
    Only reply with the SQL query, nothing else. Do NOT use the backticks to identify the code, just reply with the pure SQL query."""

    query = f"""-- Schema --
    CREATE TABLE orders (row_id BIGINT, order_id STRING, order_date TIMESTAMP, ship_date TIMESTAMP, ship_mode STRING, customer_id STRING, customer_name STRING, segment STRING, city STRING, state STRING, country STRING, postal_code FLOAT, market STRING, region STRING, product_id STRING, category STRING, sub-category STRING, product_name STRING, sales FLOAT, quantity BIGINT, discount FLOAT, profit FLOAT, shipping_cost FLOAT, order_priority STRING); 
    CREATE EXTERNAL TABLE returns (returned STRING, order_id STRING, market STRING)
    -- Query --
    {question}
    -- SQL --"""

    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        'max_tokens': 256,
        'temperature': 0.0,
        'stream': True,
    }

    resp = smr_client.invoke_endpoint_with_response_stream(
        EndpointName=endpoint_name,
        Body=json.dumps(payload),
        ContentType="application/json",
    )

    event_stream = resp["Body"]
    start_json = b"{"
    full_response = ""
    # start_time = time.time()
    token_count = 0
    for line in LineIterator(event_stream):
        if line != b"" and start_json in line:
            data = json.loads(line[line.find(start_json):].decode("utf-8"))
            token_text = data['choices'][0]['delta'].get('content', '')
            full_response += token_text
            token_count += 1
            # # For timing inference and clearing cell in jupyter
            # Calculate tokens per second
            # elapsed_time = time.time() - start_time
            # tps = token_count / elapsed_time if elapsed_time > 0 else 0
            # Clear the output and reprint everything
            # clear_output(wait=True) # requires 'from IPython.display import clear_output'
    return full_response


def analyze_qwen_results(results_file, eval_data_file=None):
    """Analyze model performance and generate summary statistics."""
    with open(results_file, 'r') as f:
        results = json.load(f)

    if eval_data_file is None:
        df = pd.DataFrame(results)
        summary = df.groupby('status').size()
        print("=== SUMMARY BY STATUS ===")
        print(summary)
        return df, summary

    with open(eval_data_file, 'r') as f:
        eval_data = json.load(f)

    results_dict = {r['query_id']: r for r in results}

    combined = []
    for item in eval_data:
        query_id = item['id']
        result = results_dict.get(query_id, {})

        combined.append(
            {
                'level': item['level'],
                'query_id': query_id,
                'status': result.get('status', 'MISSING'),
                'latency_ms': result.get('latency_ms', 0),
                'error_message': result.get('error', ''),
                'question': item['question'],
            }
        )

    df = pd.DataFrame(combined)

    summary = (
        df.groupby('level')
        .agg(
            {
                'status': lambda x: (x == 'SUCCESS').sum(),
                'query_id': 'count',
                'latency_ms': 'mean',
            }
        )
        .round(2)
    )

    summary['success_rate'] = (
        summary['status'] / summary['query_id'] * 100
    ).round(1)
    summary.columns = [
        'successful_queries',
        'total_queries',
        'avg_latency_ms',
        'success_rate_%',
    ]

    print("=== SUMMARY BY DIFFICULTY LEVEL ===")
    print(summary)

    # unnest metrics in order to log to mlflow

    nested_metrics = summary.to_dict(orient='dict')
    success_queries = nested_metrics['successful_queries']
    total_queries = nested_metrics['total_queries']
    success_rate = nested_metrics['success_rate_%']
    metrics = {}
    metrics['success_rate_advanced'] = success_rate.pop('Advanced')
    metrics['success_rate_intermediate'] = success_rate.pop('Intermediate')
    metrics['success_rate_simple'] = success_rate.pop('Simple')
    metrics['total_queries_advanced'] = total_queries.pop('Advanced')
    metrics['total_queries_intermediate'] = total_queries.pop('Intermediate')
    metrics['total_queries_simple'] = total_queries.pop('Simple')
    metrics['successful_queries_advanced'] = success_queries.pop('Advanced')
    metrics['successful_queries_intermediate'] = success_queries.pop(
        'Intermediate'
    )
    metrics['successful_queries_simple'] = success_queries.pop('Simple')

    return df, summary, metrics

import json
from mcp.server.fastmcp import Context, FastMCP
import boto3

REGION = boto3.session.Session().region_name

# Create a named server
COLLECTION_NAME = "agent-ws-collection"

# Specify dependencies for deployment and development
mcp = FastMCP("OpenSearch DSL Query App", dependencies=["pandas", "numpy"])


##############################################################
# Helper functions for OSS
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3
import os
import requests
import json
from retry import retry


def get_opensearch_collection_endpoint(collection_name, region=REGION):
    """
    Get the OpenSearch Serverless collection endpoint from a collection name
    
    Args:
        collection_name (str): The name of the OpenSearch Serverless collection
        region (str, optional): AWS region. If None, uses the default region.
        
    Returns:
        dict: Dictionary containing collection endpoints and ID
    """
    # Initialize the OpenSearch Serverless client
    aoss = boto3.client('opensearchserverless', region_name=region)
    service = 'aoss'
    session = boto3.Session(aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"],
                        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
                        aws_session_token = os.environ["AWS_SESSION_TOKEN"],
                        region_name=region)
    credentials =session.get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service)
    try:
        # Use batch_get_collection to get collection details by name
        response = aoss.batch_get_collection(names=[collection_name])
        
        # Check if collection was found
        if not response['collectionDetails']:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        # Extract collection details
        collection = response['collectionDetails'][0]
        
        # Return the endpoints and ID
        return {
            'collection_id': collection['id'],
            'collection_endpoint': collection['collectionEndpoint'],
            'dashboard_endpoint': collection['dashboardEndpoint'],
            'collection_arn': collection['arn']
        }
    
    except Exception as e:
        print(f"Error getting collection endpoint: {str(e)}")
        raise


def query_opensearch_with_dsl(collection_endpoint, dsl_json, index_name = 'guardduty-index', region=REGION):
    """
    Query an OpenSearch index using DSL with the OpenSearch Python client
    
    Args:
        collection_endpoint (str): The OpenSearch collection endpoint (without https://)
        index_name (str): Name of the index to query
        dsl_json (dict): The OpenSearch DSL query
        region (str, optional): AWS region. If None, uses the default region.
        
    Returns:
        dict: Query results
    """
    # Get AWS credentials
    session = boto3.Session(aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"],
                            aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
                            aws_session_token = os.environ["AWS_SESSION_TOKEN"],
                            region_name=region)
    credentials = session.get_credentials()
    
    # Create the auth for OpenSearch
    auth = AWSV4SignerAuth(credentials, session.region_name, 'aoss')  # Use 'aoss' for OpenSearch Serverless
    
    # Create the OpenSearch client
    client = OpenSearch(
        hosts=[{'host': collection_endpoint, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20
    )
    
    try:
        # Execute the search query
        response = client.search(
            body=dsl_json,
            index=index_name
        )
        
        # Print summary of results
        hits = response['hits']['hits']
        total = response['hits']['total']['value'] if isinstance(response['hits']['total'], dict) else response['hits']['total']
        
        print(f"Found {total} documents in {index_name}")
        print(f"Showing top {len(hits)} results:")
    
        return hits
    except Exception as e:
        print(f"Error querying OpenSearch: {str(e)}")
        raise



##################################
# TOOLS
@mcp.tool()
def query_dsl(dsl_json: dict, region=REGION):
    """Query input DSL to OpenSearch Collection.
    Args:
        dsl_json (dict): A DSL query
        region (str): us-west-2 is the default value
    """
    collection_endpoint = get_opensearch_collection_endpoint(COLLECTION_NAME)["collection_endpoint"].split("https://")[1]
    return query_opensearch_with_dsl(collection_endpoint, dsl_json, region=region)


@mcp.tool()
def get_index_schema(index_name: str) -> dict:
    """Return JSON schema of an index in the OpenSearch Collection """
    with open(index_name+"-schema.json", "r") as f:
        schema = json.load(f)
    return schema


@mcp.tool()
def add_two_numbers(a: int, b: int) -> str:
    """Add two numbers"""
    return f"{a} + {b} = {a+b} : This is to show your MCP tool has been invoked successfully."


if __name__ == "__main__":
    mcp.run()

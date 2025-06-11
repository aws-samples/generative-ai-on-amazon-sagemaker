import boto3

REGION = boto3.session.Session().region_name

###########################
# GuardDuty
###########################
def create_sample_findings(detector_id = None, finding_types=None):
    try:
        # Create GuardDuty client
        guardduty_client = boto3.client('guardduty', region_name=REGION)
        
        # Prepare the parameters
        params = {
            'DetectorId': get_detector_id() if detector_id is None else detector_id
        }
        
        # If specific finding types are provided, add them to the request
        if finding_types:
            params['FindingTypes'] = finding_types
            
        # Generate sample findings
        response = guardduty_client.create_sample_findings(**params)
        print("Successfully generated sample findings")
        
    except guardduty_client.exceptions.BadRequestException as e:
        print(f"Bad request error: {str(e)}")
    except guardduty_client.exceptions.InternalServerErrorException as e:
        print(f"Internal server error: {str(e)}")
    except Exception as e:
        print(f"Error generating sample findings: {str(e)}")

def get_guardduty_findings(detector_id=None, region=REGION):
    detector_id = get_detector_id() if detector_id is None else detector_id
    guardduty_client = boto3.client('guardduty', region_name=region)
    # Initialize variables for pagination
    all_findings = []
    next_token = None
# Paginate through all findings
    while True:
        # Prepare parameters for the API call
        params = {
            'DetectorId': detector_id,
            'MaxResults': 50,  # Maximum allowed by the API
            'SortCriteria': {
                'AttributeName': 'createdAt',
                'OrderBy': 'DESC'  # Most recent findings first
            }
        }

        # Add the next token if we have one
        if next_token:
            params['NextToken'] = next_token

        # Make the API call
        response = guardduty_client.list_findings(**params)

        # Get the finding IDs from this page
        finding_ids = response['FindingIds']

        # If we have findings, get their details
        if finding_ids:
            # Get detailed information about these findings
            findings_response = guardduty_client.get_findings(
                DetectorId=detector_id,
                FindingIds=finding_ids
            )

            # Add the findings to our collection
            all_findings.extend(findings_response['Findings'])

            print(f"Retrieved {len(finding_ids)} findings. Total so far: {len(all_findings)}")
        # Check if there are more findings to retrieve
        if len(response['NextToken']):
            next_token = response['NextToken']
        else:
            # No more findings, exit the loop
            break

    print(f"Retrieved a total of {len(all_findings)} findings")
    return all_findings

def get_detector_id():
    guardduty_client = boto3.client('guardduty', region_name=REGION)
    detectors = guardduty_client.list_detectors()
    if detectors['DetectorIds']:
        return detectors['DetectorIds'][0]
    return None


###########################
# OpenSearch Serverless
###########################
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3
import requests
import json
from retry import retry


def create_oss_index(collection_name, index_name, region=REGION):
    service = 'aoss'
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service)

    host = get_opensearch_collection_endpoint(collection_name)["collection_endpoint"].split("https://")[1]

    # create an opensearch client and use the request-signer
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
    )

    # create an index
    print('Creating index')
    try:
        create_response = client.indices.create(
            index_name
        )
    except Exception as e:
        if "already exists" in str(e):
            delete_oss_index(host, index_name)
            print('Recreating index')
            create_response = client.indices.create(
                index_name
            )
        else:
            raise e

    return create_response


def delete_oss_index(host, index_name, region=REGION):
    service = 'aoss'
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service)
    
    # create an opensearch client and use the request-signer
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
    )
    
    # create an index
    print('Deleting index')
    # delete the index
    delete_response = client.indices.delete(
        index_name
    )
    return delete_response


@retry(Exception, tries=5, delay=5)
def index_document_oss(idx, doc, collection_name, index_name, region=REGION):
    service = 'aoss'
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service)
    host = get_opensearch_collection_endpoint(collection_name)["collection_endpoint"].split("https://")[1]
    # create an opensearch client and use the request-signer
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
    )

    response = client.index(
        index=index_name,
        body=doc,
        id=str(idx)
    )
    return response


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
    credentials = boto3.Session().get_credentials()
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


def check_opensearch_index_exists(collection_endpoint, index_name = 'guardduty-index', region=REGION):
    """
    Check if an index exists in an OpenSearch Serverless collection
    
    Args:
        collection_endpoint (str): The OpenSearch collection endpoint (without https://)
        index_name (str): Name of the index to check
        region (str, optional): AWS region. If None, uses the default region.
        
    Returns:
        bool: True if the index exists, False otherwise
    """
    service = 'aoss'
    # Create AWS authentication for requests
    session = boto3.Session(region_name=region)
    credentials = session.get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service)
    
    # Prepare the URL to check if the index exists
    url = f'https://{collection_endpoint}/{index_name}'
    
    try:
        # Make a HEAD request to check if the index exists
        response = requests.head(url, auth=auth)
        
        # HTTP 200 means the index exists
        if response.status_code == 200:
            print(f"Index '{index_name}' exists in the collection.")
            return True
        # HTTP 404 means the index doesn't exist
        elif response.status_code == 404:
            print(f"Index '{index_name}' does not exist in the collection.")
            return False
        else:
            print(f"Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error checking if index exists: {str(e)}")
        return False



def query_opensearch_with_dsl(collection_name, index_name, query_dsl, region=REGION):
    """
    Query an OpenSearch index using DSL with the OpenSearch Python client
    
    Args:
        collection_endpoint (str): The OpenSearch collection endpoint (without https://)
        index_name (str): Name of the index to query
        query_dsl (dict): The OpenSearch DSL query
        region (str, optional): AWS region. If None, uses the default region.

    Returns:
        dict: Query results
    """
    # Get AWS credentials
    session = boto3.Session(region_name=region)
    credentials = session.get_credentials()
    
    # Create the auth for OpenSearch
    auth = AWSV4SignerAuth(credentials, session.region_name, 'aoss')  # Use 'aoss' for OpenSearch Serverless

    collection_endpoint = get_opensearch_collection_endpoint(collection_name)["collection_endpoint"].split("https://")[1]
    
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
            body=query_dsl,
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



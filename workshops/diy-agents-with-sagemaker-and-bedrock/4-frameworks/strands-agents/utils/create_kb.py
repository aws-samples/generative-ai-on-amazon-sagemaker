import boto3
import json
import time
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

def create_knowledge_base_with_document(pdf_path, kb_name="bedrock-kb", region="us-east-1"):
    """Create Bedrock Knowledge Base with document - clean and simple"""
    
    # Initialize clients
    bedrock_agent = boto3.client('bedrock-agent', region_name=region)
    s3 = boto3.client('s3', region_name=region)
    iam = boto3.client('iam', region_name=region)
    sts = boto3.client('sts', region_name=region)
    aoss = boto3.client('opensearchserverless', region_name=region)
    
    account_id = sts.get_caller_identity()['Account']
    credentials = boto3.Session().get_credentials()
    
    # Resource names
    bucket_name = f"{account_id}-{kb_name}-docs"
    collection_name = f"{kb_name}-collection"
    role_name = f"{kb_name}-role"
    index_name = "bedrock-knowledge-base-default-index"
    
    print(f"🚀 Creating Knowledge Base: {kb_name}")
    
    # 1. Cleanup existing KB
    try:
        kbs = bedrock_agent.list_knowledge_bases()
        for kb in kbs.get('knowledgeBaseSummaries', []):
            if kb['name'] == kb_name:
                print(f"🗑️  Deleting existing KB: {kb_name}")
                bedrock_agent.delete_knowledge_base(knowledgeBaseId=kb['knowledgeBaseId'])
                time.sleep(10)  # Wait for deletion
                break
    except:
        pass
    
    # 2. Create S3 bucket and upload
    print(f"📦 Creating S3 bucket: {bucket_name}")
    try:
        if region == 'us-east-1':
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        print(f"✅ Created S3 bucket: {bucket_name}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"✅ S3 bucket already exists: {bucket_name}")
    except Exception as e:
        print(f"❌ Error creating bucket: {e}")
        raise
    
    # Upload file to S3
    s3.upload_file(pdf_path, bucket_name, os.path.basename(pdf_path))
    print(f"✅ Uploaded to S3: {bucket_name}")
    
    # 3. Create IAM role
    role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["aoss:APIAccessAll"], "Resource": f"arn:aws:aoss:{region}:{account_id}:collection/*"},
            {"Effect": "Allow", "Action": ["s3:GetObject", "s3:ListBucket"], "Resource": [f"arn:aws:s3:::{bucket_name}", f"arn:aws:s3:::{bucket_name}/*"]},
            {"Effect": "Allow", "Action": ["bedrock:InvokeModel"], "Resource": f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0"}
        ]
    }
    
    try:
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "bedrock.amazonaws.com"}, "Action": "sts:AssumeRole"}]})
        )
        role_arn = role['Role']['Arn']
        print(f"✅ Created IAM role: {role_name}")
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = iam.get_role(RoleName=role_name)['Role']['Arn']
        print(f"✅ Using existing IAM role: {role_name}")
    
    iam.put_role_policy(RoleName=role_name, PolicyName=f"{role_name}-policy", PolicyDocument=json.dumps(role_policy))
    print(f"✅ IAM policy updated")
    
    # 4. Create OpenSearch policies
    for policy_type, policy_data in [
        ('encryption', {"Rules": [{"Resource": [f"collection/{collection_name}"], "ResourceType": "collection"}], "AWSOwnedKey": True}),
        ('network', [{"Rules": [{"Resource": [f"collection/{collection_name}"], "ResourceType": "collection"}], "AllowFromPublic": True}])
    ]:
        try:
            aoss.create_security_policy(name=f"{kb_name}-{policy_type}", type=policy_type, policy=json.dumps(policy_data))
        except:
            pass
    
    # 5. Create collection
    try:
        collection = aoss.create_collection(name=collection_name, type='VECTORSEARCH')
        collection_arn = collection['createCollectionDetail']['arn']
    except:
        collection_arn = aoss.batch_get_collection(names=[collection_name])['collectionDetails'][0]['arn']
    
    # Wait for collection to be active
    while aoss.batch_get_collection(names=[collection_name])['collectionDetails'][0]['status'] != 'ACTIVE':
        time.sleep(10)
    print(f"✅ Collection active: {collection_name}")
    
    # 6. Create data access policy
    current_user = sts.get_caller_identity()['Arn']
    if ':assumed-role/' in current_user:
        current_user = f"arn:aws:iam::{account_id}:root"
    
    data_policy = [{
        "Rules": [
            {"Resource": [f"collection/{collection_name}"], "Permission": ["aoss:CreateCollectionItems", "aoss:DeleteCollectionItems", "aoss:UpdateCollectionItems", "aoss:DescribeCollectionItems"], "ResourceType": "collection"},
            {"Resource": [f"index/{collection_name}/*"], "Permission": ["aoss:CreateIndex", "aoss:DeleteIndex", "aoss:UpdateIndex", "aoss:DescribeIndex", "aoss:ReadDocument", "aoss:WriteDocument"], "ResourceType": "index"}
        ],
        "Principal": [role_arn, current_user]
    }]
    
    try:
        aoss.delete_access_policy(name=f"{kb_name}-access", type='data')
        time.sleep(10)
    except:
        pass
    
    aoss.create_access_policy(name=f"{kb_name}-access", type='data', policy=json.dumps(data_policy))
    print(f"✅ Data access policy created")
    
    # 7. Create OpenSearch index
    collection_id = aoss.batch_get_collection(names=[collection_name])['collectionDetails'][0]['id']
    endpoint = f"{collection_id}.{region}.aoss.amazonaws.com"
    
    client = OpenSearch(
        hosts=[{'host': endpoint, 'port': 443}],
        http_auth=AWSV4SignerAuth(credentials, region, 'aoss'),
        use_ssl=True, verify_certs=True,
        connection_class=RequestsHttpConnection, timeout=300
    )
    
    # Wait for access and create index
    print("🔐 Waiting for OpenSearch access...")
    for i in range(30):  # 5 minutes
        try:
            if client.indices.exists(index=index_name):
                client.indices.delete(index=index_name)
                time.sleep(10)
            
            client.indices.create(index=index_name, body={
                "settings": {"index.knn": "true", "number_of_shards": 1, "knn.algo_param.ef_search": 512, "number_of_replicas": 0},
                "mappings": {"properties": {"vector": {"type": "knn_vector", "dimension": 1024, "method": {"name": "hnsw", "engine": "faiss", "space_type": "l2"}}, "text": {"type": "text"}, "text-metadata": {"type": "text"}}}
            })
            
            # Critical: Wait for Bedrock to see the index
            print("⏳ Waiting for index to be visible to Bedrock...")
            time.sleep(60)  # Give Bedrock time to see the index
            
            # Verify index still exists
            if client.indices.exists(index=index_name):
                print(f"✅ Index ready: {index_name}")
                break
            else:
                print("⚠️  Index disappeared, retrying...")
                
        except Exception as e:
            if i < 29:
                print(f"Retrying OpenSearch access... ({i+1}/30)")
                time.sleep(10)
            else:
                raise e
    
    # 8. Create Knowledge Base
    print("📝 Creating Knowledge Base...")
    kb_response = bedrock_agent.create_knowledge_base(
        name=kb_name,
        description=f"Knowledge base: {kb_name}",
        roleArn=role_arn,
        knowledgeBaseConfiguration={'type': 'VECTOR', 'vectorKnowledgeBaseConfiguration': {'embeddingModelArn': f'arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0'}},
        storageConfiguration={'type': 'OPENSEARCH_SERVERLESS', 'opensearchServerlessConfiguration': {'collectionArn': collection_arn, 'vectorIndexName': index_name, 'fieldMapping': {'vectorField': 'vector', 'textField': 'text', 'metadataField': 'text-metadata'}}}
    )
    
    kb_id = kb_response['knowledgeBase']['knowledgeBaseId']
    
    # Wait for KB to be active
    while bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)['knowledgeBase']['status'] != 'ACTIVE':
        time.sleep(10)
    print(f"✅ Knowledge Base active: {kb_id}")
    
    # 9. Create data source and ingest
    ds_response = bedrock_agent.create_data_source(
        knowledgeBaseId=kb_id, name=f"{kb_name}-datasource", description="S3 data source",
        dataSourceConfiguration={'type': 'S3', 's3Configuration': {'bucketArn': f'arn:aws:s3:::{bucket_name}'}}
    )
    
    ds_id = ds_response['dataSource']['dataSourceId']
    
    # Start ingestion
    job_response = bedrock_agent.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)
    job_id = job_response['ingestionJob']['ingestionJobId']
    
    # Wait for ingestion
    while True:
        job_status = bedrock_agent.get_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id)
        status = job_status['ingestionJob']['status']
        if status == 'COMPLETE':
            print("✅ Ingestion completed")
            break
        elif status == 'FAILED':
            raise Exception("Ingestion failed")
        time.sleep(10)
    
    print(f"\n🎉 Success! Knowledge Base ID: {kb_id}")
    return kb_id

def main():
    pdf_file = "utils/bedrock-agentcore-dg.pdf"
    if not os.path.exists(pdf_file):
        print(f"❌ Error: {pdf_file} not found")
        return

    # Get current AWS region
    session = boto3.Session()
    region = session.region_name
    
    if not region:
        print("❌ No AWS region configured. Please run: aws configure set region <your-region>")
        return
    
    print(f"🌍 Using region: {region}")
    kb_id = create_knowledge_base_with_document(pdf_file, region=region)
    print(f"✅ Knowledge Base ready: {kb_id}")
    
    # Write KB ID to file for notebook access
    with open('kb_id.txt', 'w') as f:
        f.write(kb_id)
    
    print(f"💾 KB ID saved to kb_id.txt")

if __name__ == "__main__":
    main()
import boto3
import json
import time
import os

def create_knowledge_base_with_s3_vectors(pdf_path, kb_name="bedrock-kb", region="us-east-1"):
    """Create Bedrock Knowledge Base with S3 Vectors - clean and simple"""
    
    # Initialize clients
    bedrock_agent = boto3.client('bedrock-agent', region_name=region)
    s3 = boto3.client('s3', region_name=region)
    s3vectors = boto3.client('s3vectors', region_name=region)
    iam = boto3.client('iam', region_name=region)
    sts = boto3.client('sts', region_name=region)
    
    account_id = sts.get_caller_identity()['Account']
    
    # Resource names
    bucket_name = f"{account_id}-{kb_name}-docs"
    vector_bucket_name = f"{account_id}-{kb_name}-vectors"
    role_name = f"{kb_name}-role"
    vector_index_name = "bedrock-knowledge-base-default-index"
    
    print(f"🚀 Creating Knowledge Base: {kb_name}")
    print(f"📊 Using S3 Vectors for vector storage")
    
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
    
    # 2. Create S3 bucket for documents and upload
    print(f"📦 Creating S3 bucket for documents: {bucket_name}")
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
    
    # 3. Create S3 Vector Bucket
    print(f"🎯 Creating S3 vector bucket: {vector_bucket_name}")
    try:
        # Delete existing vector bucket if it exists
        try:
            s3vectors.delete_vector_bucket(
                vectorBucketName=vector_bucket_name
            )
            print(f"🗑️  Deleted existing vector bucket")
            time.sleep(15)  # Wait for deletion to complete
        except:
            pass
        
        # Create new vector bucket
        vector_bucket_response = s3vectors.create_vector_bucket(
            vectorBucketName=vector_bucket_name,
            encryptionConfiguration={
                'sseType': 'AES256'
            }
        )
        # Construct the ARN  
        vector_bucket_arn = f"arn:aws:s3vectors:{region}:{account_id}:bucket/{vector_bucket_name}"
        print(f"✅ Created S3 vector bucket: {vector_bucket_name}")
        
        # Wait for vector bucket to be active
        print("⏳ Waiting for vector bucket to be active...")
        time.sleep(5)
        
    except Exception as e:
        if "already exists" in str(e):
            vector_bucket_arn = f"arn:aws:s3vectors:{region}:{account_id}:bucket/{vector_bucket_name}"
            print(f"✅ Using existing vector bucket: {vector_bucket_name}")
        else:
            print(f"❌ Error creating vector bucket: {e}")
            raise
    
    # 4. Create Vector Index
    print(f"📍 Creating vector index: {vector_index_name}")
    try:
        # Delete existing index if it exists
        try:
            s3vectors.delete_index(
                vectorBucketName=vector_bucket_name,
                indexName=vector_index_name
            )
            print(f"🗑️  Deleted existing vector index")
            time.sleep(10)  # Wait for deletion
        except:
            pass
        
        # Create vector index with proper configuration for Bedrock
        vector_index_response = s3vectors.create_index(
            vectorBucketName=vector_bucket_name,  # Use bucket name, not ARN
            indexName=vector_index_name,  # Correct parameter name
            dataType='float32',  # Required parameter
            dimension=1024,  # Amazon Titan Text Embeddings V2 dimensions (singular, not plural)
            distanceMetric='cosine',  # Lowercase, recommended for Titan embeddings
            metadataConfiguration={  # Correct structure
                'nonFilterableMetadataKeys': ['AMAZON_BEDROCK_TEXT']  # Required for large text chunks
            }
        )
        vector_index_arn = f"arn:aws:s3vectors:{region}:{account_id}:bucket/{vector_bucket_name}/index/{vector_index_name}"
        print(f"✅ Created vector index: {vector_index_name}")
        
        # Wait for index to be ready
        print("⏳ Waiting for vector index to be ready...")
        time.sleep(10)
        
    except Exception as e:
        if "already exists" in str(e):
            vector_index_arn = f"arn:aws:s3vectors:{region}:{account_id}:bucket/{vector_bucket_name}/index/{vector_index_name}"
            print(f"✅ Using existing vector index: {vector_index_name}")
        else:
            print(f"❌ Error creating vector index: {e}")
            raise
    
    # 5. Create IAM role for Bedrock Knowledge Base
    print(f"🔑 Creating IAM role for Bedrock: {role_name}")
    
    # Trust policy - allows bedrock.amazonaws.com to assume this role
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    # Permissions policy - what the role can do
    permissions_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3vectors:*"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:ListBucket"
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel"
                ],
                "Resource": f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0"
            }
        ]
    }
    
    try:
        # Try to create the role
        role_response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description=f"Role for Bedrock Knowledge Base {kb_name}"
        )
        role_arn = role_response['Role']['Arn']
        print(f"✅ Created IAM role: {role_name}")
        
        # Attach the permissions policy
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName=f"{role_name}-permissions",
            PolicyDocument=json.dumps(permissions_policy)
        )
        print(f"✅ Attached permissions policy")
        
        # Wait for the role to propagate
        print("⏳ Waiting for IAM role to propagate...")
        time.sleep(15)  # Increased wait time
        
    except iam.exceptions.EntityAlreadyExistsException:
        # Role already exists, get its ARN and update the policy
        role_arn = iam.get_role(RoleName=role_name)['Role']['Arn']
        print(f"✅ Using existing IAM role: {role_name}")
        
        # Update the permissions policy in case resources changed
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName=f"{role_name}-permissions", 
            PolicyDocument=json.dumps(permissions_policy)
        )
        print(f"✅ Updated permissions policy")
    
    except Exception as e:
        print(f"❌ Error creating/updating IAM role: {e}")
        raise
    
    # 6. Create Knowledge Base with S3 Vectors
    print("📝 Creating Knowledge Base with S3 Vectors...")
    try:
        kb_response = bedrock_agent.create_knowledge_base(
            name=kb_name,
            description=f"Knowledge base: {kb_name} using S3 Vectors",
            roleArn=role_arn,
            knowledgeBaseConfiguration={
                'type': 'VECTOR',
                'vectorKnowledgeBaseConfiguration': {
                    'embeddingModelArn': f'arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0',
                    'embeddingModelConfiguration': {
                        'bedrockEmbeddingModelConfiguration': {
                            'dimensions': 1024
                        }
                    }
                }
            },
            storageConfiguration={
                'type': 'S3_VECTORS',
                's3VectorsConfiguration': {
                    'indexArn': vector_index_arn
                }
            }
        )
        
        kb_id = kb_response['knowledgeBase']['knowledgeBaseId']
        print(f"✅ Knowledge Base created: {kb_id}")
        
    except Exception as e:
        print(f"❌ Error creating Knowledge Base: {e}")
        raise
    
    # Wait for KB to be active
    print("⏳ Waiting for Knowledge Base to be active...")
    while True:
        kb_status = bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)
        status = kb_status['knowledgeBase']['status']
        if status == 'ACTIVE':
            print("✅ Knowledge Base is active")
            break
        elif status == 'FAILED':
            raise Exception(f"Knowledge Base creation failed: {kb_status}")
        time.sleep(10)
    
    # 7. Create data source and ingest
    print("📊 Creating data source...")
    ds_response = bedrock_agent.create_data_source(
        knowledgeBaseId=kb_id,
        name=f"{kb_name}-datasource",
        description="S3 data source",
        dataSourceConfiguration={
            'type': 'S3',
            's3Configuration': {
                'bucketArn': f'arn:aws:s3:::{bucket_name}',
                'inclusionPrefixes': [os.path.basename(pdf_path)]  # Only process our specific file
            }
        }
    )
    
    ds_id = ds_response['dataSource']['dataSourceId']
    print(f"✅ Data source created: {ds_id}")
    
    # 8. Start ingestion job
    print("🔄 Starting ingestion job...")
    job_response = bedrock_agent.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id
    )
    job_id = job_response['ingestionJob']['ingestionJobId']
    
    # Wait for ingestion to complete
    print("⏳ Waiting for ingestion to complete...")
    while True:
        job_status = bedrock_agent.get_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id,
            ingestionJobId=job_id
        )
        status = job_status['ingestionJob']['status']
        
        if status == 'COMPLETE':
            print("✅ Ingestion completed successfully")
            break
        elif status == 'FAILED':
            failure_reasons = job_status['ingestionJob'].get('failureReasons', ['Unknown error'])
            raise Exception(f"Ingestion failed: {failure_reasons}")
        elif status in ['IN_PROGRESS', 'STARTING']:
            print(f"⏳ Ingestion status: {status}")
            time.sleep(15)
        else:
            print(f"❓ Unexpected status: {status}")
            time.sleep(10)
    
    print(f"\n🎉 Success! Knowledge Base created with S3 Vectors")
    print(f"📋 Knowledge Base ID: {kb_id}")
    print(f"🎯 Vector Bucket: {vector_bucket_name}")
    print(f"📍 Vector Index: {vector_index_name}")
    print(f"🔑 IAM Role: {role_name}")
    print(f"💰 Cost savings: S3 Vectors provides up to 90% cost reduction vs traditional vector databases")
    
    return kb_id

def main():
    """Main function to create the knowledge base"""
    pdf_file = "../4-frameworks/strands-agents/utils/bedrock-agentcore-dg.pdf"
    if not os.path.exists(pdf_file):
        print(f"❌ Error: {pdf_file} not found")
        return

    # Get current AWS region
    session = boto3.Session()
    region = session.region_name
    
    if not region:
        print("❌ No AWS region configured. Please run: aws configure set region <your-region>")
        return
    
    print(f"🌍 Using AWS region: {region}")
    print(f"🎯 Using S3 Vectors for cost-effective vector storage")
    
    try:
        kb_id = create_knowledge_base_with_s3_vectors(pdf_file, region=region)
        print(f"\n✅ Knowledge Base ready for queries: {kb_id}")
        
        # Write KB ID to file for notebook access
        with open('../4-frameworks/strands-agents/kb_id.txt', 'w') as f:
            f.write(kb_id)
        
        print(f"💾 KB ID saved to kb_id.txt")
        
        # Test the knowledge base
        print(f"\n🧪 Testing Knowledge Base...")
        bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=region)
        
        test_query = "What is Amazon Bedrock AgentCore?"
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={'text': test_query},
            retrievalConfiguration={'vectorSearchConfiguration': {'numberOfResults': 3}}
        )
        
        print(f"✅ Test query successful! Retrieved {len(response['retrievalResults'])} results")
        
    except Exception as e:
        print(f"\n❌ Error creating Knowledge Base: {e}")
        raise

if __name__ == "__main__":
    main()
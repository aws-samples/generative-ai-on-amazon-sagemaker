#!/bin/bash

set -e

# Configuration
REPOSITORY_NAME="${1:-my-app}"
IMAGE_TAG="${2:-latest}"
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "=== ECR Push Script ==="
echo "Repository: $REPOSITORY_NAME"
echo "Tag: $IMAGE_TAG"
echo "Region: $AWS_REGION"
echo "Account: $AWS_ACCOUNT_ID"
echo ""

# Check if repository exists
echo "Checking if ECR repository exists..."
if aws ecr describe-repositories --repository-names "$REPOSITORY_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "✓ Repository '$REPOSITORY_NAME' exists"
else
    echo "✗ Repository '$REPOSITORY_NAME' does not exist"
    echo "Creating repository..."
    aws ecr create-repository \
        --repository-name "$REPOSITORY_NAME" \
        --region "$AWS_REGION" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    echo "✓ Repository created successfully"
fi

# Get ECR login token and authenticate Docker
echo ""
echo "Authenticating with ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
echo "✓ Authentication successful"

# Build the image (assumes Dockerfile in current directory)
echo ""
echo "Building Docker image..."
docker build -t "$REPOSITORY_NAME:$IMAGE_TAG" --network sagemaker ./training
echo "✓ Image built successfully"

# Tag the image for ECR
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG"
echo ""
echo "Tagging image for ECR..."
docker tag "$REPOSITORY_NAME:$IMAGE_TAG" "$ECR_URI"
echo "✓ Image tagged: $ECR_URI"

# Push to ECR
echo ""
echo "Pushing image to ECR..."
docker push "$ECR_URI"
echo "✓ Image pushed successfully"

echo ""
echo "=== Complete ==="
echo "Image URI: $ECR_URI"
#!/usr/bin/env bash
# Build the training image and push it to Amazon ECR. Mirrors the steps in the
# Isaac Lab on SageMaker post. Requires docker and the AWS CLI.
#
#   ./build_and_push.sh                # us-east-1, tag 0.6.1
#   AWS_REGION=eu-west-1 ./build_and_push.sh
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
REPO="${REPO:-lerobot-vla}"
LEROBOT_VERSION="${LEROBOT_VERSION:-0.6.1}"
TAG="${TAG:-${LEROBOT_VERSION}}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${REPO}:${TAG}"

aws ecr describe-repositories --repository-names "$REPO" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$AWS_REGION" >/dev/null

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"
# The DLC base image lives in AWS's ECR registry; log in there too.
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com" 2>/dev/null || true

# SageMaker training instances are x86_64; force the platform when building on Apple Silicon.
docker build --platform linux/amd64 \
  --build-arg LEROBOT_VERSION="$LEROBOT_VERSION" --build-arg DLC_REGION="$AWS_REGION" \
  -t "${REPO}:${TAG}" -f "$(dirname "$0")/${DOCKERFILE}" "$(dirname "$0")"
docker tag "${REPO}:${TAG}" "$IMAGE_URI"
docker push "$IMAGE_URI"

echo
echo "Pushed ${IMAGE_URI}"
echo "export TRAINING_IMAGE_URI=${IMAGE_URI}"

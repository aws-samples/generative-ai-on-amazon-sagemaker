## Prerequisites

- Amazon SageMaker HyperPod cluster with EKS orchestration and Karpenter autoscaling
- FSx for Lustre filesystem mounted via PersistentVolumeClaim (`fsx-claim`)
- Karpenter NodePool configured for GPU instances (e.g., `gpu-efa-provisioner-medium` for `ml.g5.12xlarge`)

## Setup

### 1. Copy files to FSx

The training scripts, configuration, and data need to be available on the shared FSx for Lustre filesystem.

If you are using **SageMaker Studio integrated with HyperPod**, Studio and HyperPod share the same FSx filesystem. You can copy files directly from the Studio IDE to the shared filesystem without additional setup. For more details, see [Accelerate foundation model training and inference with Amazon SageMaker HyperPod and Amazon SageMaker Studio](https://aws.amazon.com/blogs/machine-learning/accelerate-foundation-model-training-and-inference-with-amazon-sagemaker-hyperpod-and-amazon-sagemaker-studio/).

Alternatively, you can upload files using `kubectl cp`:

```bash
kubectl run fsx-upload --image=ubuntu:latest --restart=Never \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "fsx-upload",
        "image": "ubuntu:latest",
        "command": ["/bin/bash", "-c", "sleep infinity"],
        "volumeMounts": [{
          "name": "fsx-volume",
          "mountPath": "/data"
        }]
      }],
      "volumes": [{
        "name": "fsx-volume",
        "persistentVolumeClaim": {
          "claimName": "fsx-claim"
        }
      }]
    }
  }'

kubectl wait --for=condition=ready pod/fsx-upload --timeout=60s

kubectl cp ./option-1-continued-pre-training fsx-upload:/data/shared
```

### 2. Update `args.yaml`

If needed, update the following paths to match your FSx directory structure:

```yaml
output_dir: "/data/shared/option-1-continued-pre-training/model/"
train_dataset_path: "/data/shared/option-1-continued-pre-training/data/train/"
```

### 3. Update `pod-finetuning.yaml`

If needed, update the paths in the container command to match your FSx directory structure:

```yaml
command:
  - /bin/bash
  - -c
  - |
    pip install --no-cache-dir -r /data/shared/option-1-continued-pre-training/requirements.txt && \
    torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${PET_RDZV_ENDPOINT} \
    --rdzv_id=qwen3-4b-instruct-cpt \
    --max_restarts=3 \
    /data/shared/option-1-continued-pre-training/scripts/train.py \
    --config /data/shared/option-1-continued-pre-training/args.yaml
```

## Run the training job

```bash
kubectl apply -f pod-finetuning.yaml
```

## Monitor the training job

```bash
# Watch pod status
kubectl get pods -w

# Check logs
kubectl logs -f qwen3-4b-instruct-cpt-worker-0

# Check PyTorchJob status
kubectl get pytorchjob qwen3-4b-instruct-cpt
```

## Deploy the model

After training completes, you can deploy the fine-tuned model as an inference endpoint using the HyperPod Inference Operator.

### 4. Update `deployment.yaml`

Replace `<your-fsx-filesystem-id>` with your FSx for Lustre filesystem ID:

```yaml
fsxStorage:
  fileSystemId: <your-fsx-filesystem-id>
```

To find your filesystem ID:

```bash
kubectl get pv -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.csi.volumeHandle}{"\n"}{end}'
```

### 5. Deploy the inference endpoint

```bash
kubectl apply -f deployment.yaml
```

### Monitor the deployment

```bash
# Check endpoint status
kubectl describe inferenceendpointconfig qwen3-4b-instruct-cpt-endpoint

# Check deployment and pods
kubectl get deployments,pods -n default

# Check operator logs for errors
kubectl logs -n hyperpod-inference-system \
  $(kubectl get pods -n hyperpod-inference-system -o jsonpath='{.items[0].metadata.name}') \
  --tail=30
```

## Cleanup

```bash
# Delete the inference endpoint
kubectl delete inferenceendpointconfig qwen3-4b-instruct-cpt-endpoint

# Delete the training job
kubectl delete pytorchjob qwen3-4b-instruct-cpt

# Delete the FSx upload pod (if created)
kubectl delete pod fsx-upload
```

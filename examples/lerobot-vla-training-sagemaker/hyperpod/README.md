# LeRobot VLA fine-tuning on SageMaker HyperPod (EKS)

Runs the same image and `train.sh` as `../training-jobs/` as a Kubeflow `PyTorchJob`: one pod per node, all
GPUs of the node per pod, `accelerate launch` inside the pod, EFA requested automatically for multi-node.

```
hyperpod/
├── launch.sh                  # render the manifest for (model, instance type, nodes) and kubectl apply
└── pytorchjob.yaml.template   # Master + Worker replicas, FSx PVC, /dev/shm, EFA
```

## Prerequisites on the cluster

- Kubeflow Training Operator (`pytorchjobs.kubeflow.org`), NVIDIA and EFA device plugins (HyperPod EKS default).
- A PVC for shared storage, default name `fsx-claim` (override with `--pvc`). Checkpoints go to
  `/fsx/lerobot-vla/checkpoints/<job>`, the Hugging Face cache to `/fsx/lerobot-vla/hf-cache`.
- Nodes labelled `node.kubernetes.io/instance-type=ml.<type>` (HyperPod default) in a scaled-up instance group.
- A Hugging Face token for GR00T and π₀.₅ (gated backbones), passed as `--hf-token` or `HF_TOKEN`; it is put in
  the pod spec as a plain environment variable. ACT needs none.
- The image built in `../training-jobs/` pushed to ECR, passed as `IMAGE`.

## Usage

```bash
export IMAGE=<acct>.dkr.ecr.<region>.amazonaws.com/lerobot-vla:0.6.1 HF_TOKEN=hf_xxx
./launch.sh --model groot --instance-type ml.g7e.2xlarge                       # 1 node, 1 GPU, fine-tune
./launch.sh --model act   --instance-type ml.g6.12xlarge --nodes 2             # 2 nodes x 4 GPUs, DDP over EFA
./launch.sh --model pi05  --instance-type ml.g6.12xlarge --peft --batch-size 4 # LoRA on 24 GB GPUs
./launch.sh --model groot --instance-type ml.g6.12xlarge --batch-size 4 \
            --extra-args "--policy.tune_diffusion_model=false"                 # GR00T on 24 GB (single GPU only)
./launch.sh ... --steps 300 --dry-run                                           # print the manifest
kubectl get pytorchjob; kubectl logs -f <job>-master-0
```

Other flags: `--steps`, `--save-freq`, `--batch-size`, `--dataset <hub repo id>`, `--dataset-root <fsx path>`, `--namespace`, `--pvc`,
`--job-name`, `--hf-token`, `--distributed fsdp` (opt-in, needs `FSDP_WRAP_CLS` in the preset).

## How the dataset gets into the pod

From the Hugging Face Hub, by `repo_id`: `--dataset <org>/<name>` (default `sreetz-nv/so101_teleop_vials_rack_left`)
becomes `DATASET_REPO_ID` in the pod and `train.sh` passes `--dataset.repo_id`. `HF_HOME` / `HF_LEROBOT_HOME`
point at the FSx PVC, so the download happens once per cluster and every later pod (and every worker of a
multi-node job) reads it from FSx. To use a dataset that is already on FSx (no Hub access at runtime), pass
`--dataset-root /fsx/path/to/dataset` (the directory containing `meta/info.json`); `train.sh` then adds
`--dataset.root` and LeRobot reads it in place.

## How topology works

The Kubeflow operator injects pod-level `MASTER_ADDR`, `MASTER_PORT`, `RANK` (pod index) and `WORLD_SIZE`
(pod count). `train.sh` reads them, counts GPUs with `nvidia-smi`, unsets the pod-level `RANK`/`WORLD_SIZE`, and
runs `accelerate launch --num_processes=<gpus x nodes> --num_machines=<nodes> --machine_rank=<pod index>
--main_process_ip=$MASTER_ADDR`. For `--nodes > 1` the launcher adds `vpc.amazonaws.com/efa: N` to the pod
resources (N from the instance type: 1 on g5/g6/g6e up to 12xlarge, 4 on g7e/g6e.48xlarge, 32 on p5).
Check `NCCL INFO Assigned NET plugin Libfabric` in the master log to confirm EFA is in use.

## Scaling instance groups

`aws sagemaker update-cluster --instance-groups file://groups.json` must be given **all** groups; set
`InstanceCount` on the one you want. Nodes take 10-15 minutes. If a size has no capacity the group cycles
create/replace failures; `aws sagemaker list-cluster-events` + `describe-cluster-event` show the EC2 message.
Scale back to 0 when done: HyperPod bills per running node.

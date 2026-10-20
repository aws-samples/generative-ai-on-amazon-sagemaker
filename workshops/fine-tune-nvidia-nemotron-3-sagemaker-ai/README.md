# Fine-Tuning NVIDIA Nemotron 3 Nano with Amazon SageMaker AI

Fine-tune a model for evidence-based NDA review using [ContractNLI](https://stanfordnlp.github.io/contract-nli/). For each of 17 hypotheses, the model returns JSON with a label (`Entailment`, `Contradiction` or `NotMentioned`) and supporting evidence span numbers.

**Start with the primary 4B SageMaker Training job track.** The 30B-A3B serverless track is an optional alternative, not the next lab.

## Primary: Nemotron 3 Nano 4B with SageMaker Training jobs

Open notebooks in `02-smtj-workshop/` and run them in this order:

| Step | Notebook                                                     | Purpose                                                                                                |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| 1    | [Prepare data](02-smtj-workshop/1-prepare-data.ipynb)        | Build chat prompt/completion records, measure token lengths and upload train/validation data to S3.    |
| 2    | [Fine-tune](02-smtj-workshop/2-fine-tune-llm.ipynb)          | Write the recipe, launch `ModelTrainer`, monitor the exact job and wait for `Completed`.               |
| 3    | [Deploy](02-smtj-workshop/3-deployment.ipynb)                | Deploy the merged `model.tar.gz` with vLLM and leave the endpoint running for evaluation.              |
| 4    | [Evaluate and clean up](02-smtj-workshop/4-evaluation.ipynb) | Score held-out contracts, compare baselines, optionally run the Bedrock judge and delete the endpoint. |

The model is `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`. Training runs on one single-GPU instance chosen from five instance preferences (`ml.g5.2xlarge`, `ml.g6.2xlarge`, `ml.g6e.2xlarge`, `ml.g5.xlarge`, `ml.g6.xlarge`), PyTorch 2.8.0, BF16 LoRA, one epoch, sequence length 8,192, LoRA rank 32 and learning rate `1e-4`. Batch size 1 with four accumulation steps gives an effective batch of 4. `KEEP_ALIVE_SECONDS = 0` disables the warm pool. Hosting uses one instance from four instance pools (`ml.g5.xlarge`, `ml.g6.xlarge`, `ml.g6.2xlarge`, `ml.g6e.xlarge`), vLLM 0.28.0 and no Inference Component.

Deployment precedes evaluation because this track evaluates a real-time endpoint, not a Model Package. Notebook 3 can use an explicit `TRAINING_JOB_NAME`; otherwise it discovers the newest completed matching job. Prefer the exact name printed by notebook 2 when sharing an account. Notebook 4 defaults `RUN_ENDPOINT_EVAL` and `RUN_JUDGE_EVAL` to `False` and reads reference results from `02-smtj-workshop/baselines/`. Reference results are not measurements of your new run.

## Optional: Nemotron 3 Nano 30B-A3B with serverless customization

Open notebooks in `01-serverless-workshop/` and run them in this order:

| Step | Notebook                                                         | Purpose                                                                                     |
| ---- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 1    | [Prepare data](01-serverless-workshop/1-prepare-data.ipynb)      | Upload string prompt/completion records and register AI Registry datasets.                  |
| 2    | [Fine-tune](01-serverless-workshop/2-fine-tune-llm.ipynb)        | Run serverless `SFTTrainer` and associate the result with a Model Package Group.            |
| 3    | [Evaluate](01-serverless-workshop/3-evaluation.ipynb)            | Compare models with custom scoring, a frontier baseline and LLM-as-a-Judge.                 |
| 4    | [Deploy and clean up](01-serverless-workshop/4-deployment.ipynb) | Deploy the model package with vLLM and an Inference Component, run a non-streaming smoke test and clean up. |

The JumpStart model ID is `huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16`. Training uses 10 epochs, global batch size 128, LoRA rank 32 and learning rate `1e-4`; SageMaker selects training compute. Hosting uses one `ml.g5.12xlarge` with four GPUs and vLLM 0.22.0. Evaluation comes before deployment in this track. Model support and regional capacity must be checked before launching.

## Setup and prerequisites

- Use a SageMaker Studio JupyterLab space on SageMaker Distribution 4.5.0 with Python 3.12, or an equivalent isolated Linux x86_64 notebook environment with credentials and an execution role.
- Check separate quotas for training and hosting. The primary track needs Training job quota for all five training instance preferences and endpoint quota for the four endpoint instance pools. The optional track needs serverless customization/evaluation capacity and `ml.g5.12xlarge` hosting capacity.
- Grant the execution role the needed SageMaker, S3, ECR, CloudWatch and `iam:PassRole` permissions. Serverless workflows also use AI Registry, Model Registry, Pipelines and Lambda scorer resources.
- Leave `MLFLOW_APP_ARN` empty to discover the tagged stack app automatically before reuse/create fallback. Set it only to override discovery with a specific app in the same Region or resolve ambiguity. An empty ARN does not disable tracking. No Studio domain or MLflow tracking server is required. See the MLflow setup behavior below.
- Live baseline/judge evaluation requires the Bedrock models and IAM permissions documented in the evaluation notebook. Bedrock evaluation roles also require the appropriate service trust. No Bedrock evaluation is needed to inspect the primary track's saved reference results.
- Allow downloads of ContractNLI, dependencies and Hugging Face tokenizers/weights. Review model terms, including the serverless trainer's `accept_eula=True`.

### Notebook dependencies

Run the chosen directory's `requirements.txt` install from notebook 1, then restart the kernel. Both requirements files load `../constraints-notebook.txt`, which pins the SageMaker SDK and its individual modules. The SDK metapackage alone permits newer module versions, so its version is not a sufficient API compatibility check. Training dependencies remain separate in `02-smtj-workshop/scripts/requirements.txt`; do not install those CUDA packages in JupyterLab.

The [Distribution 4.5.0 CPU package manifest](https://github.com/aws/sagemaker-distribution/blob/main/build_artifacts/v4/v4.5/v4.5.0/cpu.env.out) uses Linux x86_64 and Python 3.12.14, with mixed SDK versions (metapackage 3.12.0, train 1.14.0, core 2.21.0). It is not the pinned workshop SDK. The constraints align the AWS dependency family and the `datasets`/`fsspec`/`s3fs` family to avoid broad resolver backtracking. All existing serverless dependencies are retained.

Linux CPython 3.12 wheel-only pip dry runs were used to check the notebook requirements without installing packages or building source distributions. These are resolver checks, not proof of compatibility with every preinstalled Studio extension, and do not establish an install-time guarantee. The optional track requests NumPy 2 and datasets 5, while the image contains NumPy 1.26.4 and an older datasets version; use an isolated kernel if other projects depend on the image defaults. After installing, restart the kernel, run `%pip check`, and smoke-test notebook imports in the actual Studio image before a paid run. Cross-platform pip resolution does not fully emulate Linux environment markers, so a native Linux check remains required. The CUDA training dependency set requires a separate DLC smoke test.

## Monitor training and MLflow

Both training notebooks resolve a ready standalone MLflow App before launching training. An explicit `MLFLOW_APP_ARN` always wins: malformed, inaccessible, missing or failed configured apps stop setup, never silently create a replacement. Creating/updating apps are polled every 15 seconds for up to `MLFLOW_READY_TIMEOUT_SECONDS` (900 by default). Other states fail with the ARN and status; this API does not return a failure-reason field. Permission and service errors stop setup.

With no configured ARN, both tracks first paginate apps and check `ListTags` on `workshop-mlflow-*` candidates. A stack app must have the template's `WorkshopStackId` tag containing a CloudFormation stack ARN in the app's account/Region, with a UUID matching the app-name suffix. A supplied role must match `RoleArn`; when no role is supplied, a supplied artifact URI must match `ArtifactStoreUri` (ignoring trailing slashes). Role takes precedence because the notebook default bucket can differ from the stack bucket. Without either scope, only a unique tag/name match is accepted. Multiple matches stop setup and request an explicit ARN. The stack app wins over an existing `nemotron-contractnli-workshop` duplicate, including while creating/updating; failed or unsupported states stop setup rather than falling back. Discovery reads at most 100 pages per listing and fails on incomplete scans or permission errors, never treating them as absence.

If no stack app matches, both tracks reuse the exact `MLFLOW_APP_NAME` if present, including waiting for a pending app. Otherwise:

- **SMTJ:** paginate apps in the current account/Region, reuse the exact `MLFLOW_APP_NAME` (`nemotron-contractnli-workshop`), or create it. The name is scoped by the account/Region, not derived into an ARN. New apps use the notebook execution role and `s3://<notebook-bucket>/mlflow-artifacts`; `MLFLOW_ROLE_ARN` and `MLFLOW_ARTIFACT_STORE_URI` override those values. Existing app settings are not updated. Multiple same-name apps require an explicit ARN; failed same-name apps are not silently replaced.
- **Serverless:** preserve the pinned `sagemaker-train` native discovery/create helper. It checks domain/account defaults when available, then ready apps, and otherwise creates a timestamp-named `finetune-mlflow-*` app. A domain is not required. Creation uses `TrainDefaults.get_role(...)` and `s3://sagemaker-<region>-<account>/mlflow-artifacts`; the SDK may create the bucket/prefix. It waits up to 600 seconds on creation, then the workshop verifies readiness. The SDK logs exceptions and returns `None`; the workshop raises rather than starting untracked training. Set both creation overrides in `config.py` to use the deterministic named workshop path instead. The evaluation notebook's existing one-argument resolver call remains supported.

Both creation paths explicitly send `AccountDefaultStatus="DISABLED"`, assign no domain defaults and never call `UpdateMlflowApp`. The SDK's version-upgrade path, which can change shared defaults, is deliberately not requested. The existing template's tag/name combination is discoverable without a stack update or manual ARN configuration. An explicit ARN remains an optional override. Save the printed ARN in `config.py` to keep training and evaluation on the same app, especially in shared accounts. On timeout, reconnect to that ARN rather than repeatedly creating apps.

Existing Studio clones are not updated by edits on another machine or by the startup script. Transfer the revised shared helper and both track configs, or update to the published revision after preserving local work. Restart the kernel and rerun setup with an empty ARN; verify the printed ARN points to the stack app before training. No stack update or duplicate-app deletion is needed.

Grant the notebook caller `sagemaker:DescribeMlflowApp`, `sagemaker:CreatePresignedMlflowAppUrl` and, for fallback, `sagemaker:ListMlflowApps`, `sagemaker:ListTags`, `sagemaker:CreateMlflowApp` and `iam:PassRole` on the app role. The app role must trust SageMaker and access the artifact bucket/prefix, including KMS permissions if required. Grant the training/evaluation roles the app's MLflow tracking permissions. Native serverless creation also needs the S3 bucket/prefix permissions used by the SDK. Use an approved app role via the overrides if the training role is unsuitable. Creation can take several minutes; interrupting setup does not cancel provisioning. Review retained app/artifact costs separately during cleanup.

Notebook 2 captures the exact created Training job, prints SageMaker console and CloudWatch links, and polls status plus secondary status every 60 seconds. Failed or stopped jobs raise with the service failure reason. Interrupting polling does not stop the cloud job.

Open the configured app from Studio's **MLflow** page or the temporary app URL printed in notebook 2. Select `nemotron-3-nano-4b-contractnli-sft` for the primary track or `contractnli-serverless-sft` for serverless. Open signed links within five minutes; rerun the link cell if needed. Experiments and metrics may appear only after provisioning, dependency installation and model loading. Log and metric visibility is not instantaneous. The 4B recipe leaves `mlflow_log_model: false`, retaining metrics and parameters without uploading another merged checkpoint.

Each evaluation notebook imports its own utility as `E`: [4B endpoint evaluation utilities](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/evaluation_utils.py) or [serverless evaluation utilities](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/01-serverless-workshop/evaluation_utils.py). Their APIs differ. The notebooks retain core submissions, execution choices, and result-inspection calls; the primary track also retains streaming/cost controls backed by `inference_performance.py`. The standalone `contractnli_scorer.py` still owns per-contract parsing and metric equations. Copy the matching helper with the notebook when updating a checkout.

Serverless evaluation writes seven JSON snapshots under `./eval_result`; primary-track evaluation keeps fresh quality/judge/performance variables in memory and writes judge inputs/results to S3 only when live judging runs. Neither utility computes confidence intervals or validates complete production schemas. The primary track's cleanup remains independent of all evaluation switches, including in saved-reference mode.

Evaluation reports belong to each track's evaluation notebook. Endpoint wall-clock latency includes client, network and service overhead; do not label it model-only latency or compare it directly with streaming time-to-first-token. Saved baselines without timing fields must not be presented as newly measured latency.

## Data isolation and cleanup

Both tracks default to `DATASET_PREFIX = "contractnli-nda-review"`. They write incompatible data schemas to the same S3 prefix if the bucket/session prefix is shared. Choose one track, or set distinct dataset prefixes before preparing either dataset.

Training ends automatically, but real-time endpoints remain billable until deleted. A nonzero warm-pool duration would also retain billable training compute; the primary recipe currently sets it to zero. Run notebook 4 cleanup for your chosen track, including if you stop after deployment in the primary track. Verify deletion of endpoints, endpoint configurations, models and optional Inference Components. Review retained S3 artifacts, logs, Model Packages, datasets and MLflow resources separately. Closing a notebook does not clean them up. This workshop makes no cost, timing or quality guarantees.

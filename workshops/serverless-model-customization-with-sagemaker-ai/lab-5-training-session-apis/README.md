# Lab 5: Training session APIs

Three notebooks that write their own training loops on a SageMaker AI training session with `Qwen/Qwen3.8-27B`
(LoRA rank 32, `us-west-2`). Each opens a session, measures the base model on held-out data, trains, and measures
again.

| Notebook | Task | Training in the notebook | Held-out result | Notebook run time |
|---|---|---|---|---|
| `5a_gsm_hard_grpo.ipynb` | GSM-Hard math (non-negative whole-number answers), 512-token budget, low reasoning effort | 12 GRPO steps, 16 problems x 8 samples, lr 4e-5 | accuracy 62.5% -> 82.5%, answers cut at 512 tokens 34.5% -> 3.5% (100 problems x 2) | ~25 min |
| `5b_code_rl.ipynb` | Competitive programming (DeepCoder), code run in AgentCore Code Interpreter, medium reasoning | 2 GRPO steps, 8 problems x 8 attempts, then a pre-trained adapter | pass rate 12.0-12.5% -> 36.0-38.5% with the pre-trained adapter (100 problems x 2; the notebook uses 50) | ~45 min |
| `5c_scheduling_agent.ipynb` | Strands scheduling agent, 2 tools, per-person working hours, 9 time zones, calendars in local time or UTC; thinking off | 12 GRPO steps, 8 requests x 8 attempts | mean reward 0.699-0.719 -> 0.887-0.953 (128 requests, 2 runs) | ~55-90 min |

Pre-trained Code RL adapter (cookbook recipe, single turn, 64 problems x 8 attempts per step, 13-22 min per step).
Accuracy is the cookbook's validation on 100 test problems x 1 attempt; pass rate is `coderl/eval_adapter.py` with
the notebook's settings, 100 problems x 2 attempts:

| Step | 0 (base) | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| Accuracy | 22% | 21% | 27% | 42% | not run |
| Pass rate, medium reasoning | 12.0% | | 20.0% | 29.5% | 36.0% |
| Pass rate, low reasoning | 29.5% | | | | 36.5% |

At medium reasoning about two thirds of the base model's responses reach the 12,288-token limit before writing code.
At low reasoning, 40.5% (base) and 37.5% (adapter) reach it, and sampling takes about the same time (about 6.5
minutes for 200 responses), so low reasoning does not shorten the notebook.

## Setup

Run in SageMaker Studio JupyterLab or on a laptop. The first code cell of each notebook installs `requirements.txt`
into the current kernel, like the other labs of this workshop; restart the kernel after it. The versions in
`requirements.txt` match the other labs (`sagemaker` 3.27.0, `boto3` 1.43.117, `datasets` 5.1.0), so all labs
can share one kernel. `transformers` is pinned to 5.5.4, the version the training sessions cookbook requires.

Requirements in the account (`us-west-2`):

- IAM role `SageMakerTrainingSessionRole` that trusts `training-session.sagemaker.amazonaws.com` and can write to
  `sagemaker-<region>-<account>` buckets.
- Quota for 2 concurrent training sessions (5b uses two sessions one after the other).
- 5b: `bedrock-agentcore:CreateCodeInterpreter`, `ListCodeInterpreters`, `GetCodeInterpreter`,
  `StartCodeInterpreterSession`, `InvokeCodeInterpreter` and `StopCodeInterpreterSession` (the notebook creates a code
  interpreter named `code_rl_sandbox` with network mode `SANDBOX` if it does not exist), `sagemaker:CreateModelPackage`,
  and the pre-trained adapter in S3 (`PRETRAINED_ADAPTER_S3`, a prefix that contains
  `checkpoints/hf/adapter_config.json` and `checkpoints/hf/adapter_model.safetensors`, 0.9 GB).
- 5c: Amazon Bedrock access to `openai.gpt-oss-120b-1:0` for the demo request in section 2.

## Layout

| Path | Content |
|---|---|
| `lab/runtime.py` | `TrainingRun` (5c): training session, rollout gateway on 127.0.0.1, rollouts, GRPO step, sampler refresh, progress bars |
| `lab/agents.py` | Strands helpers: gateway or Amazon Bedrock model, transcripts |
| `usecases/scheduling_hard.py` | Scheduling environment: people, calendars, requests, correct answers, tools, reward |
| `coderl/` | Code Interpreter sandbox (`code_interpreter.py`), local `/run_code` endpoint (`sandbox_proxy.py`), progress helper, adapter evaluation (`eval_adapter.py`) |
| `tracking.py`, `workshop_mlflow.py` | MLflow tracking: finds the workshop's MLflow App (same helper as the other labs) and logs settings, step metrics and evaluation results; turns itself off if no app is available |
| `images/` | Figures used in the notebooks |
| `scripts/run.py`, `scripts/eval_package.py` | Train or evaluate the scheduling agent from the command line |
| `scripts/register_adapter.py` | Register adapter files from S3 as a model package |
| `scripts/build_*.py` | Generate the notebooks |

## Notes

- **Loading an adapter.** After `create_training_client_from_state(model_package_arn=...)`, `create_sampling_client()`
  samples the base weights. Call `save_weights_and_get_sampling_client()` once to sample the loaded adapter.
- **Adapters across accounts and Regions.** Copy the adapter prefix to a bucket in the target account and Region and
  call `CreateModelPackage` with `IsCheckpoint: false` and the `BaseModel` of the original package
  (`scripts/register_adapter.py`, section 5 of 5b).
- **Model packages.** Every `save_weights_and_get_sampling_client()` call registers a model package version, so the
  model package group holds one version per training step.
- **Sandbox.** 5b runs generated code in AgentCore Code Interpreter with network mode `SANDBOX` (no network) and
  no execution role (no AWS credentials). 16 sessions start in about 15 s; each program runs in its own directory with
  `timeout`. Grading agrees with a SandboxFusion container on 80 of 80 Qwen responses. The default quotas are 1000 concurrent
  sessions per account and 30 `InvokeCodeInterpreter` requests per second.
- **Service errors.** One of eleven notebook runs hit `OperationError: ... internal error ... try again with a new
  session` on `save_weights_and_get_sampling_client`. `TrainingRun.refresh` retries and keeps the previous sampler if
  all attempts fail; rollouts whose sampling calls failed are left out of training.

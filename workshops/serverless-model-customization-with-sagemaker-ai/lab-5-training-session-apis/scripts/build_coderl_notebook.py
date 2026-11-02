"""Build 5b_code_rl.ipynb (Qwen3.8-27B)."""
import textwrap
from pathlib import Path

import nbformat as nbf

V = dict(MODEL="Qwen/Qwen3.8-27B", MODEL_SHORT="Qwen3.8-27B", OUT="5b_code_rl.ipynb",
         REASONING="medium", EVAL_N="50", GRPO_STEPS="2",
         BASE_MODEL_SPEC='{"HubContentName": "huggingface-vlm-qwen3-8-27b", "HubContentVersion": "1.3.0"}',
         RESULTS=("At medium reasoning effort, many of the base model's responses reach the 12,288-token limit before they\n"
                  "contain code. In the reference runs (100 problems x 2 attempts), the base model passed 12.0-12.5% and the\n"
                  "pre-trained adapter 36.0-38.5%. This notebook uses 50 problems to save time, so expect some variation."))

cells = []


def fill(s):
    """Replace each {{KEY}} in s with its value from V, then dedent and strip blank edge lines."""
    for k, v in V.items():
        s = s.replace("{{" + k + "}}", v)
    return textwrap.dedent(s).strip("\n")


def md(s):
    """Append a markdown cell with the filled text s."""
    cells.append(nbf.v4.new_markdown_cell(fill(s)))


def code(s):
    """Append a code cell with the filled source s."""
    cells.append(nbf.v4.new_code_cell(fill(s)))


md("""
# Code RL with [SageMaker AI training sessions](https://docs.aws.amazon.com/sagemaker/latest/dg/training-sessions-getting-started.html)

This notebook trains a model with reinforcement learning (RL) on code. The model gets a competitive-programming
problem and writes a Python solution. The solution runs against the problem's test cases in a sandbox, and the
reward is 1.0 when all tests pass.

The same setup works for any output that you can run and check automatically: SQL queries checked against a
database, code checked by unit tests, configuration files checked by a validator, data transformations checked
against expected output. To use it for your own task, replace the problems and the `grade` function and keep the
sandbox and the loop.

### How the pieces fit together

![Code RL loop](images/code-rl-loop.svg)

- **[Training loop](https://aws.amazon.com/blogs/machine-learning/write-your-own-training-loop-without-managing-gpus-with-amazon-sagemaker-training-sessions/)** (your code, in this kernel): samples attempts from the training session, scores them and sends
  the update.
- **Training session** (SageMaker AI): holds [`{{MODEL_SHORT}}`](https://huggingface.co/{{MODEL}}) with a
  [LoRA (low-rank adaptation)](https://arxiv.org/abs/2106.09685) adapter, generates
  responses and runs `forward_backward` and `optim_step`.
- **Grading** ([training sessions cookbook](https://github.com/aws/sagemaker-training-sessions-cookbook)): extracts
  the code block from a response and runs the problem's tests with the [LiveCodeBench](https://livecodebench.github.io) harness.
- **Sandbox**: the model's code is untrusted, so it runs in
  [Amazon Bedrock AgentCore Code Interpreter](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html), without
  network access or AWS credentials. The cookbook's grading code expects a [SandboxFusion](https://github.com/bytedance/SandboxFusion)-style HTTP endpoint, so a
  small local endpoint on `127.0.0.1` in this kernel forwards each program to Code Interpreter.

### The helper files

The notebook cells show the prompt, the grading call and the training loop. The rest is in a few files and in the
cookbook package:

| File | What it does | Used in |
|---|---|---|
| `coderl/code_interpreter.py` | `CodeInterpreterSandbox` keeps 16 Code Interpreter sessions open. `run(request)` writes the program and its files to a new directory in one session, runs it with a time limit, and returns stdout, stderr and the exit code. | Section 1 and every grading call |
| `coderl/sandbox_proxy.py` | `LocalSandbox` serves `/run_code` on `127.0.0.1` and passes each request to `CodeInterpreterSandbox.run`, so the cookbook's grading code can use Code Interpreter. | Section 1 |
| `coderl/progress.py` | `gather` waits for many requests at once and shows a progress bar. | Sections 3 to 5 |
| Training sessions cookbook (installed package) | The problems (`load_deepcoder_tasks`), the prompt (`apply_hf`), code extraction, the LiveCodeBench test harness and the reward. | Sections 2 to 5 |

Section 3 has an optional cell that follows one response from text to reward. `coderl/eval_adapter.py` evaluates
an adapter from the command line; the notebook does not use it.

### What you do

Responses are up to 12,288 tokens, so a training step that moves the pass rate (64 problems x 8 attempts) takes 13
to 22 minutes. Training for hours does not fit in a workshop, so you:

1. measure the base model on held-out problems,
2. run {{GRPO_STEPS}} [GRPO (Group Relative Policy Optimization)](https://arxiv.org/abs/2402.03300) steps yourself to see the loop work,
3. load an adapter that was trained beforehand with the cookbook's Code RL recipe (20 steps, about 6 hours) and
   measure it on the same problems.

{{RESULTS}}
""")

md("## 1. Setup")
md("""
Install the requirements, then restart the kernel (**Kernel > Restart Kernel**) so that the new package versions
are loaded.
""")
code("""
%pip install -r requirements.txt
""")

md("""
The next cell connects to your AWS account:

- **Region and account** come from the default boto3 session, so the training session runs in the Region your
  notebook uses.
- **`ROLE_ARN`** is the IAM role `SageMakerTrainingSessionRole` (see the
  [prerequisites](https://docs.aws.amazon.com/sagemaker/latest/dg/training-sessions-prereqs.html)). The training session service assumes it to read the
  base model and write adapters to S3. The workshop stack creates it.
- **`S3_OUTPUT`** is the S3 prefix where the service writes the adapter files, in your
  `sagemaker-<region>-<account>` bucket.
- **`MPG_ARN`** is a [model package group](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-model-group.html),
  `code-rl`. Every adapter you save is registered there in
  [Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html) as a
  [model package version](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-version.html). The cell creates the group, or reuses it if it already exists.
- **`PRETRAINED_ADAPTER`** is the S3 prefix of the adapter trained beforehand, used in section 5. It comes from the
  environment variable `PRETRAINED_ADAPTER_S3`; the workshop environment sets it for you.
""")
code("""
import asyncio, json, os, random, time
import boto3
import numpy as np

session = boto3.Session()
REGION = session.region_name
ACCOUNT = session.client("sts").get_caller_identity()["Account"]
ROLE_ARN = session.client("iam").get_role(RoleName="SageMakerTrainingSessionRole")["Role"]["Arn"]
S3_OUTPUT = f"s3://sagemaker-{REGION}-{ACCOUNT}/training-sessions/code-rl/"
# The placeholder default is used only when the environment variable is not set; section 5 needs the real prefix.
PRETRAINED_ADAPTER = os.environ.get("PRETRAINED_ADAPTER_S3", "s3://<bucket>/code-rl/adapter/model/")

sm = session.client("sagemaker")
# Create the model package group. If it already exists, the call raises a ClientError and the existing group is used.
try:
    MPG_ARN = sm.create_model_package_group(ModelPackageGroupName="code-rl")["ModelPackageGroupArn"]
except sm.exceptions.ClientError:
    MPG_ARN = sm.describe_model_package_group(ModelPackageGroupName="code-rl")["ModelPackageGroupArn"]
print(REGION, ROLE_ARN, MPG_ARN, sep="\\n")
""")
md("""
### The sandbox

Generated code runs in Amazon Bedrock AgentCore Code Interpreter. The code interpreter
[created](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-create.html) below has network
mode `SANDBOX` (no network access) and no execution role, so the code it runs has no AWS credentials.
`CodeInterpreterSandbox` (in `coderl/code_interpreter.py`) keeps 16
[sessions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-resource-session-management.html) open and runs each program in its own
directory of one of them.

The cookbook's grading code sends programs to a SandboxFusion-style HTTP endpoint, so `LocalSandbox` serves
`/run_code` on `127.0.0.1` in this kernel and passes each request to `CodeInterpreterSandbox.run`.
""")
code("""
from coderl.code_interpreter import CodeInterpreterSandbox
from coderl.sandbox_proxy import LocalSandbox

# The control-plane client creates and describes code interpreters.
agentcore = session.client("bedrock-agentcore-control")
# Reuse the code interpreter named code_rl_sandbox if it exists, otherwise create it with network mode SANDBOX.
found = [c for c in agentcore.list_code_interpreters(maxResults=100)["codeInterpreterSummaries"]
         if c["name"] == "code_rl_sandbox"]
if found:
    CODE_INTERPRETER_ID = found[0]["codeInterpreterId"]
else:
    CODE_INTERPRETER_ID = agentcore.create_code_interpreter(
        name="code_rl_sandbox", networkConfiguration={"networkMode": "SANDBOX"})["codeInterpreterId"]
# Wait until the code interpreter can accept sessions.
while agentcore.get_code_interpreter(codeInterpreterId=CODE_INTERPRETER_ID)["status"] != "READY":
    time.sleep(5)

# Open 16 Code Interpreter sessions. sandbox.run(request) runs one program in one of them.
sandbox = CodeInterpreterSandbox(CODE_INTERPRETER_ID, REGION, sessions=16)
# Serve sandbox.run at http://127.0.0.1:<port>/run_code. The cookbook's grading code reads SANDBOX_URL and
# SANDBOX_MAX_CONCURRENCY to find the endpoint and to limit the number of requests in flight.
os.environ["SANDBOX_URL"] = LocalSandbox(sandbox.run, concurrency=16).start()
os.environ["SANDBOX_MAX_CONCURRENCY"] = "16"
# Test run: the request and response use the SandboxFusion format. This prints 45.
print(sandbox.run({"code": "print(sum(range(10)))", "run_timeout": 10})["run_result"]["stdout"])
""")

md("""
## 2. Problems and reward

`load_deepcoder_tasks` returns problems from the
[DeepCoder dataset](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset) with their test cases. The prompt asks for a solution in a fenced Python
block. `grade` extracts the last code block, runs the LiveCodeBench test harness in the sandbox, and computes the
cookbook's reward: 1.0 if all tests pass, minus 0.1 if the answer has no code block.
""")
code("""
from training_sessions_cookbook.rl.envs.code_rl import deepcoder_reward
from training_sessions_cookbook.rl.envs.deepcoder import load_deepcoder_tasks
from training_sessions_cookbook.rl.envs.lcb_utils import extract_code_from_model
from training_sessions_cookbook.rl.envs.sandbox import sandbox_check_correctness
from training_sessions_cookbook.rl.prompts import apply_hf

# 512 training problems, shuffled with a fixed seed, and the first {{EVAL_N}} problems of the test split.
train_tasks = load_deepcoder_tasks("train", seed=0, max_prompts=512)
test_tasks = load_deepcoder_tasks("test")[:{{EVAL_N}}]
print(len(train_tasks), "train |", len(test_tasks), "test")
print(test_tasks[0].problem[:800])


async def grade(task, response_text):
    \"\"\"Run the code in one response against the task's tests and return (reward, passed).

    Args:
      task: a problem from load_deepcoder_tasks. task.tests holds its test cases.
      response_text: the decoded model response.

    Returns a tuple (reward, passed). passed is True when all tests pass. reward is 1.0 when all tests pass and
    0.0 otherwise, minus 0.1 when the response has no fenced code block.

    Side effects: when the response has a code block, the code and tests are sent to SANDBOX_URL, which runs them
    in Code Interpreter. A response without a code block is not sent.
    \"\"\"
    code = extract_code_from_model(response_text)  # the last fenced code block, or None
    passed = False
    if code is not None:
        # Runs the LiveCodeBench test harness in the sandbox, with a limit of 6 seconds per test.
        passed, _ = await sandbox_check_correctness(task.tests, code, 6)
    # The cookbook's reward function. Only the first returned value, the reward, is used.
    reward, _ = deepcoder_reward(response_text, passed)
    return reward, passed
""")

md("""
## 3. Base model on held-out problems

The prompt goes through the model's [chat template](https://huggingface.co/docs/transformers/chat_templating) with {{REASONING}} reasoning effort. Responses have up to 12,288 tokens at
temperature 1.0. Each problem is sampled twice; the progress bars show sampling and then grading.

**Run time:** opening the session takes from under a minute to a few minutes. The evaluation of 100 responses
takes about 5 to 8 minutes; most of it is waiting for the longest responses.
""")
code("""
from sagemaker.train.training_session import AdamParams, SamplingParams, ServiceClient

from coderl.progress import gather

# svc is the entry point to the training session APIs.
svc = ServiceClient(region=REGION, role_arn=ROLE_ARN, model_package_group_arn=MPG_ARN, s3_output_path=S3_OUTPUT)
# Start a training session with a LoRA adapter (rank 32) on the base model. The call returns when the session is
# running; the session holds resources until training_client.stop().
training_client = svc.create_lora_training_client(
    "{{MODEL}}", training_session_name=f"code-rl-{int(time.time())}", rank=32, alpha=64)
# The model's tokenizer, loaded from the session. It turns text into token ids and back.
tokenizer = training_client.get_tokenizer()
# A sampler for the current weights. Before any update these are the base model's weights.
sampling_client = training_client.create_sampling_client()
# max_tokens caps the response length; temperature 1.0 keeps the model's probabilities unchanged.
SAMPLING = SamplingParams(max_tokens=12288, temperature=1.0)


def prompt_tokens(task):
    \"\"\"Return the prompt for one task as a list of token ids.

    Args:
      task: a problem from load_deepcoder_tasks. Only task.problem, the problem text, is used.

    Returns the list of token ids to pass to sample(). apply_hf (from the cookbook) wraps the problem text in a
    user message and calls tokenizer.apply_chat_template, which adds the special tokens the model was trained with
    and ends where the model starts its answer. The reasoning setting is passed to the chat template as the
    reasoning effort. The function has no side effects.
    \"\"\"
    return apply_hf(task.problem, tokenizer, reasoning="{{REASONING}}")


async def evaluate(client, tasks, samples_per_problem=2):
    \"\"\"Sample each task samples_per_problem times with a sampling client and grade the responses.

    Args:
      client: a sampling client, for example from create_sampling_client or save_weights_and_get_sampling_client.
      tasks: the problems to evaluate, from load_deepcoder_tasks.
      samples_per_problem: number of responses sampled for each problem.

    Returns a tuple (pass rate, pairs). The pass rate is the share of all responses that pass all tests. pairs is a
    list of (task, response text), samples_per_problem entries per task, in the order of tasks.

    Side effects: sends one sampling request per task to the training session and one grading request per response
    with a code block to the sandbox, and shows a progress bar for each phase. All sampling requests are submitted
    first; sample() returns a future right away. The weights are not changed.
    \"\"\"
    futures = [client.sample(prompt_tokens(t), num_samples=samples_per_problem, sampling_params=SAMPLING) for t in tasks]
    # f.result() blocks until the response arrives, so each wait runs in a thread. gather (from coderl/progress.py)
    # awaits them all with a progress bar and returns the results in order.
    results = await gather([asyncio.to_thread(f.result) for f in futures], "sampling")
    # Each result holds samples_per_problem sequences with the sampled token ids (tokens), the logprob of each token
    # (logprobs) and why sampling stopped (stop_reason). tokenizer.decode turns the token ids back into text.
    pairs = [(t, tokenizer.decode(s.tokens)) for t, r in zip(tasks, results) for s in r.sequences]
    graded = await gather([grade(t, x) for t, x in pairs], "grading")
    return float(np.mean([p for _, p in graded])), pairs


t0 = time.time()
# Jupyter runs top-level await in the notebook's event loop.
base_pass, base_texts = await evaluate(sampling_client, test_tasks)
print(f"base model: pass rate {base_pass:.1%} on {len(test_tasks)} held-out problems, 2 attempts each ({time.time() - t0:.0f}s)")
""")

md("""
### Optional: follow one response from text to reward

This cell takes one base-model response that contains a code block and grades it step by step:

1. `extract_code_from_model` takes the last fenced code block from the response.
2. `sandbox_check_correctness` sends one request to `SANDBOX_URL` with four files: the extracted code (`code.py`),
   the problem's test cases (`test_cases.txt`), the LiveCodeBench test runner (`testing_util.py`), and a short main
   program that runs every test and exits with 0 only if all of them pass.
3. `CodeInterpreterSandbox.run` writes the files to a new directory in one Code Interpreter session, runs the main
   program with a time limit, and returns its output and exit code.
4. `deepcoder_reward` turns the result into the reward.

**Run time:** under a minute.
""")
code("""
# Pick the first response with a code block (responses cut at the token limit often have none).
task, text = next(((t, x) for t, x in base_texts if extract_code_from_model(x)), base_texts[0])
print("PROBLEM (first 500 characters):\\n" + task.problem[:500])
print("\\nFIRST TEST CASE:", json.dumps(task.tests[0])[:300])

solution = extract_code_from_model(text)
print(f"\\nRESPONSE: {len(text)} characters. Extracted code block:")
print(solution[:1500] if solution else "(none)")

passed = False
if solution is not None:
    # The same call that grade() makes. details is the sandbox response in SandboxFusion format.
    passed, details = await sandbox_check_correctness(task.tests, solution, 6)
    run_result = details.get("run_result") or {}
    print("\\nSANDBOX: status", details.get("status"), "| exit code", run_result.get("return_code"),
          "| run time", run_result.get("execution_time"))
    # The test runner prints details only when a test fails.
    print("test runner output:", (run_result.get("stdout") or "")[:500] or "(none)")
print("\\nall tests passed:", passed, "| reward:", deepcoder_reward(text, passed)[0])
""")

md("""
## 4. A few GRPO steps

![One GRPO step](images/grpo.svg)

Each step samples 8 attempts for each of 8 training problems and grades them in the sandbox. Within each group of 8,
the advantage of an attempt is its reward minus the group mean, divided by the group standard deviation. Groups
where all 8 attempts got the same reward are skipped. One training datum per attempt holds the prompt and response
tokens, the sampling logprobs, the advantage, and weights (0 for prompt tokens, 1 for response tokens).
`forward_backward` with the `ppo` loss ([proximal policy optimization](https://arxiv.org/abs/1707.06347)) and `optim_step` update the adapter, and
`save_weights_and_get_sampling_client` gives a sampler for the new weights.

{{GRPO_STEPS}} steps of 8 problems are not enough to change the held-out pass rate; section 5 measures an adapter trained
for 20 steps of 64 problems.

**Run time:** about 5 to 10 minutes per step, so about 10 to 20 minutes for {{GRPO_STEPS}} steps.
""")
code("""
def build_datum(prompt, response_tokens, logprobs, advantage):
    \"\"\"Return one training example for forward_backward, in the format the training session API expects.

    prompt is the prompt token ids, response_tokens and logprobs come from one sampled sequence, and advantage is
    that sequence's advantage. The model is trained to predict each next token, so the inputs are all tokens but
    the last and the targets are all tokens but the first. Per position the loss also gets:
      logprobs   - the logprob of the token when it was sampled (0 for prompt positions); the ppo loss uses them to
                   compare the current weights with the weights that produced the sample
      advantages - the response's advantage on every response position, 0 on prompt positions
      weights    - 0 for prompt positions and 1 for response positions, so only the response is trained

    Returns a dict with modelInput (the input tokens) and lossFnInputs (target_tokens, logprobs, advantages and
    weights, each with data, dtype and shape). All arrays have one entry per input position. The function has no
    side effects.
    \"\"\"
    tokens = prompt + list(response_tokens)
    inputs, targets = tokens[:-1], tokens[1:]
    n_prompt = len(prompt) - 1  # positions whose target is still a prompt token
    n_resp = len(inputs) - n_prompt  # positions whose target is a response token
    seq = len(inputs)
    return {
        "modelInput": {"chunks": [{"tokens": inputs, "type": "encoded_text"}]},
        "lossFnInputs": {
            "target_tokens": {"data": targets, "dtype": "int64", "shape": [seq]},
            "logprobs": {"data": [0.0] * n_prompt + list(logprobs), "dtype": "float32", "shape": [seq]},
            "advantages": {"data": [0.0] * n_prompt + [advantage] * n_resp, "dtype": "float32", "shape": [seq]},
            "weights": {"data": [0.0] * n_prompt + [1.0] * n_resp, "dtype": "float32", "shape": [seq]},
        },
    }


STEPS, PROBLEMS_PER_STEP, K = {{GRPO_STEPS}}, 8, 8  # training steps, problems per step, attempts per problem
ADAM = AdamParams(learning_rate=1e-5, beta2=0.95, grad_clip_norm=1.0)  # optimizer settings for optim_step
rng = random.Random(0)  # fixed seed, so every run picks the same training problems

for step in range(STEPS):
    t0 = time.time()
    batch = rng.sample(train_tasks, PROBLEMS_PER_STEP)  # PROBLEMS_PER_STEP distinct problems for this step
    prompts = [prompt_tokens(t) for t in batch]
    # One sample() call per problem returns K attempts for it.
    futures = [sampling_client.sample(p, num_samples=K, sampling_params=SAMPLING) for p in prompts]
    results = await gather([asyncio.to_thread(f.result) for f in futures], f"step {step} sampling")
    # Grade all attempts of all problems at once. graded is in order: K entries per problem.
    graded = await gather([grade(t, tokenizer.decode(s.tokens)) for t, r in zip(batch, results) for s in r.sequences],
                          f"step {step} grading")

    datums, rewards_all, skipped = [], [], 0
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        seqs = result.sequences
        rewards = np.array([r for r, _ in graded[i * K:(i + 1) * K]])
        rewards_all += list(rewards)
        if rewards.std() < 1e-6:  # all K attempts got the same reward: no signal, skip the group
            skipped += 1
            continue
        advantages = (rewards - rewards.mean()) / rewards.std()  # reward relative to the group
        datums += [build_datum(prompt, s.tokens, s.logprobs, float(a)) for s, a in zip(seqs, advantages)]

    if datums:
        # forward_backward computes the loss and gradients; optim_step applies them. Both return futures, so the
        # optimizer step is queued right behind the gradient computation.
        fb = training_client.forward_backward(datums, loss_fn="ppo")
        opt = training_client.optim_step(ADAM)
        await asyncio.to_thread(fb.result)
        await asyncio.to_thread(opt.result)
        # Snapshot the updated weights and get a sampler for them, so the next step samples from the new model.
        sampling_client = await asyncio.to_thread(training_client.save_weights_and_get_sampling_client)
    print(f"step {step}  train reward {np.mean(rewards_all):.3f}  skipped groups {skipped}/{PROBLEMS_PER_STEP}  "
          f"datums {len(datums)}  {time.time() - t0:.0f}s")
""")
md("Stop this session before opening the next one.")
code("""
# Stop the session and release its resources.
training_client.stop()
""")

md("""
## 5. Evaluate the pre-trained adapter

The adapter was trained with the cookbook's Code RL recipe: 20 steps of 64 problems x 8 attempts on
`{{MODEL_SHORT}}`, LoRA rank 32, about 6 hours. Its files (`adapter_config.json`, `adapter_model.safetensors`) are in
S3. To use them, copy them to your bucket and register them as a model package in your account with the layout that
`save_weights_for_sampler` produces. A new session then starts from these weights with
`create_training_client_from_state`.

The same steps move an adapter between accounts or Regions, and let you continue training from an adapter that a
long-running script produced.

**Run time:** about 1 minute to copy the 0.9 GB adapter and register it.
""")
code("""
from urllib.parse import urlparse

s3 = session.client("s3")
src = urlparse(PRETRAINED_ADAPTER)  # src.netloc is the bucket, src.path the key prefix
dest_prefix = "training-sessions/code-rl/pretrained/model/"
# Copy every object under the source prefix to the same relative key under dest_prefix in your bucket.
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=src.netloc, Prefix=src.path.lstrip("/")):
    for obj in page.get("Contents", []):
        rel = obj["Key"][len(src.path.lstrip("/")):]
        # s3.copy is a managed copy within S3; the object data does not pass through the notebook.
        s3.copy({"Bucket": src.netloc, "Key": obj["Key"]}, f"sagemaker-{REGION}-{ACCOUNT}", dest_prefix + rel)

# Register the copied files as a model package version in the code-rl group. BaseModel identifies the base model
# the adapter was trained on, by hub content name and version.
package_arn = sm.create_model_package(
    ModelPackageGroupName="code-rl",
    ModelPackageDescription="pre-trained Code RL adapter",
    ModelApprovalStatus="Approved",
    InferenceSpecification={"Containers": [{
        "ModelDataSource": {"S3DataSource": {"S3Uri": f"s3://sagemaker-{REGION}-{ACCOUNT}/{dest_prefix}",
                                             "S3DataType": "S3Prefix", "CompressionType": "None"}},
        "IsCheckpoint": False,
        "BaseModel": {{BASE_MODEL_SPEC}},
    }]},
)["ModelPackageArn"]
print(package_arn)
""")
md("""
`create_sampling_client()` on a client created from a model package samples the base weights. Call
`save_weights_and_get_sampling_client()` once to get a sampler for the loaded adapter.

**Run time:** about 6 to 10 minutes: opening the session, then the same evaluation as in section 3.
""")
code("""
# Start a new training session from the registered adapter's weights.
pretrained_client = svc.create_training_client_from_state(
    model_package_arn=package_arn, training_session_name=f"code-rl-pretrained-{int(time.time())}", rank=32, alpha=64)
# Snapshot the loaded weights and get a sampler for them.
pretrained_sampler = pretrained_client.save_weights_and_get_sampling_client()

t0 = time.time()
pretrained_pass, pretrained_texts = await evaluate(pretrained_sampler, test_tasks)
print(f"pass rate: base {base_pass:.1%} -> pre-trained adapter {pretrained_pass:.1%} "
      f"({len(test_tasks)} held-out problems, 2 attempts each, {time.time() - t0:.0f}s)")
""")
code("""
# Stop the session and close the Code Interpreter sessions. The code interpreter and the model package are kept.
pretrained_client.stop()
sandbox.close()
""")

md("""
## Use this for your own task

Replace three things and keep the sandbox and the training loop:

| Part | In this notebook | For your task |
|---|---|---|
| Problems | DeepCoder competitive programming problems, each with test cases | Your tasks, each with an automatic check: unit tests, a SQL query against a small database with the expected rows, a schema or configuration validator |
| Check | `grade`: extract the code, run the tests in the sandbox | Run your check in the sandbox. Return 1.0 when it passes; partial credit is possible, for example the share of tests passed |
| Prompt | The problem text through the chat template | Your instructions, context and output format |

Files the check needs, such as a database file or a test suite, travel with each request in its `files` field,
as `test_cases.txt` does here. Measure the base model first: training needs tasks that the base model solves
sometimes and fails sometimes. Here it passed about 12% before training.
""")
md("""
## From this notebook to production

- **Long training runs belong in a script.** The adapter in section 5 was trained for about 6 hours with the
  cookbook's Code RL recipe, which runs from the command line (`examples/rl_coderl.sh` in the cookbook). `save_state` and
  `create_training_client_from_state_with_optimizer` let a run continue in a new session.
- **The sandbox scales with sessions.** `CodeInterpreterSandbox` uses 16 sessions here. The default Code Interpreter
  quotas are 1,000 concurrent sessions per account and 30 `InvokeCodeInterpreter` requests per second, so more
  sessions grade more responses in parallel.
- **The adapter deploys like other models.** It is registered as a model package and can be deployed to a SageMaker
  endpoint. The [training sessions cookbook](https://github.com/aws/sagemaker-training-sessions-cookbook) has a
  deployment notebook that serves several adapters on one endpoint.
- **Run generated code in a sandbox in production too.** If the trained model's code runs in your application,
  run it in Code Interpreter or another isolated environment, as during training.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / V["OUT"]
nbf.write(nb, out)
print("wrote", out)

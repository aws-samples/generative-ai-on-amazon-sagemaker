"""Build 5a_gsm_hard_grpo.ipynb (Qwen3.8-27B, GSM-Hard)."""
import textwrap
from pathlib import Path

import nbformat as nbf

V = dict(MODEL="Qwen/Qwen3.8-27B", MODEL_SHORT="Qwen3.8-27B", DATASET="GSM-Hard", EFFORT="low",
         TAG="gsm-hard", OUT="5a_gsm_hard_grpo.ipynb", PROMPTS="16", LR="4e-5")

cells = []


def fill(s):
    """Replace the {{KEY}} placeholders in s with the values in V, then dedent and strip blank edge lines."""
    for k, v in V.items():
        s = s.replace("{{" + k + "}}", v)
    return textwrap.dedent(s).strip("\n")


def md(s):
    """Append a markdown cell with the filled text s."""
    cells.append(nbf.v4.new_markdown_cell(fill(s)))


def code(s):
    """Append a code cell with the filled text s."""
    cells.append(nbf.v4.new_code_cell(fill(s)))


md("""
# Write your own reinforcement learning loop with [SageMaker AI training sessions](https://docs.aws.amazon.com/sagemaker/latest/dg/training-sessions-getting-started.html): {{DATASET}}

> **Note:** This is a toy problem. It is small enough to run in minutes, so you can see the mechanics of a
> reinforcement learning loop and the parts of the training session APIs. It does not show how to build a
> production math model.

With a managed RLVR (reinforcement learning from verifiable rewards) job, you supply a
dataset and a reward function, and the service runs the training loop. In this notebook you write that loop yourself
on a **[training session](https://aws.amazon.com/blogs/machine-learning/write-your-own-training-loop-without-managing-gpus-with-amazon-sagemaker-training-sessions/)**,
a live training client for a [LoRA (low-rank adaptation)](https://arxiv.org/abs/2106.09685) adapter on
[`{{MODEL_SHORT}}`](https://huggingface.co/{{MODEL}}),
with three operations:

| Operation | What it does |
|---|---|
| `sample` | Generate responses from the current weights |
| `forward_backward` | Compute the loss and gradients for a batch of token sequences |
| `optim_step` | Apply the gradients (AdamW) |

### The task

The model solves math word problems from [GSM-Hard](https://huggingface.co/datasets/reasoning-machines/gsm-hard).
GSM-Hard takes the grade-school word problems of [GSM8K](https://huggingface.co/datasets/openai/gsm8k) and replaces their numbers with much larger ones, so the
reasoning stays simple but the arithmetic gets long. An example from the training set:

> James is in charge of running messages from the office to each teacher's classroom. If he delivers 428598 messages
> to Ms. Thompson and 1/3 as many messages to Mr. Yu, how many messages does he deliver total?
>
> Answer: 571464

The model thinks with {{EFFORT}} reasoning effort and has a budget of 512 generated tokens for thinking and answer
together. It must end with the final number after `####`. The reward is 1.0 for the correct number and 0.0 otherwise,
including when the response runs out of tokens before the answer. You measure two things on 100 held-out problems:
accuracy, and the share of answers cut off at 512 tokens.

### What this notebook is for

The task is small, so the training loop runs in about 15 to 20 minutes and every part of it is a cell you can read and
change: the reward, the advantage, the loss, the batch size. Use it to see how a
[GRPO (Group Relative Policy Optimization)](https://arxiv.org/abs/2402.03300) loop maps onto the training session APIs, and as a starting point to try your own reward function or
training algorithm. The last section lists what to change for that.

| Step | Time |
|---|---|
| 1. Setup | 5 min |
| 2. Data, prompt and reward | 5 min |
| 3. Open a training session, baseline | 5 min |
| 4. The GRPO loop | 20 min |
| 5. Evaluate, save, clean up | 5 min |
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
  `gsm8k-grpo`. Every adapter you save is registered there in
  [Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html) as a
  [model package version](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-version.html). The cell creates the group, or reuses it if it already exists.
- **`svc`** is the `ServiceClient`, the entry point to the training session APIs. It creates the session in
  section 3.
""")
code("""
import json, random, re, time
import boto3
import numpy as np
from sagemaker.train.training_session import AdamParams, SamplingParams, ServiceClient

# The default boto3 session gives the Region and credentials of this notebook.
session = boto3.Session()
REGION = session.region_name
ACCOUNT = session.client("sts").get_caller_identity()["Account"]
# IAM role that the training session service assumes to read the base model and write adapters to S3.
ROLE_ARN = session.client("iam").get_role(RoleName="SageMakerTrainingSessionRole")["Role"]["Arn"]
# S3 prefix where the service writes the adapter files.
S3_OUTPUT = f"s3://sagemaker-{REGION}-{ACCOUNT}/training-sessions/gsm8k/"

# Saved adapters are registered as versions in a model package group. Create the group, or look it up
# if it already exists (create_model_package_group raises a ClientError then).
sm = session.client("sagemaker")
try:
    MPG_ARN = sm.create_model_package_group(ModelPackageGroupName="gsm8k-grpo")["ModelPackageGroupArn"]
except sm.exceptions.ClientError:
    MPG_ARN = sm.describe_model_package_group(ModelPackageGroupName="gsm8k-grpo")["ModelPackageGroupArn"]

# ServiceClient holds the account-level settings and opens training sessions (section 3).
svc = ServiceClient(region=REGION, role_arn=ROLE_ARN, model_package_group_arn=MPG_ARN, s3_output_path=S3_OUTPUT)
print(REGION, ROLE_ARN, MPG_ARN, sep="\\n")
""")

md("""
## 2. Data, prompt and reward

GSM-Hard is GSM8K with the numbers in each problem replaced by larger ones, so the arithmetic is harder. Some
substitutions produce answers that do not fit the problem (negative or fractional counts), so only problems with a
non-negative whole-number answer are kept. The dataset has a single split; the first 100 shuffled problems are used
for evaluation and the next 400 for training.
""")
code("""
from datasets import load_dataset

# GSM-Hard has a single split. Shuffle it with a fixed seed so every run uses the same problems.
gsm_hard = load_dataset("reasoning-machines/gsm-hard")["train"].shuffle(seed=0)
# Keep (question, answer) pairs whose answer is a non-negative whole number, at most 500 of them.
problems = [(r["input"], str(int(r["target"]))) for r in gsm_hard
        if r["target"] >= 0 and float(r["target"]).is_integer()][:500]
# The first 100 are held out for evaluation; the other 400 are used for training.
eval_problems, train_problems = problems[:100], problems[100:]
print(len(train_problems), "train |", len(eval_problems), "eval")
print(train_problems[0])
""")
md("""
The prompt uses the model's [chat template](https://huggingface.co/docs/transformers/chat_templating). The reward is 1.0 when the number after the last `####` matches the
ground truth, 0.0 otherwise.
""")
code("""
def build_prompt(question, tokenizer):
    \"\"\"Return the prompt text for one problem, formatted with the model's chat template.

    tokenizer is the model's tokenizer, which you get from the training session in section 3.

    apply_chat_template wraps the message in the special tokens the model was trained with (the user turn, then
    the start of the assistant turn). add_generation_prompt=True ends the text where the model starts its answer.
    tokenize=False returns a string; it is tokenized later with tokenizer.encode. reasoning_effort="low" is an
    option of the Qwen3.8 chat template that asks for a short thinking phase.
    \"\"\"
    # One user message: the problem plus the instruction for the answer format that reward_fn checks.
    messages = [{"role": "user", "content": f"{question}\\n\\nSolve this step by step, then give the final numeric "
                                            "answer on its own line after \\"#### \\"."}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, reasoning_effort="{{EFFORT}}")


def extract_answer(text):
    \"\"\"Return the number after the last "####" in the last 300 characters of a response, or None.

    Commas and a trailing period are removed, so "#### 1,234." gives "1234".
    \"\"\"
    # Optional "$", then a number with optional minus sign, thousands commas and decimals.
    found = re.findall(r"####\\s*\\$?\\s*(-?[0-9][0-9,]*\\.?[0-9]*)", text[-300:])
    return found[-1].replace(",", "").rstrip(".") if found else None


def reward_fn(text, truth):
    \"\"\"Return 1.0 if the response's final answer equals the ground truth, else 0.0.

    The comparison allows a relative difference of 1e-6, because GSM-Hard answers can have many digits.
    A response without a "####" answer, for example one cut off at the token budget, gets 0.0.
    \"\"\"
    # Compare as numbers, so "571464" and "571464.0" are equal.
    answer = extract_answer(text)
    try:
        return float(answer is not None and abs(float(answer) - float(truth)) <= 1e-6 * max(1.0, abs(float(truth))))
    except ValueError:
        return 0.0
""")

md("""
## 3. Open a training session and measure the base model

`create_lora_training_client` starts the session and returns when it is running. `create_sampling_client` returns
a sampler for the current weights (the base model, before any update).

**Run time:** from under a minute to a few minutes, until the session is running.
""")
code("""
# Start a training session with a LoRA adapter (rank 32) on the base model. The call returns when the session is
# running; it counts against your concurrent-session quota until training_client.stop() at the end of the notebook.
training_client = svc.create_lora_training_client(
    "{{MODEL}}", training_session_name=f"{{TAG}}-grpo-{int(time.time())}", rank=32, alpha=64)
# The model's tokenizer, loaded from the session. It turns text into token ids and back.
tokenizer = training_client.get_tokenizer()
# A sampler for the current weights. Before any update these are the base model's weights.
sampling_client = training_client.create_sampling_client()
print("session:", training_client.training_session_name)
""")
md("""
### One example: tokenizer and sampler

The model reads and writes token ids, not text. The tokenizer converts between the two. The cell below builds the
prompt for one problem, shows the text the chat template produces, and encodes it into token ids.
""")
code("""
# The first held-out problem and its correct answer.
question, truth = eval_problems[0]
prompt_text = build_prompt(question, tokenizer)
print(prompt_text)

# Turn the prompt text into token ids. add_special_tokens=False because the chat template already
# added the special tokens.
prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
print(len(prompt_tokens), "tokens. The first 20 ids:", prompt_tokens[:20])
# Decoding each id on its own shows which piece of text it stands for.
print("as text pieces:", [tokenizer.decode([t]) for t in prompt_tokens[:20]])
""")
md("""
The sampler sends the token ids to the model and returns the generated tokens. Each sampled sequence has the token
ids (`tokens`), the log probability the model assigned to each of them (`logprobs`), and why sampling stopped
(`stop_reason`). Decoding the tokens gives the response text, which the reward function scores.
""")
code("""
# sample() returns a future; result() waits for the response.
result = sampling_client.sample(prompt_tokens, num_samples=1,
                                sampling_params=SamplingParams(max_tokens=512, temperature=0.7, top_p=0.9)).result()
# result.sequences holds num_samples sequences; here there is one.
seq = result.sequences[0]
print(len(seq.tokens), "tokens generated, stop reason:", seq.stop_reason)
print("first 5 token ids:", seq.tokens[:5])
print("their logprobs:   ", [round(lp, 3) for lp in seq.logprobs[:5]])

# Turn the generated token ids back into text, then score it like the training loop does.
response_text = tokenizer.decode(seq.tokens)
print("\\n--- response ---\\n" + response_text)
print("--- end ---\\n")
print("extracted answer:", extract_answer(response_text), "| ground truth:", truth,
      "| reward:", reward_fn(response_text, truth))
""")
md("""
The evaluation below does the same for every evaluation problem, twice each. Sampling is random at temperature
0.7, so each problem is sampled twice to reduce the noise in the accuracy. `sample` returns a future right away,
so all requests are submitted before waiting on any of them.

**Run time:** about 1 minute for 200 responses.
""")
code("""
from sagemaker.train.training_session import StopReason
from tqdm.auto import tqdm

MAX_TOKENS = 512  # token budget for thinking and answer together
# Sampling settings for evaluation: max_tokens caps the response length, temperature and top_p control randomness.
SAMPLING = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.7, top_p=0.9)


def evaluate(problems, samples_per_problem=2):
    \"\"\"Sample each problem samples_per_problem times with the current sampler and score the responses.

    Returns (accuracy, share of responses cut at MAX_TOKENS, mean response length in tokens).
    All requests are submitted first: sample() returns a future right away, and result() waits for it.
    \"\"\"
    # tokenizer.encode turns the prompt text into token ids. add_special_tokens=False because the chat template
    # already added the special tokens.
    futures = [sampling_client.sample(tokenizer.encode(build_prompt(q, tokenizer), add_special_tokens=False),
                                      num_samples=samples_per_problem, sampling_params=SAMPLING) for q, _ in problems]
    results = [f.result() for f in tqdm(futures, desc="evaluation", leave=False)]
    # Each result holds num_samples sequences. A sequence has the sampled token ids (tokens), the logprob of each
    # token (logprobs) and why sampling stopped (stop_reason).
    seqs = [(s, truth) for (_, truth), r in zip(problems, results) for s in r.sequences]
    # tokenizer.decode turns the sampled token ids back into text for the reward function.
    accuracy = float(np.mean([reward_fn(tokenizer.decode(s.tokens), truth) for s, truth in seqs]))
    # StopReason.LENGTH: the response reached MAX_TOKENS before the model ended it.
    cut = float(np.mean([s.stop_reason == StopReason.LENGTH for s, _ in seqs]))
    # Mean response length in generated tokens.
    tokens = float(np.mean([len(s.tokens) for s, _ in seqs]))
    return accuracy, cut, tokens


# Score the base model on the 100 held-out problems, 2 responses each.
t0 = time.time()
base_accuracy, base_cut, base_tokens = evaluate(eval_problems)
print(f"base model: accuracy {base_accuracy:.1%}, answers cut at {MAX_TOKENS} tokens {base_cut:.1%}, "
      f"mean length {base_tokens:.0f} tokens ({time.time() - t0:.0f}s)")
""")

md("""
## 4. The GRPO loop

GRPO compares the model's answers with each other: for each problem, the model answers several times, and the
answers that score better than the others in their group are made more likely. No separate value model or reward
model is trained.

![One GRPO step](images/grpo.svg)

The figure shows one problem in one step. The model answers it 8 times; 4 answers are correct (reward 1) and 4 are
not (reward 0). The group mean is 0.5, so the correct answers get an advantage of +1 and the wrong ones -1. The update
raises the probability of the +1 answers and lowers the probability of the -1 answers. If all 8 answers had the same
reward, all advantages would be 0 and the problem would add nothing to the update, so it is skipped.

Each step of the loop below does this for {{PROMPTS}} problems at once:

1. Take {{PROMPTS}} training problems and sample 8 responses for each (one `sample` call with `num_samples=8`).
2. Score every response. Within each group of 8, the advantage of a response is its reward minus the group mean,
   divided by the group standard deviation. Groups where all 8 responses scored the same are skipped.
3. Build one training datum per response: prompt and response tokens, the response tokens' sampling logprobs,
   the advantage on every response token, and weights (0 for prompt tokens, 1 for response tokens).
4. `forward_backward` with the `ppo` loss ([proximal policy optimization](https://arxiv.org/abs/1707.06347)), then `optim_step`.
5. `save_weights_and_get_sampling_client` so the next step samples from the updated weights.

**Run time:** about 15 to 20 minutes for 12 steps. A step takes 30 seconds to 3.5 minutes; steps where most groups
are skipped are faster.
""")
code("""
def build_datum(prompt_tokens, response_tokens, logprobs, advantage):
    \"\"\"Return one training example for forward_backward, in the format the training session API expects.

    The model is trained to predict each next token, so the inputs are all tokens but the last and the targets are
    all tokens but the first. Per position the loss also gets:
      logprobs   - the logprob of the token when it was sampled (0 for prompt positions); the ppo loss uses them to
                   compare the current weights with the weights that produced the sample
      advantages - the response's advantage on every response position, 0 on prompt positions
      weights    - 0 for prompt positions and 1 for response positions, so only the response is trained
    \"\"\"
    # The full sequence: prompt followed by the sampled response.
    tokens = prompt_tokens + list(response_tokens)
    # Next-token prediction: the target at each position is the following token.
    inputs, targets = tokens[:-1], tokens[1:]
    n_prompt = len(prompt_tokens) - 1  # positions whose target is still a prompt token
    n_resp = len(inputs) - n_prompt    # positions whose target is a response token
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
""")
code("""
STEPS, PROMPTS_PER_STEP, K = 12, {{PROMPTS}}, 8  # training steps, problems per step, responses per problem
TRAIN_SAMPLING = SamplingParams(max_tokens=MAX_TOKENS, temperature=1.0)  # temperature 1.0 for varied responses
ADAM = AdamParams(learning_rate={{LR}}, beta2=0.95, grad_clip_norm=1.0)  # optimizer settings for optim_step
# Fixed seed, so every run draws the same training problems in the same order.
rng = random.Random(0)
# Mean training reward per step, for the plot below.
history = []

for step in range(STEPS):
    t0 = time.time()
    # 1. Pick PROMPTS_PER_STEP training problems and build their prompts as token ids.
    batch = rng.sample(train_problems, PROMPTS_PER_STEP)
    prompts = [tokenizer.encode(build_prompt(q, tokenizer), add_special_tokens=False) for q, _ in batch]
    # 2. Sample. One sample() call per problem returns K responses for it; all requests are submitted
    # before waiting on any of them.
    futures = [sampling_client.sample(p, num_samples=K, sampling_params=TRAIN_SAMPLING) for p in prompts]

    # 3. Score each group of K responses, compute advantages, and build one training datum per response.
    datums, rewards_all, skipped = [], [], 0
    for prompt, (_, truth), future in zip(prompts, tqdm(batch, desc=f"step {step} sampling", leave=False), futures):
        # Wait for this problem's K responses.
        seqs = future.result().sequences
        # Reward of each response: 1.0 for the correct final answer, 0.0 otherwise.
        rewards = np.array([reward_fn(tokenizer.decode(s.tokens), truth) for s in seqs])
        rewards_all += list(rewards)
        if rewards.std() < 1e-6:  # all K responses got the same reward: no signal, skip the group
            skipped += 1
            continue
        advantages = (rewards - rewards.mean()) / rewards.std()  # reward relative to the group
        # Every response of the group becomes one datum with its own advantage.
        datums += [build_datum(prompt, s.tokens, s.logprobs, float(a)) for s, a in zip(seqs, advantages)]

    # 4. Update the adapter, unless every group was skipped.
    if datums:
        # forward_backward computes the loss and gradients; optim_step applies them. Both return futures, so the
        # optimizer step is queued right behind the gradient computation.
        fb = training_client.forward_backward(datums, loss_fn="ppo")
        opt = training_client.optim_step(ADAM)
        fb.result(); opt.result()
        # Snapshot the updated weights and get a sampler for them, so the next step samples from the new model.
        sampling_client = training_client.save_weights_and_get_sampling_client()
    # Mean reward over all sampled responses of this step (the training accuracy at temperature 1.0).
    history.append(float(np.mean(rewards_all)))
    print(f"step {step:2d}  train accuracy {history[-1]:.1%}  skipped groups {skipped}/{PROMPTS_PER_STEP}  "
          f"datums {len(datums):3d}  {time.time() - t0:.0f}s")
""")
code("""
import matplotlib.pyplot as plt

# Training accuracy per step. It varies from step to step because each step uses different problems.
plt.plot(history, marker="o")
plt.xlabel("step"); plt.ylabel("train accuracy (temperature 1.0)"); plt.grid(alpha=0.3)
plt.show()
""")

md("## 5. Evaluate, save, clean up\n\n**Run time:** about 1 minute for the evaluation.")
code("""
# Score the trained adapter on the same 100 held-out problems, 2 responses each. sampling_client
# now samples from the weights after the last update.
t0 = time.time()
trained_accuracy, trained_cut, trained_tokens = evaluate(eval_problems)
print(f"accuracy      {base_accuracy:.1%} -> {trained_accuracy:.1%}")
print(f"cut answers   {base_cut:.1%} -> {trained_cut:.1%}")
print(f"mean length   {base_tokens:.0f} -> {trained_tokens:.0f} tokens   ({time.time() - t0:.0f}s)")
""")
md("""
`save_weights_for_sampler` registers the adapter as a model package in the model package group; it can be deployed
to a SageMaker endpoint like any other model package. `save_state` also stores the optimizer state, so a later session can continue
training with `create_training_client_from_state_with_optimizer`.
""")
code("""
# Save the current adapter and register it as a model package in the model package group (MPG_ARN).
# save_weights_for_sampler returns a future; result() waits until the model package is registered.
weights = training_client.save_weights_for_sampler(description="{{TAG}} grpo").result()
print("adapter:", weights.model_package_arn)
""")
md("Stop the session. It counts against your concurrent-session quota until it is stopped.")
code("""
# Stop the session and release its resources. The saved model package is kept.
training_client.stop()
""")
md("""
## Try your own reward or algorithm

Change one part of the loop, run sections 3 to 5 again, and compare accuracy and cut answers with this run:

| To try | Change |
|---|---|
| A different reward | `reward_fn`, for example partial credit or a length penalty |
| A different advantage | The `advantages = ...` line, for example no division by the standard deviation |
| A different loss | `loss_fn="ppo"` in `forward_backward`; the SDK also offers `importance_sampling` and `cispo` |
| Token-level credit | The `weights` and `advantages` arrays in `build_datum` |
| More or less data per step | `PROMPTS_PER_STEP` and `K` |
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / V["OUT"]
nbf.write(nb, out)
print("wrote", out)

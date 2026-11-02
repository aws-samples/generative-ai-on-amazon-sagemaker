"""Build 5c_scheduling_agent.ipynb (Qwen3.8-27B, thinking off)."""
from pathlib import Path

import nbformat as nbf

cells = []


def md(s):
    """Append a markdown cell with the text s, without leading and trailing blank lines."""
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    """Append a code cell with the source s, without leading and trailing blank lines."""
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md("""
# Train a scheduling agent with SageMaker AI training sessions

In this notebook you train the model behind a [Strands](https://strandsagents.com) agent with reinforcement
learning (RL). The agent books meetings. A user asks for a meeting with several people on a given day; the agent
reads each attendee's calendar with a tool, finds the earliest start time that fits everyone's working hours across
time zones, and books it. It is a multi-turn agent: one request takes several model calls and tool calls.

The task stands for a common type of business agent: one that collects data through tools and has to combine it
correctly under rules. Examples are scheduling, resource allocation, eligibility checks and order changes. The
same notebook trains such an agent if you replace the tools, the requests and the reward.

### How the pieces fit together

![Agent training loop](images/agent-loop.svg)

- **Strands agent** (your code): a normal Strands agent with two [tools](https://strandsagents.com/docs/user-guide/sdk/tools/),
  `get_calendar` and `book_meeting`.
- **Rollout gateway** ([AgentCore RL Toolkit](https://github.com/awslabs/agentcore-rl-toolkit)): runs on
  `127.0.0.1` in this kernel and serves an OpenAI-compatible chat API. The agent sends its model calls there
  through the Strands [OpenAI model provider](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/openai/).
  The gateway turns the chat messages into tokens with the model's
  [chat template](https://huggingface.co/docs/transformers/chat_templating), samples from the training session,
  parses tool calls, and records the token ids and logprobs of every call.
- **Training session**
  ([SageMaker AI training sessions](https://docs.aws.amazon.com/sagemaker/latest/dg/training-sessions-getting-started.html)):
  holds [`Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) with a [LoRA](https://arxiv.org/abs/2106.09685)
  (low-rank adaptation) adapter, generates tokens, and runs `forward_backward` and `optim_step`.
- **GRPO step** ([Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300), your code):
  scores each episode from the booking the agent made and turns the recorded model calls into training data.

The model runs with thinking turned off (`enable_thinking=False` in the chat template), so each model call answers
directly with text or a tool call.

| Step | Time |
|---|---|
| 1. Setup | 5 min |
| 2. The calendars and the agent | 10 min |
| 3. Open a training session, measure the base model | 5 min |
| 4. Train with GRPO | 30 to 60 min |
| 5. Evaluate, save the adapter | 5 to 10 min |
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
- **`ROLE_ARN`** is the IAM (AWS Identity and Access Management) role `SageMakerTrainingSessionRole`. The training session service assumes it to read the
  base model and write adapters to Amazon S3 (Amazon Simple Storage Service). The workshop stack creates it.
- **`S3_OUTPUT`** is the S3 prefix where the service writes the adapter files, in your
  `sagemaker-<region>-<account>` bucket.
- **`MPG_ARN`** is a [model package group](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-model-group.html),
  `scheduling-agent`, in [SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html).
  Every adapter you save is registered there as a
  [model package version](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-version.html). The cell creates the group, or reuses it if it already exists.
- **`BASE_MODEL_ARN`** identifies `Qwen3.8-27B` in the SageMaker Public Hub. `TrainingRun` opens the session with it
  in section 3.
- **Logging:** the `lab` helpers report at `INFO` level when the session is ready or a weight refresh fails.
""")
code("""
import asyncio, json, logging, random, time
import boto3
import numpy as np

logging.basicConfig(level=logging.WARNING)
# Show INFO messages from the lab helpers (lab/runtime.py) only.
logging.getLogger("lab").setLevel(logging.INFO)

session = boto3.Session()
REGION = session.region_name
ACCOUNT = session.client("sts").get_caller_identity()["Account"]
BASE_MODEL_ARN = f"arn:aws:sagemaker:{REGION}:aws:hub-content/SageMakerPublicHub/Model/huggingface-vlm-qwen3-8-27b/1.3.0"
ROLE_ARN = session.client("iam").get_role(RoleName="SageMakerTrainingSessionRole")["Role"]["Arn"]
S3_OUTPUT = f"s3://sagemaker-{REGION}-{ACCOUNT}/training-sessions/scheduling-agent/"

sm = session.client("sagemaker")
# Create the model package group, or look up its ARN if it already exists.
try:
    MPG_ARN = sm.create_model_package_group(ModelPackageGroupName="scheduling-agent")["ModelPackageGroupArn"]
except sm.exceptions.ClientError:
    MPG_ARN = sm.describe_model_package_group(ModelPackageGroupName="scheduling-agent")["ModelPackageGroupArn"]
print(REGION, ROLE_ARN, MPG_ARN, sep="\\n")
""")

md("""
## 2. The calendars and the agent

The environment is generated in Python, so it is fast, deterministic and free to call many times:

- **People:** 20 people in nine cities, including the half-hour offsets of Mumbai and Adelaide. Each has their
  own working hours (for example 08:00-16:00 or 12:00-20:00 local time) and one to four busy blocks on the
  requested day. Some calendars list busy blocks in local time, others in UTC.
- **Requests:** a meeting with 3 to 5 attendees and a duration of 30 to 90 minutes, for example
  "Please book a 60-minute meeting with Hugo, Greta and Elif on Thursday 2026-11-12, as early as possible."
- **Correct answer:** the earliest start in UTC that fits every attendee, found by checking every 15-minute step
  of the day. About 6% of the requests have no possible slot; the agent should then book `none`.
- **Train and test** requests use different people and calendars, so the test measures whether the agent applies
  the rules, not whether it remembers the data.

The next cell loads the requests and prints one with its correct answer.
""")
code("""
from usecases.scheduling_hard import load, people_for, valid_for, _fmt

# Each task is a dict with the request text ("message"), the attendees, the duration, the seed of its calendars
# and the earliest valid start ("expected", minutes after 00:00 UTC, or None when no slot exists).
train_tasks = load("train", 512)
test_tasks = load("test", 128)
task = test_tasks[0]
print(task["message"])
print("earliest slot (UTC):", _fmt(task["expected"]) if task["expected"] is not None else "none")
""")

md("""
### What the agent does

The agent is multi-turn: in one episode it calls the model several times. Each tool result goes back to the model
as the next turn, until the model answers without a tool call.

![One episode of the scheduling agent](images/agent-episode.svg)

The figure shows a typical episode for a request from the test set; in the reference runs, an episode took about three
model calls and four tool calls on average. The model first asks for the three calendars in one call. With the results, it
converts everyone's working hours and busy blocks to UTC (one calendar is already in UTC), finds the earliest
90-minute gap, and books 20:00 UTC (14:00 in Chicago). A third model call confirms the booking in text. The reward
is computed after the episode from the booking that `book_meeting` recorded. During training, the gateway joins the
three model calls into one token sequence, which becomes one training datum. The loss covers only the tokens the
model generated, not the tool results.

The agent is a plain Strands agent with two tools defined with the
[`@tool` decorator](https://strandsagents.com/docs/user-guide/sdk/tools/custom-tools/). `run_agent` builds it for one request, runs it, and scores the
single booking it made: 1.0 for the earliest valid slot, 0.5 for a valid but later slot, 0 for an invalid slot or
when it books zero or several times. The model comes from `model_config`: during training that is the rollout
gateway, but the same function runs against
[Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html).
""")
code("""
from datetime import datetime

# Strands classes: Agent runs the model and tool loop, @tool turns a Python function into a tool the model can call,
# OpenAIModel talks to an OpenAI-compatible chat API (the rollout gateway), BedrockModel talks to Amazon Bedrock.
from strands import Agent, tool
from strands.agent.conversation_manager import NullConversationManager
from strands.models.openai import OpenAIModel
from strands.models import BedrockModel

SYSTEM_PROMPT = (
    "You schedule meetings. Each person has their own working hours in their local time zone. Calendars list busy "
    "blocks either in the person's local time or in UTC, as stated in the calendar. Find the earliest start time on "
    "the requested date (in UTC, a multiple of 15 minutes) where the meeting fits inside every attendee's working "
    "hours and overlaps none of their busy blocks. Then call book_meeting once with the start time in UTC. If no "
    "slot exists that day, call book_meeting with start_utc set to 'none'."
)


def make_tools(people, bookings):
    \"\"\"Return the two tools of the agent, get_calendar and book_meeting, for one request.

    Args:
        people: Dict that maps each first name to that person's calendar (city, UTC offset, working hours, busy
            blocks and calendar_in_utc), as returned by people_for.
        bookings: List that book_meeting appends one dict per call to, with the keys "start_utc" (str) and
            "duration" (int, minutes). The reward is computed from this list after the episode.

    Returns:
        A list of two Strands tool objects, [get_calendar, book_meeting].

    The function itself has no side effects; calling book_meeting changes bookings. The docstrings of the tool
    functions below are the tool descriptions that Strands sends to the model, so they are written for the model.
    \"\"\"
    # @tool reads the function signature and docstring and turns them into a tool specification for the model.
    @tool
    def get_calendar(person: str) -> str:
        \"\"\"Get a person's city, UTC offset, working hours and busy blocks for the requested day.

        Args:
            person: First name.
        \"\"\"
        # The tool returns a JSON string (str) that the agent passes back to the model as the tool result.
        # Normalize the name, so "anna " and "Anna" find the same person.
        p = people.get(person.strip().title())
        if p is None:
            return "unknown person"
        off = int(p["utc_offset"] * 60)  # UTC offset in minutes, for example 330 for Mumbai
        # Busy blocks are stored in local minutes. For a calendar shared in UTC, subtract the offset in minutes.
        shift, tz = (off, "UTC") if p["calendar_in_utc"] else (0, "local")
        return json.dumps({"person": person.strip().title(), "city": p["city"], "utc_offset_hours": p["utc_offset"],
                           "working_hours_local": f"{_fmt(p['hours'][0])}-{_fmt(p['hours'][1])}",
                           "busy_times_in": tz, "busy": [f"{_fmt(a - shift)}-{_fmt(b - shift)}" for a, b in p["busy"]]})

    @tool
    def book_meeting(attendees: list[str], start_utc: str, duration_minutes: int) -> str:
        \"\"\"Book the meeting. Call this exactly once.

        Args:
            attendees: First names of all attendees.
            start_utc: Start time in UTC as HH:MM on the requested date, or 'none' if no slot exists.
            duration_minutes: Meeting length in minutes.
        \"\"\"
        # Record the booking only; the reward is computed after the episode from this list.
        bookings.append({"start_utc": str(start_utc).strip(), "duration": int(duration_minutes)})
        return "booked"

    return [get_calendar, book_meeting]


def reward_fn(task, bookings):
    \"\"\"Return the reward for one episode from the bookings the agent made.

    Args:
        task: Task dict from load, with the keys "seed", "attendees", "duration" (minutes) and "expected" (earliest
            valid start in minutes after 00:00 UTC, or None when no slot exists).
        bookings: List of booking dicts recorded by book_meeting, each with "start_utc" (str) and "duration" (int).

    Returns:
        A float: 1.0 for the earliest valid slot (or "none" when no slot exists), 0.5 for a valid but later slot,
        and 0.0 for an invalid slot, a wrong duration, a start time that cannot be parsed, or zero or several
        bookings. The function has no side effects.
    \"\"\"
    if len(bookings) != 1:
        return 0.0
    b = bookings[0]
    if b["start_utc"].lower() == "none":
        return float(task["expected"] is None)
    try:
        # Keep the last 5 characters, so "2026-11-12 14:00" and "14:00Z" both parse as 14:00.
        t = datetime.strptime(b["start_utc"].replace("Z", "").strip()[-5:], "%H:%M")
    except ValueError:
        return 0.0
    start = t.hour * 60 + t.minute  # minutes after 00:00 UTC
    if b["duration"] != task["duration"]:
        return 0.0
    people = people_for(task["seed"])
    # people_for returns the calendars generated from the task's seed (cached after the first call).
    # valid_for checks that the meeting is inside the person's working hours and overlaps none of their busy blocks.
    if not all(valid_for(people[a], start, start + task["duration"]) for a in task["attendees"]):
        return 0.0
    return 1.0 if start == task["expected"] else 0.5


def make_model(model_config):
    \"\"\"Return the Strands model object for model_config.

    Args:
        model_config: Dict. With a "bedrock_model_id" key (an Amazon Bedrock model ID), the model is that Amazon
            Bedrock model at temperature 0. Otherwise the dict must have "base_url" and "api_key", and the model is
            the OpenAI-compatible chat API at base_url, which during training is the rollout gateway.

    Returns:
        A strands BedrockModel or OpenAIModel. Creating it sends no request; the model is called when the agent
        runs.
    \"\"\"
    if "bedrock_model_id" in model_config:
        return BedrockModel(model_id=model_config["bedrock_model_id"], temperature=0.0)
    # max_retries=0: a failed model call ends the episode instead of being sent again. The gateway uses api_key to
    # tell the episodes apart, and it serves the training session's model under the name "policy".
    return OpenAIModel(client_args={"base_url": model_config["base_url"], "api_key": model_config["api_key"],
                                    "max_retries": 0, "timeout": 600},
                       model_id="policy")


def run_agent(task, model_config):
    \"\"\"Run the agent on one request and score it.

    Args:
        task: Task dict from load, with the request text ("message") and the fields that reward_fn reads.
        model_config: Dict that selects the model (see make_model). During training, run.rollout passes a
            model_config that points to the rollout gateway.

    Returns:
        A dict with "reward" (float from reward_fn), "messages" (the Strands conversation, a list of message
        dicts), "error" (the first 200 characters of the exception message if the episode ended with an
        exception, else None) and "turns" (the number of model calls).

    Side effects: sends model calls to the model selected by model_config. The tools only read the generated
    calendars and record the booking in a local list.
    \"\"\"
    bookings = []
    # NullConversationManager keeps the full conversation (no trimming of old turns). callback_handler=None turns
    # off printing of the streamed output.
    agent = Agent(model=make_model(model_config), tools=make_tools(people_for(task["seed"]), bookings),
                  system_prompt=SYSTEM_PROMPT, conversation_manager=NullConversationManager(), callback_handler=None)
    try:
        agent(task["message"])  # runs model calls and tool calls until the model answers without a tool call
        error = None
    except Exception as exc:  # turn limit, context limit: the episode ends here
        error = str(exc)[:200]
    return {"reward": reward_fn(task, bookings), "messages": agent.messages, "error": error,
            "turns": sum(m["role"] == "assistant" for m in agent.messages)}
""")

md("""
Run one request with a model on Amazon Bedrock
([`gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b)) to see the agent work. `show` prints the tool calls
and results.

**Run time:** under a minute.
""")
code("""
def show(messages, width=300):
    \"\"\"Print the tool calls (->), tool results (<-) and text of a Strands conversation.

    Args:
        messages: Strands conversation, a list of message dicts with "role" and "content" (for example
            run_agent(...)["messages"]).
        width: Maximum number of characters printed for each tool input, tool result or text block.

    Returns None. The only side effect is printing to standard output.
    \"\"\"
    for m in messages:
        for b in m.get("content") or []:
            if "toolUse" in b:
                print(f"  -> {b['toolUse']['name']}({json.dumps(b['toolUse']['input'])[:width]})")
            elif "toolResult" in b:
                print("  <-", " ".join(c.get("text", "") for c in b["toolResult"]["content"])[:width])
            elif b.get("text", "").strip():
                print(f"  {m['role']}: {b['text'][:width]}")


# Run the agent with a model on Amazon Bedrock instead of the rollout gateway.
out = run_agent(task, {"bedrock_model_id": "openai.gpt-oss-120b-1:0"})
show(out["messages"])
print("reward:", out["reward"], "| earliest slot:", _fmt(task["expected"]) if task["expected"] is not None else "none")
""")

md("""
## 3. Open a training session and measure the base model

`TrainingRun` (in `lab/runtime.py`) creates a LoRA training client on `Qwen3.8-27B` and starts the rollout gateway
on `127.0.0.1`. The gateway renders prompts with the model's chat template; `chat_template_kwargs` turns thinking
off. `run.rollout(run_agent, task)` registers a gateway session, calls `run_agent` with a `model_config` pointing
to the gateway, and returns the reward together with the token-level records of every model call.

**Run time:** from under a minute to a few minutes, until the session is running.
""")
code("""
from lab.runtime import RunConfig, TrainingRun

# Run settings: LoRA rank, learning rate and loss for the update; chat_template_kwargs turns thinking off;
# max_new_tokens caps each model call, max_context_tokens caps the conversation, max_turns caps the model calls per
# episode, rollout_timeout (seconds) caps one episode; refresh="sync" waits for the updated weights after each step.
cfg = RunConfig(region=REGION, base_model_arn=BASE_MODEL_ARN, role_arn=ROLE_ARN,
                model_package_group_arn=MPG_ARN, s3_output_path=S3_OUTPUT,
                lora_rank=32, learning_rate=4e-5, loss="ppo",
                chat_template_kwargs={"enable_thinking": False},
                max_new_tokens=1536, max_context_tokens=16384, max_turns=10, rollout_timeout=300, refresh="sync")
# open() starts the training session (returns when it is running) and the rollout gateway on 127.0.0.1.
run = TrainingRun(cfg).open()
print("training session:", run.session_name)
""")
md("""
The base model runs once on each of the 128 test requests at temperature 0.

**Run time:** about 3 to 5 minutes.
""")
code("""
async def evaluate(tasks):
    \"\"\"Run the agent once on each task at temperature 0 with the current weights.

    Args:
        tasks: List of task dicts from load.

    Returns:
        A tuple (mean reward as a float, list of rollouts). Each rollout has the task (payload), the reward and
        the result dict of run_agent (result).

    Side effects: the episodes run concurrently and sample from the training session through the rollout gateway.
    The weights are not updated.
    \"\"\"
    # run.evaluate (lab/runtime.py) calls run_agent for each task with a model_config that points to the gateway.
    rollouts = await run.evaluate(run_agent, tasks, temperature=0.0)
    return float(np.mean([r.reward for r in rollouts])), rollouts

t0 = time.time()
# Notebooks allow await at the top level of a cell.
base_score, base_rollouts = await evaluate(test_tasks)
print(f"base model: mean reward {base_score:.3f} on {len(test_tasks)} held-out requests ({time.time() - t0:.0f}s)")
""")

md("""
## 4. Train with GRPO

![One GRPO step](images/grpo.svg)

Each step takes 8 requests and runs the agent 8 times on each (temperature 1.0). Within each group of 8, the
advantage of an attempt is its reward minus the group mean, divided by the group standard deviation. Groups where
all 8 attempts scored the same are skipped. The conversation of an attempt only grows, so the gateway records all
its model calls as one token sequence. That sequence becomes one training datum with the attempt's advantage; its
loss mask is 1 on the tokens the model generated and 0 on the prompt and the tool results.

`run.train_step` (in `lab/runtime.py`) does this in one call. The core of it:

```python
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
for r, adv in zip(group, advantages):
    datums += [trace_record_to_datum(rec, float(adv)) for rec in r.records]
fb = run.tc.forward_backward(datums, loss_fn="ppo")
opt = run.tc.optim_step(AdamParams(learning_rate=cfg.learning_rate, beta2=0.95, grad_clip_norm=1.0))
await run.refresh()   # save_weights_and_get_sampling_client
```

`ppo` is the [proximal policy optimization](https://arxiv.org/abs/1707.06347) loss. After the update, `run.refresh()` snapshots the new weights, so the
next step samples from the updated model. The progress bar shows the 64 episodes of the current step.

**Run time:** about 30 to 60 minutes for 12 steps. A step takes 2 to 9 minutes, depending on how long the episodes
are and on the load of the service.
""")
code("""
GROUPS, STEPS = 8, 12
history = []
rng = random.Random(0)

for step in range(STEPS):
    # train_step runs GROUPS x 8 episodes, computes the advantages, calls forward_backward and optim_step, and
    # refreshes the sampler to the updated weights. It returns the step's metrics and the rollouts per group.
    metrics, groups = await run.train_step(run_agent, rng.sample(train_tasks, GROUPS))
    history.append(metrics["reward_mean"])
    print(f"step {step:2d}  train reward {metrics['reward_mean']:.3f}  skipped groups {metrics['groups_skipped']}/{GROUPS}  "
          f"datums {metrics['datums']:3d}  {metrics['time_step']:.0f}s")
""")
code("""
import matplotlib.pyplot as plt

plt.plot(history, marker="o")
plt.xlabel("step"); plt.ylabel("mean train reward (temperature 1.0)"); plt.grid(alpha=0.3)
plt.show()
""")

md("## 5. Evaluate and save\n\n**Run time:** about 3 to 7 minutes for the evaluation.")
code("""
t0 = time.time()
trained_score, trained_rollouts = await evaluate(test_tasks)
print(f"base {base_score:.3f} -> trained {trained_score:.3f} on {len(test_tasks)} held-out requests ({time.time() - t0:.0f}s)")
""")
md("Requests the base model got wrong and the trained model got right:")
code("""
# Pair each test request's base rollout with its trained rollout (same order) and keep the ones that improved to 1.0.
fixed = [(b, t) for b, t in zip(base_rollouts, trained_rollouts) if b.reward < 1 and t.reward == 1]
print(len(fixed), "requests fixed")
for b, t in fixed[:2]:
    p = b.payload
    print("\\nREQUEST:", p["message"], "| earliest slot:", _fmt(p["expected"]) if p["expected"] is not None else "none")
    print("base model:"); show(b.result["messages"])
    print("trained model:"); show(t.result["messages"])
""")
md("""
Save the adapter as a model package. It can be deployed to a SageMaker endpoint like any other model package; the
[training sessions cookbook](https://github.com/aws/sagemaker-training-sessions-cookbook) has a deployment notebook
that serves several adapters on one endpoint.
""")
code("""
# Save the adapter weights and register them as a model package in the model package group (MPG_ARN).
model_package_arn = run.save(inference=True)
print(model_package_arn)
""")
md("## Clean up\\n\\nStop the training session. It is billed while it is open.")
code("""
# Stop the training session and the rollout gateway. The saved model package is kept.
run.close()
""")
md("""
## Use this for your own agent

Replace three things and keep the training loop:

| Part | In this notebook | For your agent |
|---|---|---|
| Tools | `get_calendar`, `book_meeting` | Your APIs, or mocks of them that return realistic data |
| Requests | Generated, each with its correct answer | Real or generated requests, each with a way to check the outcome |
| Reward | Check the booking against every calendar | Compare the agent's final action or tool calls with the correct ones |

Measure the base model first. Training needs requests where the base model succeeds sometimes and fails
sometimes; here it scored about 0.72 before training.
""")
md("""
## From this notebook to production

- **The agent code stays the same.** `run_agent` takes a `model_config`. In this notebook it points to the rollout
  gateway or to Amazon Bedrock; in production it points to the endpoint that serves the trained adapter. The agent
  itself can run on
  [Amazon Bedrock AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html).
- **Tools become your APIs.** In production, the tools call your systems, for example as MCP (Model Context Protocol) tools
  behind [Amazon Bedrock AgentCore Gateway](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html). For training, use mocks or a sandbox copy of those systems that return realistic data,
  as this notebook does: training runs thousands of episodes, and every tool call must be safe to repeat.
- **Longer training runs** belong in a script rather than a notebook. The
  AgentCore RL Toolkit used here for the rollout gateway also
  supports rollouts from agents hosted on AgentCore Runtime.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "5c_scheduling_agent.ipynb"
nbf.write(nb, out)
print("wrote", out)

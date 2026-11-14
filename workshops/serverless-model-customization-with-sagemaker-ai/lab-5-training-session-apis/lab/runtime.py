"""Train a Strands agent with GRPO (Group Relative Policy Optimization) on a SageMaker AI training session.

This module connects three parts: an agent function that runs in this Python process, a local rollout gateway
that serves the agent's model calls, and a SageMaker AI training session that holds the model weights, samples
tokens and applies the updates.

Components
----------
Training session
    A SageMaker AI training session on the base model (for example Qwen3.8-27B) with one LoRA (low-rank
    adaptation) adapter. Only the adapter weights are trained. TrainingRun.open creates it through
    sagemaker.train.training_session.ServiceClient and keeps the returned TrainingClient in TrainingRun.tc. The
    TrainingClient runs forward_backward, optim_step, the weight saves and the creation of SamplingClient
    objects. A SamplingClient samples from one fixed snapshot of the weights.

SamplingBackend
    The object the gateway calls to generate tokens. It wraps the current SamplingClient (token ids in, token
    ids and logprobs out) and is switched to a new SamplingClient after each weight update. It also records the
    rollouts whose sampling call failed, so that train_step can leave them out.

Rollout gateway and server
    RolloutGateway (AgentCore RL Toolkit) serves an OpenAI-compatible chat completions API. For each request it
    renders the messages to token ids with the model's chat template (HfTemplateRenderer), calls the
    SamplingBackend, parses the generated tokens back into an assistant message (text and tool calls) and
    records the token ids and logprobs. ThreadedGatewayServer runs the gateway on 127.0.0.1 in a separate thread
    with its own event loop. Each rollout has its own gateway session, identified by a session id that the
    agent sends as its API key. When the rollout ends, the gateway returns trace records for the session: token
    ids, logprobs, and a loss mask that is 1 on tokens the model generated and 0 on other tokens (tool results,
    user messages, chat template tokens).

Agent function (AgentFn)
    A synchronous function agent_fn(payload, model_config) -> dict. payload is one task dict. model_config is
    the dict returned by TrainingRun.model_config: "base_url" (gateway URL), "model_id" ("policy"), "api_key"
    (the gateway session id) and "max_turns". The agent must send all its model calls to base_url with that API
    key. The returned dict must contain "reward" (a float). It may also contain "metrics" (a dict of numbers,
    averaged per step), "turns" (number of model calls), "transcript" (a readable record of the episode) and
    "error" (an error message or None). Other keys are kept in Rollout.result but not used here.

One train_step, step by step
----------------------------
1. Rollouts: for each payload in the batch, run the agent cfg.group_size times at cfg.temperature. At most
   cfg.concurrency rollouts run at once. Each rollout opens a gateway session, runs the agent function in a
   worker thread, closes the session and collects its trace records.
2. Grouping: the rollouts of one payload form a group. Rollouts whose sampling call failed are removed from
   their group.
3. Advantages: within each group, advantage = (reward - group mean) / (group standard deviation + 1e-6). A
   group with fewer than 2 rollouts, or with equal rewards, has no learning signal and is skipped.
4. Datums: each trace record of a rollout becomes one training datum (trace_record_to_datum). A datum holds the
   input tokens, the target tokens (inputs shifted by one), the sampling logprobs, a per-token weight from the
   loss mask, and the rollout's advantage on every generated token.
5. forward_backward: the datums are sent to the training session with loss cfg.loss ("ppo", the proximal policy
   optimization loss, by default). The service computes the loss and the gradients of the adapter weights.
6. optim_step: the service applies one Adam optimizer step with cfg.learning_rate, cfg.adam_beta2 and
   cfg.grad_clip_norm.
7. Refresh: the training session saves the updated weights and returns a new SamplingClient for them, and the
   SamplingBackend switches to it. Later model calls sample from the updated weights.

Steps 5 to 7 run only when the step produced at least one datum.

Sync and overlap refresh
------------------------
With cfg.refresh = "sync", train_step waits for the refresh, so every rollout of the next step samples from the
updated weights (on-policy). With cfg.refresh = "overlap", train_step starts the refresh as a background task and
returns. The next step's rollouts start with the previous weights and switch to the new ones when the snapshot is
ready, so a rollout can sample from weights that are at most one update old. The next train_step waits for the
pending refresh before it calls forward_backward. wait_refresh waits for it explicitly (for example before save).

Logging
-------
When TrainingRun is given out_dir, it appends JSONL (JSON Lines, one JSON object per line) files there:
metrics.jsonl gets one line per train_step with the step metrics, and log_rollouts appends one line per rollout
to a file name chosen by the caller. With out_dir=None nothing is written. The module logger "lab" reports at
INFO level when the training session is ready and at WARNING level when a refresh or the session stop fails.

Usage
-----
The notebook uses the module like this (inside a notebook cell, where top-level await is allowed)::

    from lab.runtime import RunConfig, TrainingRun

    cfg = RunConfig(region=REGION, base_model_arn=BASE_MODEL_ARN, role_arn=ROLE_ARN,
                    model_package_group_arn=MPG_ARN, s3_output_path=S3_OUTPUT,
                    chat_template_kwargs={"enable_thinking": False}, refresh="sync")
    run = TrainingRun(cfg).open()                     # starts the training session and the gateway
    base = await run.evaluate(run_agent, test_tasks)  # temperature 0, no update
    for step in range(STEPS):
        metrics, groups = await run.train_step(run_agent, rng.sample(train_tasks, GROUPS))
    model_package_arn = run.save(inference=True)      # registers the adapter as a model package
    run.close()                                       # stops the training session and the gateway
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm.auto import tqdm
from sagemaker.train.training_session import AdamParams, ServiceClient

from agentcore_rl_toolkit.backends.experimental.sagemaker.datum import trace_record_to_datum
from agentcore_rl_toolkit.rollout_gateway import BaseTrace, HfTemplateRenderer, response_schemas
from agentcore_rl_toolkit.rollout_gateway.gateway import RolloutGateway
from agentcore_rl_toolkit.rollout_gateway.sampling_backends.sagemaker_sdk import SageMakerSdkBackend
from agentcore_rl_toolkit.rollout_gateway.server import ThreadedGatewayServer

# The renderer picks the tool-call parser by the SHA-256 hash of the chat template. Register the hash of the current
# Qwen3.8 template (Hugging Face commit 37561816) with the Qwen3.8 schema; agentcore-rl-toolkit 0.2.0 does not list it.
response_schemas._TEMPLATE_HASHES.setdefault(
    "af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375", "qwen3_5"
)


logger = logging.getLogger("lab")

# Agent function contract: run_agent(payload, model_config) -> {"reward": float, ...}.
# payload is one task dict; model_config is the dict from TrainingRun.model_config.
# Optional keys in the returned dict: "metrics" (dict of numbers), "turns" (int), "transcript", "error".
AgentFn = Callable[[dict, dict], dict]


class SamplingBackend(SageMakerSdkBackend):
    """Gateway sampling backend that remembers which rollouts had a failed sampling call.

    SageMakerSdkBackend (AgentCore RL Toolkit) turns a gateway generate request into one
    SamplingClient.sample call on the training session and returns the generated token ids and their
    logprobs. Its set_sampling_client method replaces the SamplingClient; calls already in progress finish
    with the previous client and later calls use the new one. This subclass adds failed_sessions, the set of
    gateway session ids whose sampling call raised an error.
    """

    def __init__(self, sampling_client):
        """Create the backend.

        Args:
            sampling_client: The training session's SamplingClient for the current weights, from
                TrainingClient.create_sampling_client or TrainingClient.save_weights_and_get_sampling_client.
        """
        super().__init__(sampling_client)
        # gateway session ids (one per rollout) whose sampling call raised an error
        self.failed_sessions: set[str] = set()

    async def generate(self, **kw):
        """Sample one completion through the training session.

        Args:
            **kw: Keyword arguments from the gateway: prompt_ids (token ids of the rendered prompt),
                sampling_params (max_new_tokens, temperature and other settings) and session_id (the gateway
                session id of the rollout), passed unchanged to SageMakerSdkBackend.generate.

        Returns:
            The TurnRecord from SageMakerSdkBackend.generate: prompt ids, generated token ids, their logprobs and
            the finish reason ("stop" or "length").

        Raises:
            Exception: Any error from the sampling call is re-raised after kw["session_id"] (when present) is
                added to failed_sessions. The gateway then returns an error response to the agent.
        """
        try:
            return await super().generate(**kw)
        except Exception:
            # TrainingRun.rollout checks this set and marks the rollout as failed, so it is left out of training
            if kw.get("session_id"):
                self.failed_sessions.add(kw["session_id"])
            raise


@dataclass
class RunConfig:
    """Settings for one training run. Fields without a default are required.

    Training session:
        region: AWS Region of the training session, for example "us-west-2".
        base_model_arn: SageMaker Public Hub ARN (or model id) of the base model of the training session.
        role_arn: ARN of the IAM role that the training session assumes.
        model_package_group_arn: ARN of the model package group where saved adapters are registered as model
            package versions.
        s3_output_path: S3 URI where the service writes adapter files.
        lora_rank: Rank of the LoRA adapter matrices. Default 32.
        lora_alpha: LoRA scaling factor alpha. Default 64.

    Optimization:
        learning_rate: Adam learning rate (step size). Default 4e-5.
        loss: Loss function name passed to forward_backward. Default "ppo" (proximal policy optimization, a
            clipped surrogate loss that uses the sampling logprobs and advantages). The service also accepts
            "cross_entropy", "importance_sampling" and "cispo".
        groups_per_step: Number of payloads (problems) per step. Default 8. train_step itself uses len(batch);
            the caller uses this field to choose the batch size.
        group_size: Rollouts per payload in each step. Default 8.
        adam_beta2: Adam decay rate for the second moment estimates. Default 0.95.
        grad_clip_norm: Maximum gradient norm; gradients with a larger norm are scaled down. Default 1.0.
            None sends no clipping value to the service.

    Sampling and rollouts:
        temperature: Sampling temperature for training rollouts. Default 1.0. evaluate uses its own
            temperature argument.
        max_new_tokens: Maximum number of tokens generated per model call. Default 2048.
        max_context_tokens: Maximum number of tokens (prompt plus generation) of one rollout's conversation.
            Default 16384. The gateway lowers max_new_tokens to fit, and returns a context length error to the
            agent when the remaining room is below the backend minimum (512 tokens).
        max_turns: Maximum number of model calls per rollout. Default 12. The gateway answers further calls
            with HTTP status 429; the value is also passed to the agent in model_config.
        rollout_timeout: Maximum wall time of one rollout, in seconds. Default 600.
        concurrency: Maximum number of rollouts running at once. Default 64.
        chat_template_kwargs: Extra keyword arguments for the chat template, for example
            {"enable_thinking": False}. Default empty.
        progress: Show a progress bar for the rollouts of each step and each evaluation. Default True.
        history_mode: How the gateway tracks the conversation: "linear" (default) or "tree". In "linear" mode
            the gateway reuses the exact token ids it served for earlier turns when it renders the next prompt,
            so an append-only conversation yields one trace record per rollout.

    Training session calls:
        refresh: "sync" (default) or "overlap"; see the module docstring.
        train_op_timeout: Maximum time to wait for forward_backward, optim_step and save operations, in
            seconds. Default 900.
    """

    region: str
    base_model_arn: str
    role_arn: str
    model_package_group_arn: str
    s3_output_path: str
    lora_rank: int = 32
    lora_alpha: int = 64
    learning_rate: float = 4e-5
    loss: str = "ppo"
    groups_per_step: int = 8
    group_size: int = 8
    temperature: float = 1.0
    max_new_tokens: int = 2048
    max_context_tokens: int = 16384
    max_turns: int = 12
    rollout_timeout: float = 600.0
    concurrency: int = 64
    chat_template_kwargs: dict = field(default_factory=dict)
    progress: bool = True  # show a progress bar for the rollouts of each step and each evaluation
    history_mode: str = "linear"
    adam_beta2: float = 0.95
    grad_clip_norm: float | None = 1.0
    # "sync": wait for the new sampler after each update; "overlap": keep sampling with
    # the previous weights while the snapshot is created (rollouts are then at most one step stale)
    refresh: str = "sync"
    train_op_timeout: float = 900.0


@dataclass
class Rollout:
    """Result of one agent run.

    Fields:
        payload: The task dict passed to the agent function.
        reward: result["reward"] as a float, or 0.0 when the agent function raised an error, timed out or
            returned something other than a dict.
        result: The dict the agent function returned, or {} when it raised an error or timed out.
        records: Gateway trace records of the rollout that contain tokens. Each record has token_ids (the full
            token sequence), loss_mask (1 for generated tokens, 0 for other tokens, covering the response part of
            the sequence) and logprobs (sampling logprobs aligned with loss_mask). In "linear" history mode an
            append-only conversation gives one record that holds all model calls. Empty when failed is True.
        seconds: Wall time of the rollout in seconds, from the start of the agent function to the end of the
            gateway session.
        error: "timeout", "<ExceptionType>: <message>" (at most 300 characters), "sampling failed", or None.
        failed: True when a sampling call on the training session failed. Such rollouts are not trained on.
    """

    payload: dict
    reward: float
    result: dict
    records: list
    seconds: float
    error: str | None = None
    # True when a sampling call failed on the service side; such rollouts are not trained on
    failed: bool = False


class TrainingRun:
    """One training session plus the local rollout gateway the agent talks to.

    Attributes:
        cfg: The RunConfig.
        out_dir: Directory for JSONL logs, or None.
        tc: The training session's TrainingClient, set by open and cleared by close.
        gateway: The RolloutGateway, set by open.
        server: The ThreadedGatewayServer that serves the gateway on 127.0.0.1, set by open.
        backend: The SamplingBackend used by the gateway, set by open.
        step: Number of completed train_step calls.
        refresh_seconds: Duration in seconds of each successful refresh.
    """

    def __init__(self, cfg: RunConfig, out_dir: str | Path | None = None):
        """Store the settings. No training session or gateway is created until open().

        Args:
            cfg: Settings for the run.
            out_dir: Directory for the JSONL logs (metrics.jsonl and the files written by log_rollouts). The
                directory is created on the first write. When None, nothing is written.
        """
        self.cfg = cfg
        self.out_dir = Path(out_dir) if out_dir else None
        self.tc = None
        self.gateway = None
        self.server = None
        self.backend = None
        self.step = 0
        # pending background refresh in "overlap" mode
        self._refresh_task = None
        self.refresh_seconds: list[float] = []
        # limits the number of rollouts running at once to cfg.concurrency
        self._sem = asyncio.Semaphore(cfg.concurrency)

    # -- lifecycle -----------------------------------------------------------

    def open(self, resume_from_model_package: str | None = None, init_from_model_package: str | None = None) -> "TrainingRun":
        """Start the training session and the local rollout gateway.

        Args:
            resume_from_model_package: ARN of a model package saved with save(inference=False). The new
                session loads its adapter weights and optimizer state
                (ServiceClient.create_training_client_from_state_with_optimizer).
            init_from_model_package: ARN of a model package with adapter weights. The new session loads the
                weights only and starts with a new optimizer state
                (ServiceClient.create_training_client_from_state). Takes precedence over
                resume_from_model_package.

        With neither argument, a new LoRA adapter is created on cfg.base_model_arn
        (ServiceClient.create_lora_training_client).

        Returns:
            self, so that TrainingRun(cfg).open() can be assigned directly.

        Side effects: creates a training session (it counts against the concurrent-session quota until close) and blocks until it is running; loads
        the tokenizer; creates a SamplingClient for the current weights; starts the gateway server thread on a
        free port of 127.0.0.1. Errors from the training session service and from the gateway start
        (TimeoutError, RuntimeError) are raised unchanged.
        """
        cfg = self.cfg
        # ServiceClient holds the account-level settings; the create_* calls below create the training session
        svc = ServiceClient(
            region=cfg.region,
            role_arn=cfg.role_arn,
            model_package_group_arn=cfg.model_package_group_arn,
            s3_output_path=cfg.s3_output_path,
        )
        t0 = time.monotonic()
        if init_from_model_package:
            # weights only (e.g. an adapter registered with scripts/register_adapter.py); the base model is read
            # from the model package metadata
            self.tc = svc.create_training_client_from_state(
                model_package_arn=init_from_model_package, rank=cfg.lora_rank, alpha=cfg.lora_alpha
            )
        elif resume_from_model_package:
            # weights and Adam optimizer state
            self.tc = svc.create_training_client_from_state_with_optimizer(
                model_package_arn=resume_from_model_package,
                base_model=cfg.base_model_arn,
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
            )
        else:
            # new adapter on the base model
            self.tc = svc.create_lora_training_client(
                base_model=cfg.base_model_arn, rank=cfg.lora_rank, alpha=cfg.lora_alpha
            )
        logger.info("training session %s ready in %.0fs", self.tc.training_session_name, time.monotonic() - t0)
        # Hugging Face tokenizer of the base model; the renderer uses its chat template
        tokenizer = self.tc.get_tokenizer()
        # sampler for the current weights; train steps replace it after each update (see refresh)
        self.backend = SamplingBackend(self.tc.create_sampling_client())
        renderer = HfTemplateRenderer(tokenizer, chat_template_kwargs=cfg.chat_template_kwargs)
        # The gateway serves an OpenAI-compatible API, renders messages to tokens and records each model call.
        # max_turns_per_sid caps the model calls per gateway session (one session per rollout).
        self.gateway = RolloutGateway(
            backend=self.backend,
            renderer=renderer,
            tokenizer=tokenizer,
            adapters=["openai"],
            history_mode=cfg.history_mode,
            max_turns_per_sid=cfg.max_turns,
        )
        # Loopback only: the agent runs in this process. port=0 picks a free port; server.base_url has the result.
        # start() runs the server in a background thread with its own event loop and returns when it serves.
        self.server = ThreadedGatewayServer(self.gateway, host="127.0.0.1", port=0)
        self.server.start()
        return self

    def close(self) -> None:
        """Stop the training session and shut down the gateway server.

        Calls TrainingClient.stop, which sends the operations already queued and then stops the session. A
        failed stop is logged as a warning and not raised. Safe to call more than once.
        """
        if self.tc is not None:
            try:
                self.tc.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("stopping training session failed: %s", exc)
            self.tc = None
        if self.server is not None:
            self.server.shutdown()
            self.server = None

    def __enter__(self):
        """Open the run with default arguments (see open) and return it."""
        return self.open()

    def __exit__(self, *exc):
        """Close the run (see close). Exceptions from the with block are not suppressed."""
        self.close()

    @property
    def session_name(self) -> str:
        """Name of the training session. Available after open()."""
        return self.tc.training_session_name

    # -- rollouts ------------------------------------------------------------

    def model_config(self, sid: str) -> dict:
        """Return the model settings the agent uses to call the gateway for one rollout.

        Args:
            sid: Gateway session id of the rollout.

        Returns:
            A dict with "base_url" (the gateway URL, ending in /v1), "model_id" ("policy"), "api_key" (sid) and
            "max_turns" (cfg.max_turns). The gateway reads the session id from the Authorization: Bearer header
            that OpenAI clients build from the API key, so it can attribute each model call to its rollout.
        """
        return {"base_url": self.server.base_url, "model_id": "policy", "api_key": sid, "max_turns": self.cfg.max_turns}

    async def rollout(self, agent_fn: AgentFn, payload: dict, temperature: float | None = None) -> Rollout:
        """Run agent_fn once on payload through the gateway and return the Rollout.

        Args:
            agent_fn: The agent function (see AgentFn).
            payload: The task dict passed to agent_fn.
            temperature: Sampling temperature; None uses cfg.temperature.

        Returns:
            A Rollout. Timeouts and agent errors are not raised: they are recorded in Rollout.error and give
            reward 0.0. A rollout with a failed sampling call is returned with failed=True and no records.

        Gateway calls: create_session registers the session id with the sampling settings and the context
        limit; finish_session waits for model calls still in progress, turns the session's conversation into
        trace records and removes the session. The agent's model calls between the two sample from the
        training session through the SamplingBackend.
        """
        cfg = self.cfg
        # new random session id per rollout; the agent sends it as its API key
        sid = uuid.uuid4().hex
        sampling = {"max_new_tokens": cfg.max_new_tokens, "temperature": cfg.temperature if temperature is None else temperature}
        async with self._sem:
            # one gateway session per rollout, so its model calls are recorded together; sampling_defaults apply
            # to every model call of the session
            self.gateway.create_session(sid, sampling_defaults=sampling, max_context_tokens=cfg.max_context_tokens)
            t0 = time.monotonic()
            result: dict = {}
            error = None
            try:
                # The agent function is synchronous, so it runs in a worker thread. On timeout, wait_for stops
                # waiting but the worker thread is not interrupted.
                result = await asyncio.wait_for(
                    asyncio.to_thread(agent_fn, payload, self.model_config(sid)), cfg.rollout_timeout
                )
            except asyncio.TimeoutError:
                error = "timeout"
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"[:300]
            reward = float(result.get("reward", 0.0)) if isinstance(result, dict) else 0.0
            # Finish the session on the gateway's own event loop (the server thread). finish_session waits up to
            # 5 seconds for calls still in progress, then returns the trace records with the reward attached.
            fut = asyncio.run_coroutine_threadsafe(
                self.gateway.finish_session(sid, base_sample=BaseTrace(rollout_id=sid), reward=reward), self.server._loop
            )
            # trace records: token ids, logprobs and loss mask (see Rollout.records)
            records = await asyncio.wrap_future(fut)
            # SamplingBackend.generate added sid here if any sampling call of this rollout raised an error
            if sid in self.backend.failed_sessions:
                self.backend.failed_sessions.discard(sid)
                error = error or "sampling failed"
                return Rollout(payload, reward, result, [], time.monotonic() - t0, error, failed=True)
            # records without tokens (no model output) are dropped
            return Rollout(payload, reward, result, [r for r in records if r.token_ids], time.monotonic() - t0, error)

    async def _gather(self, coros: list, desc: str) -> list:
        """Run coroutines concurrently and return their results in the input order.

        Args:
            coros: Coroutines to await (here, calls to rollout).
            desc: Label of the progress bar, shown when cfg.progress is True.

        Returns:
            A list with the result of each coroutine, in the order of coros. An exception from one coroutine is
            raised (asyncio.gather).
        """
        bar = tqdm(total=len(coros), desc=desc, leave=False) if self.cfg.progress else None

        async def tick(c):
            """Await coroutine c, advance the progress bar by one, and return the result of c."""
            r = await c
            if bar is not None:
                bar.update(1)
            return r

        try:
            return list(await asyncio.gather(*[tick(c) for c in coros]))
        finally:
            if bar is not None:
                bar.close()

    async def evaluate(self, agent_fn: AgentFn, payloads: list[dict], temperature: float = 0.0) -> list[Rollout]:
        """Run agent_fn once per payload with the current weights. No training.

        Args:
            agent_fn: The agent function (see AgentFn).
            payloads: Task dicts, one rollout each.
            temperature: Sampling temperature. Default 0.0 (greedy sampling).

        Returns:
            One Rollout per payload, in the order of payloads. The rollouts run concurrently (at most
            cfg.concurrency at once) and sample from the training session's current SamplingClient. Use
            summarize_eval for aggregate numbers.
        """
        return await self._gather([self.rollout(agent_fn, p, temperature) for p in payloads], "evaluation")

    # -- training ------------------------------------------------------------

    async def train_step(self, agent_fn: AgentFn, batch: list[dict]) -> tuple[dict, list[list[Rollout]]]:
        """Run one GRPO step on batch.

        Args:
            agent_fn: The agent function (see AgentFn).
            batch: Task dicts. Each one gets cfg.group_size rollouts at cfg.temperature.

        Returns:
            A tuple (metrics, groups). groups has one list of cfg.group_size Rollouts per payload, in the order
            of batch (failed rollouts included). metrics is a dict with:
                step: Index of this step (self.step before the increment).
                reward_mean, reward_std: Mean and standard deviation of the reward over all rollouts.
                groups_skipped: Groups with fewer than 2 usable rollouts or equal rewards.
                groups, rollouts: Number of groups and of rollouts.
                errors: Rollouts with an error (timeout, agent error or failed sampling).
                failed: Rollouts with a failed sampling call.
                datums: Number of datums sent to forward_backward.
                turns_mean: Mean of result["turns"] (0 when absent).
                rollout_s_mean: Mean rollout wall time in seconds.
                gen_tokens_mean: Mean number of generated tokens per rollout (sum of the loss masks).
                m/<key>: Mean of each key of result["metrics"] over the rollouts that report it.
                train/<key>: Numeric metrics returned by forward_backward (only when datums were sent).
                time_rollout, time_train, time_refresh, time_step: Durations in seconds of the rollout phase,
                    the forward_backward and optim_step phase (including the wait for a pending overlap
                    refresh), the refresh phase (near 0 in "overlap" mode), and the whole step.

        Training session calls: forward_backward(datums, loss_fn=cfg.loss) and optim_step(AdamParams(...)),
        then save_weights_and_get_sampling_client through refresh. These run only when there is at least one
        datum; otherwise the weights and the sampler stay unchanged.

        Side effects: appends metrics to out_dir/metrics.jsonl and increments self.step. In "overlap" mode the
        refresh is left running as a background task (see wait_refresh).

        Raises:
            Exception: Errors of forward_backward or optim_step, including TimeoutError after
                cfg.train_op_timeout seconds, are raised. self.step is then not incremented.
        """
        cfg = self.cfg
        t0 = time.monotonic()
        # all rollouts of the step run concurrently; payload order is kept, each payload repeated group_size times
        flat_rollouts = await self._gather(
            [self.rollout(agent_fn, p) for p in batch for _ in range(cfg.group_size)], f"step {self.step} rollouts")
        # split the flat list back into one group of group_size rollouts per payload
        groups = [flat_rollouts[i * cfg.group_size:(i + 1) * cfg.group_size] for i in range(len(batch))]
        t_rollout = time.monotonic() - t0

        datums, skipped = [], 0
        for group in groups:
            # rollouts with a failed sampling call are left out
            group = [r for r in group if not r.failed]
            rewards = np.array([r.reward for r in group], dtype=np.float64)
            # a group with fewer than 2 rollouts or equal rewards gives no learning signal
            if len(group) < 2 or rewards.std() < 1e-6:
                skipped += 1
                continue
            # GRPO advantage: reward relative to the group mean, in units of the group standard deviation
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            for r, a in zip(group, adv):
                # One datum per trace record (see Rollout.records). trace_record_to_datum builds input tokens and
                # target tokens (shifted by one), the sampling logprobs, weights from the loss mask (0 on the
                # prompt and on tokens the model did not generate, such as tool results) and the rollout's
                # advantage on generated tokens. It returns None when the record has no generated tokens.
                for rec in r.records:
                    d = trace_record_to_datum(rec, float(a))
                    if d is not None:
                        datums.append(d)

        # metrics are computed over all rollouts, including failed ones and those of skipped groups
        flat = [r for g in groups for r in g]
        m = {
            "step": self.step,
            "reward_mean": float(np.mean([r.reward for r in flat])),
            "reward_std": float(np.std([r.reward for r in flat])),
            "groups_skipped": skipped,
            "groups": len(groups),
            "rollouts": len(flat),
            "errors": sum(r.error is not None for r in flat),
            "failed": sum(r.failed for r in flat),
            "datums": len(datums),
            "turns_mean": float(np.mean([r.result.get("turns", 0) for r in flat if isinstance(r.result, dict)] or [0])),
            "rollout_s_mean": float(np.mean([r.seconds for r in flat])),
            # the loss mask is 1 on each generated token, so its sum counts the generated tokens
            "gen_tokens_mean": float(np.mean([sum(sum(rec.loss_mask) for rec in r.records) for r in flat])),
            "time_rollout": t_rollout,
        }
        # mean of each use-case metric reported in result["metrics"]
        for key in sorted({k for r in flat if isinstance(r.result, dict) for k in r.result.get("metrics", {})}):
            vals = [r.result["metrics"][key] for r in flat if isinstance(r.result, dict) and key in r.result.get("metrics", {})]
            m[f"m/{key}"] = float(np.mean(vals))

        t1 = time.monotonic()
        if self._refresh_task is not None:
            # overlap mode: the previous refresh must finish before the next one starts
            await self._refresh_task
            self._refresh_task = None
        if datums:
            # Both calls return futures (APIFuture) at once; the requests are sent from a background thread.
            # optim_step is submitted right away, and sequence numbers assigned at call time keep it behind
            # forward_backward on the service. forward_backward computes the loss and accumulates the
            # gradients of the adapter weights; optim_step applies one Adam update with those gradients.
            # The waits (future.result) block, so they run in threads to keep the event loop free.
            fb = self.tc.forward_backward(datums, loss_fn=cfg.loss)
            op = self.tc.optim_step(AdamParams(learning_rate=cfg.learning_rate, beta2=cfg.adam_beta2,
                                               grad_clip_norm=cfg.grad_clip_norm))
            fb_res = await asyncio.to_thread(fb.result, timeout=cfg.train_op_timeout)
            await asyncio.to_thread(op.result, timeout=cfg.train_op_timeout)
            # keep only numeric loss metrics reported by the service
            m.update({f"train/{k}": v for k, v in (fb_res.metrics or {}).items() if isinstance(v, (int, float))})
        m["time_train"] = time.monotonic() - t1

        t2 = time.monotonic()
        # without an update the weights did not change, so the sampler is kept
        if datums:
            if cfg.refresh == "overlap":
                # next step starts sampling with the current weights and switches when the snapshot is ready
                self._refresh_task = asyncio.create_task(self._refresh())
            else:
                # sync mode: the next step samples only from the updated weights
                await self._refresh()
        m["time_refresh"] = time.monotonic() - t2
        m["time_step"] = time.monotonic() - t0
        self.step += 1
        self._log("metrics.jsonl", m)
        return m, groups

    async def refresh(self, attempts: int = 3) -> None:
        """Snapshot the current weights and switch the sampler to them.

        Args:
            attempts: Number of tries. After failed try i (counting from 1) the method waits 10 * i seconds.

        Training session call: TrainingClient.save_weights_and_get_sampling_client, which saves the adapter
        weights (registering a model package version in cfg.model_package_group_arn), loads them for sampling
        and returns a new SamplingClient. The SamplingBackend then switches to that client.

        Side effects: appends the duration to self.refresh_seconds on success. Errors are logged as warnings
        and not raised; if every attempt fails, sampling continues with the previous weights.
        """
        t = time.monotonic()
        for i in range(attempts):
            try:
                # save_weights_and_get_sampling_client blocks until the snapshot is ready, so it runs in a thread
                sc = await asyncio.to_thread(self.tc.save_weights_and_get_sampling_client)
                # Later sampling calls from the gateway use the new sampler. Calls already in progress finish
                # with the previous one.
                self.backend.set_sampling_client(sc)
                self.refresh_seconds.append(time.monotonic() - t)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("refresh attempt %d failed: %s", i + 1, str(exc)[:200])
                # wait 10 s, 20 s, 30 s, ... before the next attempt
                await asyncio.sleep(10 * (i + 1))
        logger.warning("refresh failed %d times; keeping the previous sampler", attempts)

    # train_step calls the refresh through this alias
    _refresh = refresh

    async def wait_refresh(self) -> None:
        """Wait for a pending background refresh ("overlap" mode). Returns at once if there is none.

        Call it before evaluate or save when the evaluation or the saved adapter must use the sampler for the
        latest weights.
        """
        if self._refresh_task is not None:
            await self._refresh_task
            self._refresh_task = None

    def save(self, inference: bool = True) -> str:
        """Save the adapter as a model package version in cfg.model_package_group_arn and return its ARN.

        Args:
            inference: True saves the adapter weights for deployment (TrainingClient.save_weights_for_sampler,
                with description "step <self.step>"). False saves the weights and the optimizer state
                (TrainingClient.save_state), which open(resume_from_model_package=...) can resume from.

        Returns:
            The model package ARN.

        Blocks until the save completes. Raises the error of the save operation, including TimeoutError after
        cfg.train_op_timeout seconds.
        """
        if inference:
            res = self.tc.save_weights_for_sampler(description=f"step {self.step}").result(timeout=self.cfg.train_op_timeout)
        else:
            res = self.tc.save_state().result(timeout=self.cfg.train_op_timeout)
        return res.model_package_arn

    # -- logging -------------------------------------------------------------

    def _log(self, name: str, row: dict) -> None:
        """Append row as one JSON line to out_dir/name.

        Args:
            name: File name inside out_dir.
            row: Dict to write. Values that JSON cannot encode are written with str().

        Creates out_dir if needed. Does nothing when out_dir is None.
        """
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / name, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def log_rollouts(self, name: str, rollouts: list[Rollout], **extra) -> None:
        """Append one JSON line per rollout to out_dir/name.

        Args:
            name: File name inside out_dir, for example "rollouts.jsonl".
            rollouts: Rollouts to write.
            **extra: Fields added to every line, for example step=3.

        Each line has the extra fields and "id" (payload["id"]), "reward", "error" (Rollout.error, else
        result["error"]), "seconds" (rounded to 0.1), "turns", "metrics" and "transcript" (from the result dict,
        None when absent). Does nothing when out_dir is None.
        """
        for r in rollouts:
            res = r.result if isinstance(r.result, dict) else {}
            self._log(name, {**extra, "id": r.payload.get("id"), "reward": r.reward, "error": r.error or res.get("error"),
                             "seconds": round(r.seconds, 1), "turns": res.get("turns"),
                             "metrics": res.get("metrics"), "transcript": res.get("transcript")})


def summarize_eval(rollouts: list[Rollout]) -> dict:
    """Summarize evaluation rollouts.

    Args:
        rollouts: Rollouts, for example from TrainingRun.evaluate. With an empty list the means are NaN.

    Returns:
        A dict with "n" (number of rollouts), "reward_mean", "errors" (rollouts with an error), "seconds_mean"
        (mean wall time in seconds), "gen_tokens_mean" (mean number of generated tokens per rollout) and, for each
        key of result["metrics"], the mean over the rollouts that report it (under the key name itself).
    """
    out = {"n": len(rollouts), "reward_mean": float(np.mean([r.reward for r in rollouts])),
           "errors": sum(r.error is not None for r in rollouts),
           "seconds_mean": float(np.mean([r.seconds for r in rollouts])),
           "gen_tokens_mean": float(np.mean([sum(sum(rec.loss_mask) for rec in r.records) for r in rollouts]))}
    keys = {k for r in rollouts if isinstance(r.result, dict) for k in r.result.get("metrics", {})}
    for k in sorted(keys):
        vals = [r.result["metrics"][k] for r in rollouts if isinstance(r.result, dict) and k in r.result.get("metrics", {})]
        out[k] = float(np.mean(vals))
    return out


def config_dict(cfg: RunConfig) -> dict:
    """Return cfg as a plain dict (dataclasses.asdict), for example to write it to a JSON file."""
    return asdict(cfg)

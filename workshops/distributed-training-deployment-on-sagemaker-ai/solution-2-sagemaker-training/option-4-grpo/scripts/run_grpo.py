"""veRL GRPO override construction and trainer invocation.

This module runs **only inside the GPU container**. Per Requirement 1.7 it
imports nothing from ``src/grpo_sagemaker/``: the hyperparameters arrive as a
plain ``dict[str, str]`` that ``entrypoint.py`` reads from
``/opt/ml/input/config/hyperparameters.json``, which is where SageMaker writes
the mapping produced by ``config.resolved_hyperparameters``. SageMaker renders
every hyperparameter value as a string, so every value is parsed and validated
here rather than trusted.

:func:`build_verl_argv` is a pure function of its arguments. It reads no file,
touches no network, and consults the environment only through the injectable
``env`` parameter, so the override list can be asserted in full on a workstation
with no veRL, Ray, or GPU present (Property 24).

## Where each override comes from

Requirement 13.7 requires every veRL override to derive from the validated run
configuration. Overrides therefore fall into exactly three declared groups, and
each group is a module-level table so the provenance is inspectable rather than
buried in procedural code:

* :data:`HP_OVERRIDES` -- derived from a named hyperparameter key. This is the
  overwhelming majority, and the table records which key each one reads.
* :data:`CHANNEL_OVERRIDES` -- derived from the resolved SageMaker channel
  paths, which the launcher wrote to S3 and SageMaker mounted locally.
* :data:`CONTRACT_OVERRIDES` -- fixed by the SageMaker ``/opt/ml`` contract or
  by the container environment, not by anything tunable. There are only two, and
  each is justified at its definition.

Nothing else is emitted. Knobs that veRL exposes but this project's
configuration does not model -- batch sizes, tensor parallel degree, offload
flags -- deliberately keep veRL's own defaults rather than being invented here,
because a value with no configuration field behind it cannot satisfy
Requirement 13.7 and would drift silently from the recipe.

## Reward scoring: no override is required

Per Requirements 9.1 and 9.2 this project writes no reward function, and it also
passes **no reward-selection flag**, because veRL needs none. The mechanism, as
of v0.8.0, is data-driven:

1. ``reward.custom_reward_function.path`` defaults to ``null``, so no custom
   scorer is loaded.
2. The default ``naive`` reward manager therefore falls back to
   ``verl.utils.reward_score.default_compute_score``.
3. That function switches on the row's ``data_source`` field -- located via
   ``data.reward_fn_key``, whose default is ``"data_source"`` -- and dispatches
   ``"openai/gsm8k"`` to ``verl/utils/reward_score/gsm8k.py``.

``prepare_data.py`` writes ``data_source="openai/gsm8k"`` into every row, which
is what selects the built-in scorer. veRL's own reference recipe
(``examples/grpo_trainer/run_qwen3_4b_fsdp.sh``) likewise passes no reward
override. Emitting ``custom_reward_function.path`` here would be wrong: it is
the hook for *replacing* the built-in scorer, which
``docs/adapting-customer-data.md`` documents as an appendix (Requirement 9.3).
"""

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

VERL_TRAINER_MODULE = "verl.trainer.main_ppo"
"""The veRL entry point this project invokes (Requirement 13.6).

Run as ``python -m verl.trainer.main_ppo``, which is how veRL's own example
recipes launch it and what its Hydra ``@hydra.main`` decorator expects.

``main_ppo_sync.py`` is the newer synchronous trainer introduced alongside
TransferQueue. It is recorded in ``docs/container.md`` as a future option and is
not adopted here.
"""

CHECKPOINT_DIR = "/opt/ml/checkpoints"
"""Where veRL writes checkpoints (Requirement 7.3).

SageMaker continuously syncs this directory to the checkpoint S3 URI that
``launch_training.py`` sets on the job.
"""

TRAIN_CHANNEL = "train"
VALIDATION_CHANNEL = "validation"

_SUPPORTED_DATA_SUFFIXES = (".parquet", ".json", ".jsonl")
"""Extensions veRL's ``RLHFDataset`` accepts.

``verl/utils/dataset/rl_dataset.py`` dispatches on the file suffix and raises
``ValueError: Unsupported file format`` for anything else. It does **not** expand
a directory, so a channel path must name the file rather than the mount point.
"""

_TRUE_LITERALS = frozenset({"true", "1", "yes"})
_FALSE_LITERALS = frozenset({"false", "0", "no"})


class GrpoRunError(RuntimeError):
    """Base class for every failure raised by this module."""


class MissingHyperparameterError(GrpoRunError):
    """A hyperparameter the override list needs was absent."""


class InvalidHyperparameterError(GrpoRunError):
    """A hyperparameter was present but could not be parsed or was out of range."""


class MissingChannelError(GrpoRunError):
    """A data channel the override list needs was absent or unusable."""


# --------------------------------------------------------------------------- #
# Value rendering
#
# SageMaker hands every hyperparameter over as a string. Each renderer parses
# the string, range-checks it, and returns the token Hydra should receive, so a
# malformed hyperparameters.json fails here with the offending key named rather
# than inside a Ray actor twenty minutes into a paid job.
# --------------------------------------------------------------------------- #


def render_str(key: str, raw: str) -> str:
    """Return a non-empty string value unchanged."""
    value = raw.strip()
    if not value:
        raise InvalidHyperparameterError(f"hyperparameter {key!r} must not be empty")
    return value


def render_int(key: str, raw: str, *, minimum: int | None = None) -> str:
    """Parse an integer and range-check it, returning its canonical rendering."""
    try:
        value = int(raw.strip())
    except (TypeError, ValueError) as exc:
        raise InvalidHyperparameterError(
            f"hyperparameter {key!r} must be an integer; got {raw!r}"
        ) from exc
    if minimum is not None and value < minimum:
        raise InvalidHyperparameterError(
            f"hyperparameter {key!r} must be at least {minimum}; got {value}"
        )
    return str(value)


def render_float(key: str, raw: str) -> str:
    """Parse a float, rejecting NaN and infinity, and return its rendering.

    Hydra would accept ``nan`` or ``inf`` and pass it into the optimiser, where
    it becomes a silent training failure rather than a loud configuration one.
    """
    try:
        value = float(raw.strip())
    except (TypeError, ValueError) as exc:
        raise InvalidHyperparameterError(
            f"hyperparameter {key!r} must be a number; got {raw!r}"
        ) from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise InvalidHyperparameterError(f"hyperparameter {key!r} must be finite; got {raw!r}")
    return repr(value)


def render_bool(key: str, raw: str) -> str:
    """Normalise a boolean hyperparameter to the ``True``/``False`` Hydra parses.

    ``config.resolved_hyperparameters`` already emits exactly these two spellings,
    but ``hyperparameters.json`` is a plain JSON file that an operator can edit,
    so the accepted spellings are widened and everything else is refused rather
    than being coerced to a surprising truthiness.
    """
    value = raw.strip().lower()
    if value in _TRUE_LITERALS:
        return "True"
    if value in _FALSE_LITERALS:
        return "False"
    raise InvalidHyperparameterError(
        f"hyperparameter {key!r} must be a boolean; got {raw!r} "
        f"(accepted: {sorted(_TRUE_LITERALS | _FALSE_LITERALS)})"
    )


# --------------------------------------------------------------------------- #
# Override provenance tables
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OverrideSpec:
    """One veRL override and the hyperparameter it reads.

    Making provenance data rather than control flow is what lets Property 24 be
    checked exhaustively: a test can walk this table, and for every entry assert
    the emitted value for ``verl_key`` was derived from ``hp[hp_key]``. A new
    override cannot be added without declaring where its value comes from.
    """

    verl_key: str
    """The veRL/Hydra dotted key, as it appears left of the ``=``."""

    hp_key: str
    """The hyperparameter key whose value this override carries."""

    kind: str
    """One of ``str``, ``int``, ``float``, ``bool``; selects the renderer."""

    minimum: int | None = None
    """Inclusive lower bound, applied to ``int`` values only."""

    required: bool = True
    """When false, the override is omitted if the hyperparameter is absent.

    ``config.resolved_hyperparameters`` drops keys whose configured value is
    ``None``, so absence is meaningful: it means "leave veRL's default alone"
    rather than "pass the string ``None``".
    """

    mandated_value: str | None = None
    """A value the requirements fix, verified rather than substituted.

    Set only where the configuration model pins the field to a single literal
    (``adv_estimator``, ``rollout_name``). Because the configuration cannot
    legally hold anything else, checking it can never contradict Property 24's
    "equals the corresponding configuration field" -- it only catches a
    hyperparameters.json that was corrupted or hand-edited in transit. Fields
    the operator may legitimately vary carry no mandated value and are passed
    through as configured.
    """


HP_OVERRIDES: tuple[OverrideSpec, ...] = (
    # --- The GRPO recipe itself (Requirements 13.1 - 13.5) ---
    # Selects group-relative advantage instead of the GAE default.
    OverrideSpec("algorithm.adv_estimator", "adv_estimator", "str", mandated_value="grpo"),
    # veRL's rollout backend is a mandatory field with no default (`???`).
    OverrideSpec("actor_rollout_ref.rollout.name", "rollout_name", "str", mandated_value="vllm"),
    # The rollout group size. GRPO's advantage is relative within a group, so a
    # group of one carries no signal; config enforces >= 2 and so does this.
    OverrideSpec("actor_rollout_ref.rollout.n", "rollout_n", "int", minimum=2),
    # veRL defaults use_kl_loss to false, so this must be passed explicitly for
    # the KL term to apply at all.
    OverrideSpec("actor_rollout_ref.actor.use_kl_loss", "use_kl_loss", "bool"),
    OverrideSpec("actor_rollout_ref.actor.kl_loss_coef", "kl_loss_coef", "float"),
    # Node count, which must agree with the Ray cluster start_ray.py formed.
    OverrideSpec("trainer.nnodes", "instance_count", "int", minimum=1),
    # --- Actor model and sequence budget ---
    OverrideSpec("actor_rollout_ref.model.path", "actor_path", "str"),
    OverrideSpec("data.max_prompt_length", "max_prompt_length", "int", minimum=1),
    OverrideSpec("data.max_response_length", "max_response_length", "int", minimum=1),
    # --- Loss aggregation and rollout memory ---
    OverrideSpec("actor_rollout_ref.actor.loss_agg_mode", "loss_agg_mode", "str"),
    # Console-only by default. veRL defaults to ["console", "wandb"] and its wandb
    # backend calls wandb.init() during setup, which would kill a paid job in a
    # non-interactive container holding no credentials. Carried in configuration
    # rather than hardcoded here so the override traces to a config field (13.7).
    OverrideSpec("trainer.logger", "logger", "str"),
    OverrideSpec(
        "actor_rollout_ref.rollout.gpu_memory_utilization", "gpu_memory_utilization", "float"
    ),
    # --- Run length. total_training_steps is optional: absent means "run
    # total_epochs to completion", which is veRL's null default.
    OverrideSpec("trainer.total_epochs", "total_epochs", "int", minimum=1),
    OverrideSpec(
        "trainer.total_training_steps", "total_training_steps", "int", minimum=1, required=False
    ),
    # --- Reproducibility. Seeds veRL's dataloader shuffle generator with the
    # same seed prepare_data.py used to derive the splits.
    OverrideSpec("data.seed", "seed", "int"),
    # --- Batching. Not optional, and not merely tuning.
    #
    # veRL v0.7.1 ships use_dynamic_bsz=false with both ppo_micro_batch_size and
    # ppo_micro_batch_size_per_gpu unset, and ActorConfig.__post_init__ asserts
    # that at least one is set in that case. So a run that overrides none of these
    # does not get veRL's defaults -- it crashes during trainer construction.
    OverrideSpec("data.train_batch_size", "train_batch_size", "int", minimum=1),
    OverrideSpec(
        "actor_rollout_ref.actor.ppo_mini_batch_size", "ppo_mini_batch_size", "int", minimum=1
    ),
    # Setting this on the actor also configures the reference and rollout workers:
    # their configs derive log_prob_use_dynamic_bsz and
    # log_prob_max_token_len_per_gpu from the actor's values through Hydra
    # interpolation, so all three stay consistent by construction.
    OverrideSpec("actor_rollout_ref.actor.use_dynamic_bsz", "use_dynamic_bsz", "bool"),
    OverrideSpec(
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
        "ppo_max_token_len_per_gpu",
        "int",
        minimum=1,
    ),
    # Only meaningful when dynamic batching is off, so absent by default.
    OverrideSpec(
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
        "ppo_micro_batch_size_per_gpu",
        "int",
        minimum=1,
        required=False,
    ),
    # veRL defaults this to 2. Left implicit it silently decides how vLLM shards.
    OverrideSpec(
        "actor_rollout_ref.rollout.tensor_model_parallel_size",
        "tensor_model_parallel_size",
        "int",
        minimum=1,
    ),
    # --- Memory relief. Off by default; the documented remedy for an OOM.
    OverrideSpec(
        "actor_rollout_ref.actor.fsdp_config.param_offload", "param_offload", "bool"
    ),
    OverrideSpec(
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload", "optimizer_offload", "bool"
    ),
    # --- Checkpointing.
    #
    # veRL defaults save_freq to -1, meaning "save only when training finishes".
    # Under a SageMaker MaxRuntimeInSeconds cap that converts a merely-slow run
    # into a total loss: the job is stopped mid-training and no checkpoint exists
    # to export. Saving periodically means a truncated run still yields a model.
    OverrideSpec("trainer.save_freq", "save_freq", "int", minimum=1),
    OverrideSpec(
        "trainer.max_actor_ckpt_to_keep",
        "max_actor_ckpt_to_keep",
        "int",
        minimum=1,
        required=False,
    ),
    # --- Validation.
    #
    # veRL defaults test_freq to -1, and unlike save_freq's -1 that does not mean
    # "only at the end" -- it means never. The final-step validation is evaluated
    # inside the frequency guard:
    #
    #     if self.config.trainer.test_freq > 0 and (
    #         is_last_step or self.global_steps % self.config.trainer.test_freq == 0
    #     ):
    #
    # so is_last_step is never reached while test_freq is negative. A run on the
    # default emits one accuracy from val_before_train and nothing afterwards,
    # which is a baseline with nothing to compare against. Passing this explicitly
    # is what makes a run able to report whether the policy improved.
    OverrideSpec("trainer.test_freq", "test_freq", "int", minimum=1),
)
"""Every override whose value comes from a named hyperparameter.

Note what is absent. ``instance_type`` is carried in the hyperparameters for the
run record but has no veRL equivalent, and the GPU count per node comes from
SageMaker's ``SM_NUM_GPUS`` rather than from a lookup table keyed on instance
type -- such a table would be exactly the hardcoded literal that could diverge
from reality when a new instance family appears.
"""


@dataclass(frozen=True)
class ChannelSpec:
    """One veRL override whose value is a resolved SageMaker channel path."""

    verl_key: str
    channel: str


CHANNEL_OVERRIDES: tuple[ChannelSpec, ...] = (
    ChannelSpec("data.train_files", TRAIN_CHANNEL),
    ChannelSpec("data.val_files", VALIDATION_CHANNEL),
)
"""Training and validation data paths.

``entrypoint.py`` resolves these from ``SM_CHANNEL_TRAIN`` and
``SM_CHANNEL_VALIDATION`` and passes concrete file paths, because veRL reads the
file suffix to choose a loader and does not expand a directory.
"""

GPUS_PER_NODE_KEY = "trainer.n_gpus_per_node"
"""Set from ``SM_NUM_GPUS``, which is the only authoritative source.

veRL defaults this to 8 and multiplies it by ``trainer.nnodes`` to size the Ray
resource pool. Left at the default on a 4-GPU ``ml.g6e.12xlarge`` the pool would
request GPUs that do not exist and the job would wait forever instead of
failing, so this is not optional.
"""

CONTRACT_OVERRIDES: tuple[tuple[str, str], ...] = (
    # Requirement 7.3. SageMaker syncs this directory to S3 during the run, so
    # the value is dictated by the /opt/ml contract, not by a tunable field.
    ("trainer.default_local_dir", CHECKPOINT_DIR),
)
"""Values fixed by the execution environment rather than by configuration.

Kept to the strict minimum, and each one is a fact about where the job runs
rather than a tuning decision that belongs in ``configs/*.yaml``.
"""

_RENDERERS = {
    "str": lambda key, raw, spec: render_str(key, raw),
    "int": lambda key, raw, spec: render_int(key, raw, minimum=spec.minimum),
    "float": lambda key, raw, spec: render_float(key, raw),
    "bool": lambda key, raw, spec: render_bool(key, raw),
}


# --------------------------------------------------------------------------- #
# Pure override construction
# --------------------------------------------------------------------------- #


def _render_override(spec: OverrideSpec, raw: str) -> str:
    try:
        renderer = _RENDERERS[spec.kind]
    except KeyError as exc:  # pragma: no cover - guards a typo in the table above
        raise GrpoRunError(
            f"override {spec.verl_key!r} declares unknown kind {spec.kind!r}"
        ) from exc

    value = renderer(spec.hp_key, raw, spec)

    if spec.mandated_value is not None and value != spec.mandated_value:
        raise InvalidHyperparameterError(
            f"hyperparameter {spec.hp_key!r} must be {spec.mandated_value!r} for this "
            f"project's GRPO recipe, but hyperparameters.json carried {raw!r}; refusing "
            f"to train with a configuration that does not match the documented recipe"
        )

    return f"{spec.verl_key}={value}"


def render_data_files(channel: str, path: str) -> str:
    """Validate a channel path and return the value for ``data.*_files``.

    Accepts a single path or a Hydra bracket list of paths, and requires every
    element to carry a suffix veRL's dataset loader recognises. Catching a
    directory here converts a ``ValueError`` raised inside a Ray actor into an
    immediate failure that names the channel.
    """
    value = path.strip()
    if not value:
        raise MissingChannelError(f"data channel {channel!r} resolved to an empty path")

    inner = value[1:-1] if value.startswith("[") and value.endswith("]") else value
    elements = [element.strip() for element in inner.split(",")]

    if not all(elements):
        raise MissingChannelError(f"data channel {channel!r} lists an empty path among {value!r}")

    offenders = [
        element for element in elements if not element.lower().endswith(_SUPPORTED_DATA_SUFFIXES)
    ]
    if offenders:
        raise MissingChannelError(
            f"data channel {channel!r} must name files ending in one of "
            f"{list(_SUPPORTED_DATA_SUFFIXES)}, but got {offenders!r}; veRL reads the "
            f"file suffix to choose a loader and does not expand a directory, so the "
            f"channel mount point must be resolved to the parquet file it contains"
        )

    return value


def resolve_gpus_per_node(
    gpus_per_node: int | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Return the GPU count per node, from the argument or from ``SM_NUM_GPUS``.

    Taking the value as an argument is what keeps :func:`build_verl_argv` pure:
    ``entrypoint.py`` already holds this number on the ``ResourceConfig`` that
    ``start_ray.read_resource_config`` produced, and a test can pass it directly.
    Reading the environment is the fallback for a direct call.
    """
    if gpus_per_node is not None:
        if gpus_per_node < 1:
            raise InvalidHyperparameterError(
                f"gpus_per_node must be at least 1; got {gpus_per_node}"
            )
        return gpus_per_node

    env = os.environ if env is None else env
    raw = env.get("SM_NUM_GPUS")
    if raw is None:
        raise InvalidHyperparameterError(
            "SM_NUM_GPUS is not set and no gpus_per_node was supplied, so "
            f"{GPUS_PER_NODE_KEY} cannot be derived; veRL would default it to 8 and "
            "size the Ray resource pool for GPUs that may not exist"
        )
    return int(render_int("SM_NUM_GPUS", raw, minimum=1))


def build_verl_argv(
    hp: Mapping[str, str],
    channels: Mapping[str, str],
    *,
    gpus_per_node: int | None = None,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Build the veRL Hydra override list for a GRPO run.

    Pure with respect to ``hp`` and ``channels``: the same inputs always produce
    the same list, in a stable order, and nothing is read from disk or the
    network. ``gpus_per_node`` is the one value SageMaker rather than the
    configuration supplies; pass it explicitly and the function consults no
    ambient state at all.

    Preconditions: ``hp`` carries every ``required`` key in
    :data:`HP_OVERRIDES`, and ``channels`` names a resolved data file for both
    ``train`` and ``validation``.

    Postconditions: every returned element is a ``key=value`` token whose value
    was derived from a hyperparameter, a channel path, or one of the two
    documented environment-contract constants. The list always contains
    ``algorithm.adv_estimator=grpo``, ``actor_rollout_ref.rollout.name=vllm``,
    ``actor_rollout_ref.rollout.n``, ``actor_rollout_ref.actor.use_kl_loss``,
    ``actor_rollout_ref.actor.kl_loss_coef``, ``trainer.nnodes``, and
    ``trainer.default_local_dir=/opt/ml/checkpoints`` (Requirements 13.1 - 13.5,
    13.7, 7.3, Property 24).

    No reward override is emitted, by design; see the module docstring for why
    veRL requires none (Requirements 9.1, 9.2).
    """
    missing = [
        spec.hp_key
        for spec in HP_OVERRIDES
        if spec.required and not hp.get(spec.hp_key, "").strip()
    ]
    if missing:
        raise MissingHyperparameterError(
            f"hyperparameters.json is missing required key(s) {sorted(missing)}; "
            f"present keys are {sorted(hp)}"
        )

    argv: list[str] = []

    for spec in HP_OVERRIDES:
        raw = hp.get(spec.hp_key, "")
        if not spec.required and not raw.strip():
            continue
        argv.append(_render_override(spec, raw))

    missing_channels = [
        spec.channel for spec in CHANNEL_OVERRIDES if not channels.get(spec.channel, "").strip()
    ]
    if missing_channels:
        raise MissingChannelError(
            f"missing resolved path for data channel(s) {sorted(missing_channels)}; "
            f"present channels are {sorted(channels)}"
        )

    for channel_spec in CHANNEL_OVERRIDES:
        value = render_data_files(channel_spec.channel, channels[channel_spec.channel])
        argv.append(f"{channel_spec.verl_key}={value}")

    argv.append(f"{GPUS_PER_NODE_KEY}={resolve_gpus_per_node(gpus_per_node, env)}")

    for key, value in CONTRACT_OVERRIDES:
        argv.append(f"{key}={value}")

    return argv


# --------------------------------------------------------------------------- #
# Trainer invocation
# --------------------------------------------------------------------------- #


def run(
    argv: Sequence[str],
    *,
    python_executable: str | None = None,
    cwd: str | None = None,
) -> int:
    """Invoke ``verl.trainer.main_ppo`` with ``argv`` and return its exit code.

    Launched as a subprocess rather than imported and called, for three reasons
    that all matter inside a training job: veRL's entry point is a Hydra
    application that reads ``sys.argv``, rewrites the working directory, and
    reconfigures logging, none of which should happen to ``entrypoint.py``'s own
    process; it calls ``sys.exit`` on completion; and a process boundary yields
    an exit code directly, which is what this function's contract returns.

    Standard output and error are inherited rather than captured so veRL's
    progress reaches CloudWatch live. A caller that captured them would hold a
    multi-hour run's logs in memory and surface nothing until it ended.

    Returns the child's exit code. The caller decides what a non-zero code
    means; ``entrypoint.py`` propagates it so the SageMaker job fails.
    """
    command = [python_executable or sys.executable, "-m", VERL_TRAINER_MODULE, *argv]

    print(f"[run_grpo] invoking {VERL_TRAINER_MODULE} with {len(argv)} override(s):", flush=True)
    for override in argv:
        print(f"[run_grpo]   {override}", flush=True)

    completed = subprocess.run(command, cwd=cwd, check=False)

    if completed.returncode != 0:
        print(
            f"[run_grpo] {VERL_TRAINER_MODULE} exited {completed.returncode}; "
            f"see docs/troubleshooting.md",
            flush=True,
        )
    else:
        print(f"[run_grpo] {VERL_TRAINER_MODULE} completed successfully", flush=True)

    return completed.returncode

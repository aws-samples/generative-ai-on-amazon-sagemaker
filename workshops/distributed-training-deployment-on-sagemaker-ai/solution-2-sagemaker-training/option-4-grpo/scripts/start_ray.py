"""Ray cluster bootstrap for the in-container GRPO training job.

This module runs **only inside the GPU container**. Per Requirement 1.7 it imports
nothing from ``src/grpo_sagemaker/``; the only third-party import is ``ray``, and
that import is deliberately deferred into the functions that need it (see
``_import_ray``) so the pure election, membership, and counting logic stays
importable — and therefore testable — on a workstation with no Ray installed.

``ResourceConfig`` is defined here rather than in ``entrypoint.py`` because
``entrypoint.py`` imports this module to call :func:`bootstrap`; owning the type
here keeps that dependency one-directional.

Responsibilities, in the order :func:`bootstrap` performs them:

1. Refuse to continue when ``SM_CURRENT_HOST`` is absent from ``SM_HOSTS``,
   reporting both values (Requirement 12.5).
2. Elect exactly one head from the host list using a rule that is independent of
   list order, so every node independently agrees (Requirement 12.2). SageMaker
   does not promise ``resourceconfig.json`` lists hosts in the same order on
   every node, which is why "the first element" would be wrong.
3. Start a Ray head on the elected host and join every other host to it.
4. Block until the cluster reports the expected number of GPUs — ``SM_NUM_GPUS``
   on a single node, ``nodes * gpus_per_node`` on several (Requirements 12.1,
   12.3) — and fail with each node's registration state on timeout
   (Requirement 12.4).

Multi-node is an explicitly untested profile. It is implemented to spec and no
further.
"""

import json
import os
import socket
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

RESOURCE_CONFIG_PATH = Path("/opt/ml/input/config/resourceconfig.json")
"""SageMaker writes the host list and current host here (Requirement 7.2)."""

RAY_HEAD_PORT = 6379
"""Ray's default GCS port. The self-referencing security group permits it."""

DEFAULT_BOOTSTRAP_TIMEOUT_S = 600
"""Total wall-clock budget for DNS resolution, joining, and GPU registration.

A single deadline covers every waiting step so a stuck bootstrap cannot bill GPU
time indefinitely.
"""

_GPU_RESOURCE_KEY = "GPU"


class RayBootstrapError(RuntimeError):
    """Base class for every failure raised by this module."""


class HostMembershipError(RayBootstrapError):
    """``SM_CURRENT_HOST`` is not a member of ``SM_HOSTS`` (Requirement 12.5)."""


class RayBootstrapTimeoutError(RayBootstrapError):
    """Expected GPUs never registered within the budget (Requirement 12.4)."""


@dataclass(frozen=True)
class ResourceConfig:
    """The subset of the SageMaker resource contract this module needs.

    ``current_host`` and ``hosts`` come from ``resourceconfig.json``;
    ``gpus_per_node`` comes from ``SM_NUM_GPUS``. Carrying the GPU count as data
    rather than reading the environment inside :func:`bootstrap` keeps the
    expected-count arithmetic a pure function of this object.
    """

    current_host: str
    hosts: tuple[str, ...]
    gpus_per_node: int
    network_interface_name: str | None = None

    @property
    def node_count(self) -> int:
        return len(self.hosts)

    @property
    def is_multi_node(self) -> bool:
        return self.node_count > 1


@dataclass(frozen=True)
class RayClusterInfo:
    """What :func:`bootstrap` proved about the cluster before training starts."""

    head_host: str
    is_head: bool
    address: str
    node_count: int
    gpus_per_node: int
    gpu_count: int
    """GPUs actually registered with the cluster.

    On a single node this equals ``SM_NUM_GPUS``; on several it equals
    ``node_count * gpus_per_node``.
    """

    expected_gpu_count: int


# --------------------------------------------------------------------------- #
# Pure logic. No Ray, no subprocesses, no network, no filesystem.
# --------------------------------------------------------------------------- #


def elect_head(hosts: Sequence[str]) -> str:
    """Return the single host that every node will independently agree is head.

    The rule is the lexicographic minimum. What matters is not *which* host wins
    but that the rule is total, deterministic, and a function of the host *set*
    rather than its order: SageMaker may present ``hosts`` in a different order
    on each node, so an order-sensitive rule such as ``hosts[0]`` can elect two
    heads. ``min`` treats the list as a set, so reordering cannot change the
    answer (Requirement 12.2, Property 23).

    Host names are treated as opaque strings; no ``algo-N`` shape is assumed.
    """
    if not hosts:
        raise RayBootstrapError(
            "cannot elect a Ray head from an empty host list; "
            "expected SM_HOSTS to name at least the current host"
        )
    return min(hosts)


def validate_host_membership(current_host: str, hosts: Sequence[str]) -> None:
    """Raise unless ``current_host`` appears in ``hosts`` (Requirement 12.5).

    Both values are reported, because the useful diagnostic is the mismatch: a
    host absent from its own cluster's roster means the resource config and the
    environment disagree, and neither alone identifies the fault.
    """
    if current_host not in hosts:
        raise HostMembershipError(
            f"SM_CURRENT_HOST {current_host!r} is absent from SM_HOSTS "
            f"{list(hosts)!r}; refusing to bootstrap Ray because head election "
            f"would not be well defined"
        )


def is_head_node(resource_cfg: ResourceConfig) -> bool:
    """Return whether this node is the elected head.

    Validates membership first, so calling this on a node missing from its own
    host list raises rather than quietly returning ``False``.
    """
    validate_host_membership(resource_cfg.current_host, resource_cfg.hosts)
    return elect_head(resource_cfg.hosts) == resource_cfg.current_host


def expected_gpu_count(resource_cfg: ResourceConfig) -> int:
    """GPUs the cluster must register before training may start.

    ``node_count * gpus_per_node``, which on a single node reduces to
    ``SM_NUM_GPUS`` (Requirements 12.1, 12.3).
    """
    return resource_cfg.node_count * resource_cfg.gpus_per_node


def render_registration_state(nodes: Sequence[Mapping[str, object]]) -> str:
    """Render one line per Ray node: address, liveness, and registered GPUs.

    Takes the shape ``ray.nodes()`` returns but requires only plain mappings, so
    the timeout report is verifiable without a live cluster.
    """
    if not nodes:
        return "  (no nodes registered)"

    lines = []
    for node in nodes:
        address = node.get("NodeManagerAddress") or node.get("NodeID") or "<unknown>"
        hostname = node.get("NodeName") or "<unknown>"
        alive = bool(node.get("Alive", False))
        resources = node.get("Resources") or {}
        gpus = 0.0
        if isinstance(resources, Mapping):
            gpus = float(resources.get(_GPU_RESOURCE_KEY, 0.0) or 0.0)
        lines.append(f"  - {hostname} ({address}): alive={alive} gpus={gpus:g}")
    return "\n".join(lines)


def parse_resource_config(payload: Mapping[str, object], gpus_per_node: int) -> ResourceConfig:
    """Build a :class:`ResourceConfig` from parsed ``resourceconfig.json``.

    Kept separate from the file read so the parsing rules are testable without a
    ``/opt/ml`` tree.
    """
    current_host = payload.get("current_host")
    hosts = payload.get("hosts")

    if not isinstance(current_host, str) or not current_host:
        raise RayBootstrapError(
            f"resourceconfig.json has no usable 'current_host'; got {current_host!r}"
        )
    if not isinstance(hosts, Sequence) or isinstance(hosts, str) or not hosts:
        raise RayBootstrapError(f"resourceconfig.json has no usable 'hosts' list; got {hosts!r}")
    if not all(isinstance(host, str) and host for host in hosts):
        raise RayBootstrapError(
            f"resourceconfig.json 'hosts' must be non-empty strings; got {list(hosts)!r}"
        )

    interface = payload.get("network_interface_name")

    return ResourceConfig(
        current_host=current_host,
        hosts=tuple(hosts),
        gpus_per_node=gpus_per_node,
        network_interface_name=interface if isinstance(interface, str) else None,
    )


# --------------------------------------------------------------------------- #
# Environment and process interaction.
# --------------------------------------------------------------------------- #


def read_resource_config(
    path: Path = RESOURCE_CONFIG_PATH,
    env: Mapping[str, str] | None = None,
) -> ResourceConfig:
    """Read ``resourceconfig.json`` and ``SM_NUM_GPUS`` into a config object.

    ``entrypoint.py`` may call this or construct :class:`ResourceConfig` itself;
    either way the same validation applies.
    """
    env = os.environ if env is None else env

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RayBootstrapError(
            f"SageMaker resource configuration not found at {path}; "
            f"this script runs only inside a SageMaker training container"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RayBootstrapError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise RayBootstrapError(f"{path} must contain a JSON object; got {type(payload).__name__}")

    raw_gpus = env.get("SM_NUM_GPUS", "0")
    try:
        gpus_per_node = int(raw_gpus)
    except (TypeError, ValueError) as exc:
        raise RayBootstrapError(f"SM_NUM_GPUS is not an integer: {raw_gpus!r}") from exc
    if gpus_per_node < 0:
        raise RayBootstrapError(f"SM_NUM_GPUS must not be negative; got {gpus_per_node}")

    return parse_resource_config(payload, gpus_per_node)


def _import_ray():
    """Import ``ray`` on demand.

    Deferred so this module imports cleanly in an environment without Ray,
    keeping the pure functions above testable on a workstation.
    """
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - Ray is present in the container
        raise RayBootstrapError(
            "the 'ray' package is not importable; start_ray runs only inside the "
            "veRL GPU container, which ships Ray"
        ) from exc
    return ray


def _run(command: Sequence[str]) -> None:
    """Run a Ray CLI command, surfacing its output on failure."""
    print(f"[start_ray] $ {' '.join(command)}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.returncode != 0:
        raise RayBootstrapError(
            f"command {' '.join(command)!r} exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _resolve_host_ip(hostname: str, deadline: float) -> str:
    """Resolve a SageMaker host name to an IP, retrying until ``deadline``.

    Peer DNS entries are not guaranteed to exist the moment a container starts,
    so a single lookup can fail on a healthy cluster.
    """
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return socket.gethostbyname(hostname)
        except OSError as exc:
            last_error = exc
            time.sleep(2.0)
    raise RayBootstrapTimeoutError(
        f"could not resolve Ray head host {hostname!r} to an IP address before the "
        f"bootstrap timeout elapsed; last error: {last_error}"
    )


def _start_head(gpus_per_node: int, port: int, node_ip: str | None) -> None:
    command = [
        "ray",
        "start",
        "--head",
        f"--port={port}",
        f"--num-gpus={gpus_per_node}",
        "--disable-usage-stats",
    ]
    if node_ip is not None:
        command.append(f"--node-ip-address={node_ip}")
    _run(command)


def _join_worker(head_ip: str, gpus_per_node: int, port: int, node_ip: str | None) -> None:
    command = [
        "ray",
        "start",
        f"--address={head_ip}:{port}",
        f"--num-gpus={gpus_per_node}",
        "--disable-usage-stats",
    ]
    if node_ip is not None:
        command.append(f"--node-ip-address={node_ip}")
    _run(command)


def wait_for_workers(
    expected_gpus: int,
    timeout_s: int = DEFAULT_BOOTSTRAP_TIMEOUT_S,
    *,
    poll_seconds: float = 5.0,
) -> int:
    """Block until the cluster registers ``expected_gpus``, then return that count.

    Requires an already-connected Ray driver. On timeout raises
    :class:`RayBootstrapTimeoutError` carrying every node's registration state,
    so a cluster that never forms is diagnosable instead of leaving veRL to hang
    (Requirements 12.3, 12.4).
    """
    ray = _import_ray()
    deadline = time.monotonic() + timeout_s
    registered = 0

    while True:
        registered = int(ray.cluster_resources().get(_GPU_RESOURCE_KEY, 0))
        if registered >= expected_gpus:
            print(
                f"[start_ray] {registered} GPU(s) registered; expected {expected_gpus}",
                flush=True,
            )
            return registered
        if time.monotonic() >= deadline:
            break
        print(
            f"[start_ray] waiting for GPUs: {registered}/{expected_gpus} registered",
            flush=True,
        )
        time.sleep(poll_seconds)

    raise RayBootstrapTimeoutError(
        f"Ray registered {registered} of {expected_gpus} expected GPU(s) within "
        f"{timeout_s}s. Per-node registration state:\n"
        f"{render_registration_state(ray.nodes())}\n"
        f"If nodes are missing, check that the training job's security group "
        f"permits ingress from itself on port {RAY_HEAD_PORT} (see "
        f"docs/troubleshooting.md)."
    )


def bootstrap(
    resource_cfg: ResourceConfig,
    *,
    timeout_s: int = DEFAULT_BOOTSTRAP_TIMEOUT_S,
    port: int = RAY_HEAD_PORT,
) -> RayClusterInfo:
    """Start or join the Ray cluster and prove it is ready for veRL.

    Preconditions: ``resource_cfg`` was read successfully and its
    ``current_host`` appears in its ``hosts``.

    Postconditions: on a single node a local head is running and the returned
    ``gpu_count`` equals ``SM_NUM_GPUS``. On several nodes exactly one host is
    head, every other host has joined, and ``gpu_count`` equals
    ``node_count * gpus_per_node``. If that count is unmet within ``timeout_s``
    this raises with per-node registration state rather than letting veRL hang.

    The Ray cluster is started with the ``ray`` CLI and outlives this call; the
    verification driver connected here disconnects before returning, so veRL is
    free to attach its own driver afterwards.
    """
    validate_host_membership(resource_cfg.current_host, resource_cfg.hosts)

    head_host = elect_head(resource_cfg.hosts)
    is_head = resource_cfg.current_host == head_host
    expected = expected_gpu_count(resource_cfg)
    deadline = time.monotonic() + timeout_s

    print(
        f"[start_ray] host={resource_cfg.current_host} hosts={list(resource_cfg.hosts)} "
        f"head={head_host} role={'head' if is_head else 'worker'} "
        f"gpus_per_node={resource_cfg.gpus_per_node} expected_gpus={expected}",
        flush=True,
    )

    # Single node needs no addressing: Ray binds locally and nothing else joins.
    node_ip = (
        _resolve_host_ip(resource_cfg.current_host, deadline)
        if resource_cfg.is_multi_node
        else None
    )

    if is_head:
        _start_head(resource_cfg.gpus_per_node, port, node_ip)
        address = f"{node_ip or '127.0.0.1'}:{port}"
    else:
        head_ip = _resolve_host_ip(head_host, deadline)
        _join_worker(head_ip, resource_cfg.gpus_per_node, port, node_ip)
        address = f"{head_ip}:{port}"

    ray = _import_ray()
    ray.init(address="auto", ignore_reinit_error=True)
    try:
        remaining = max(1, int(deadline - time.monotonic()))
        registered = wait_for_workers(expected, remaining)
    finally:
        # Release only this verification driver. The cluster started above keeps
        # running so veRL's trainer can attach to it.
        ray.shutdown()

    return RayClusterInfo(
        head_host=head_host,
        is_head=is_head,
        address=address,
        node_count=resource_cfg.node_count,
        gpus_per_node=resource_cfg.gpus_per_node,
        gpu_count=registered,
        expected_gpu_count=expected,
    )

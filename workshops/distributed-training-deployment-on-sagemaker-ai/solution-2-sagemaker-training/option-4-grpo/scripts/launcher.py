# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ray distributed workload launcher for SageMaker training jobs.

This script serves as an entrypoint for SageMaker training jobs and handles both
single-node and multi-node distributed workload scenarios using Ray.

Supports both Python (.py) and Bash (.sh) scripts as entrypoints.
"""
from __future__ import absolute_import
import argparse
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import importlib.util
import logging
import os
import requests
import sagemaker_training.environment
import shlex
import signal
import subprocess
import sys
import time
import re
import tarfile
from typing import Dict, List, Optional, Any, Tuple
import yaml
import ray


# Configure logger
def get_logger():
    """Get configured logger for the launcher."""
    logger = logging.getLogger(__name__)

    # Prevent duplicate handlers in distributed environments
    if logger.handlers:
        return logger

    # Only add handler if none exist and we haven't already configured this logger
    if not hasattr(logger, "_configured"):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Mark this logger as configured to prevent re-configuration
        logger._configured = True

        # Prevent propagation to avoid duplicate messages from parent loggers
        logger.propagate = False

    return logger


logger = get_logger()

# Exit codes
SUCCESS_EXIT_CODE = 0
DEFAULT_FAILURE_CODE = 1

# Ray configuration constants
DEFAULT_RAY_PORT = 6379
RAY_WORKER_POLL_INTERVAL = 10  # seconds
RAY_CONNECTION_TIMEOUT = 300  # seconds (5 minutes)
# Port each Ray node exposes its Prometheus metrics on. Independent of the
# Dashboard: the exporter lives in the per-node agent, not the Dashboard UI.
RAY_METRICS_EXPORT_PORT = 8080
RAY_DASHBOARD_PORT = 8265

# Prometheus timer
PROMETHEUS_WAIT_SECONDS = 300

# Grafana configuration
DEFAULT_GRAFANA_PORT = 3000
GRAFANA_WAIT_SECONDS = 120
GRAFANA_CONFIG_WAIT_SECONDS = 60
# Folder where the Ray Dashboard generates its Grafana configuration
RAY_GRAFANA_CONFIG_DIR = "/tmp/ray/session_latest/metrics/grafana"
# Dashboard JSON shipped with this repository, provisioned into the embedded
# Grafana when it can be found next to the launcher.
REPO_DASHBOARD_FILENAME = "ray_sagemaker_training_dashboard.json"

# Ports already taken on the head node; the embedded Grafana must avoid them
RESERVED_PORTS = {
    DEFAULT_RAY_PORT: "the Ray GCS",
    RAY_METRICS_EXPORT_PORT: "the Ray metrics export",
    RAY_DASHBOARD_PORT: "the Ray Dashboard",
    9090: "the local Prometheus",
}

# Status and ready files
FAILURE_REASON_PATH = "/opt/ml/output/failure"

# Global variable to track Ray initialization status
ray_initialized = False
# Global variable to track if there was a failure
has_failure = False
# Global variable to track the Prometheus folder name
prometheus_folder_name = None
# Global variable to track the Grafana folder name
grafana_folder_name = None
# Global variable to track the embedded Grafana process
grafana_process = None


def signal_handler(signum: int, frame: Any) -> None:
    """Handle termination signals gracefully.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    global ray_initialized, has_failure
    signal_name = signal.Signals(signum).name
    logger.info("Received %s signal, initiating graceful shutdown...", signal_name)

    if ray_initialized:
        try:
            logger.info("Shutting down Ray...")
            ray.shutdown()
            ray_initialized = False
            logger.info("Ray shutdown completed successfully")
        except Exception as e:
            logger.warning("Error during Ray shutdown: %s", e)

    # Exit with failure code if there was a failure, otherwise success
    exit_code = DEFAULT_FAILURE_CODE if has_failure else SUCCESS_EXIT_CODE
    logger.info("Signal handler exiting with code: %s", exit_code)
    sys.exit(exit_code)


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def get_efa_supported_gpu_instances(region_name=None):
    """
    Dynamically fetch EFA-supported GPU instance types from AWS EC2.

    Args:
        region_name: AWS region name. If None, uses default region from environment/config

    Returns:
        List of SageMaker ML instance names (e.g., ['ml.p4d.24xlarge', 'ml.g5.12xlarge'])
    """
    try:
        # Use region from environment if not specified
        if region_name is None:
            region_name = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        ec2_client = boto3.client("ec2", region_name=region_name)

        # Use paginator to ensure we get all results
        paginator = ec2_client.get_paginator("describe_instance_types")

        page_iterator = paginator.paginate(
            Filters=[{"Name": "network-info.efa-supported", "Values": ["true"]}]
        )

        instance_names = []

        for page in page_iterator:
            for instance_type in page["InstanceTypes"]:
                # Only include instances that have GPU info
                if "GpuInfo" in instance_type:
                    ml_instance_name = f"ml.{instance_type['InstanceType']}"
                    instance_names.append(ml_instance_name)

        logger.info(
            f"Found {len(instance_names)} EFA-supported GPU instance types in {region_name}"
        )
        logger.debug(f"EFA-supported GPU instances: {sorted(instance_names)}")

        if len(instance_names) > 0:
            return sorted(instance_names)
        else:
            return SM_EFA_NCCL_INSTANCES_FALLBACK

    except NoCredentialsError:
        logger.warning(
            "AWS credentials not found. Using fallback static EFA instance list."
        )
        return SM_EFA_NCCL_INSTANCES_FALLBACK
    except ClientError as e:
        logger.warning(
            f"AWS API error when fetching EFA instances: {e}. Using fallback static list."
        )
        return SM_EFA_NCCL_INSTANCES_FALLBACK
    except Exception as e:
        logger.warning(
            f"Unexpected error when fetching EFA instances: {e}. Using fallback static list."
        )
        return SM_EFA_NCCL_INSTANCES_FALLBACK


# Fallback static lists (your current lists as backup)
SM_EFA_NCCL_INSTANCES_FALLBACK = [
    "ml.g6e.48xlarge",
    "ml.g6.24xlarge",
    "ml.g6e.8xlarge",
    "ml.g6e.16xlarge",
    "ml.g6.48xlarge",
    "ml.g5.24xlarge",
    "ml.g6.8xlarge",
    "ml.g4dn.12xlarge",
    "ml.p4de.24xlarge",
    "ml.g6e.12xlarge",
    "ml.g5.12xlarge",
    "ml.p6-b200.48xlarge",
    "ml.p4d.24xlarge",
    "ml.p5.48xlarge",
    "ml.p5.4xlarge",
    "ml.g5.16xlarge",
    "ml.p5en.48xlarge",
    "ml.g6.16xlarge",
    "ml.g6.12xlarge",
    "ml.dl1.24xlarge",
    "ml.g4dn.16xlarge",
    "ml.gr6.8xlarge",
    "ml.g6e.24xlarge",
    "ml.g5.8xlarge",
    "ml.g5.48xlarge",
    "ml.p3dn.24xlarge",
    "ml.g4dn.8xlarge",
]

SM_EFA_RDMA_INSTANCES = [
    "ml.p4d.24xlarge",
    "ml.p4de.24xlarge",
    "ml.trn1.32xlarge",
]

# Initialize dynamic lists at module level (cached)
try:
    SM_EFA_NCCL_INSTANCES = get_efa_supported_gpu_instances()
except Exception as e:
    logger.warning(
        f"Failed to initialize dynamic EFA instance lists: {e}. Using static fallback lists."
    )
    SM_EFA_NCCL_INSTANCES = SM_EFA_NCCL_INSTANCES_FALLBACK


def _parse_args():
    """Parse CLI arguments.

    Args:
        sys_args: Command line arguments

    Returns:
        Parsed arguments and unknown arguments
    """
    parser = argparse.ArgumentParser(
        description="SageMaker Ray distributed workload launcher", allow_abbrev=False
    )

    parser.add_argument(
        "-e",
        "--entrypoint",
        type=str,
        help="Entry point script path (e.g., training/train.py, ./training/train.py, or training/train.sh)",
    )

    parser.add_argument(
        "--head-instance-group",
        type=str,
        default=None,
        help="Instance group that should act as head node",
    )

    parser.add_argument(
        "--head-num-cpus",
        type=int,
        default=None,
        help="Number of CPUs to reserve to the head node",
    )

    parser.add_argument(
        "--head-num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to reserve to the head node",
    )

    parser.add_argument(
        "--include-dashboard",
        type=lambda v: v.lower() in ("true", "1", "yes", "on"),
        default=True,
        help="Include ray dashboard",
    )

    parser.add_argument(
        "--launch-prometheus",
        type=lambda v: v.lower() in ("true", "1", "yes", "on"),
        default=True,
        help="Launch local Prometheus on the head node for metrics collection",
    )

    parser.add_argument(
        "--prometheus-path",
        type=str,
        default=None,
        help="Path to the prometheus binary tar.gz file to copy to /opt/ml/code",
    )

    parser.add_argument(
        "--grafana-path",
        type=str,
        default=None,
        help="Path to the grafana tar.gz file to copy to /opt/ml/code. Providing it starts an embedded Grafana server on the head node",
    )

    parser.add_argument(
        "--grafana-port",
        type=int,
        # Defaults to None, not DEFAULT_GRAFANA_PORT, so the env-var fallback
        # below can tell "user did not pass the flag" from "user passed the
        # default value". The default is applied once both sources are resolved.
        default=None,
        help=f"Port used by the embedded Grafana server (default: {DEFAULT_GRAFANA_PORT})",
    )

    parser.add_argument(
        "--wait-shutdown",
        type=int,
        default=None,
        help="Number of seconds to wait before shutting down Ray server",
    )

    args, unknown = parser.parse_known_args()

    logger.info(f"Arguments: {args}")

    if unknown:
        logger.info(f"Ignoring unknown arguments: {unknown}")

    # Handle wait_shutdown parameter from environment variable if not provided as argument
    if args.wait_shutdown is None:
        env_wait_shutdown = os.environ.get("wait_shutdown")
        if env_wait_shutdown is not None:
            try:
                args.wait_shutdown = int(env_wait_shutdown)
                logger.info(
                    "Using wait_shutdown from environment variable: %s",
                    args.wait_shutdown,
                )
            except ValueError:
                logger.warning(
                    "Invalid wait_shutdown environment variable value: %s. Must be an integer.",
                    env_wait_shutdown,
                )

    # Handle head_instance_group parameter from environment variable if not provided as argument
    if args.head_instance_group is None:
        env_head_instance_group = os.environ.get("head_instance_group")
        if env_head_instance_group is not None:
            args.head_instance_group = env_head_instance_group
            logger.info(
                "Using head_instance_group from environment variable: %s",
                args.head_instance_group,
            )

    # Handle head_num_cpus parameter from environment variable if not provided as argument
    if args.head_num_cpus is None:
        env_head_num_cpus = os.environ.get("head_num_cpus")
        if env_head_num_cpus is not None:
            try:
                args.head_num_cpus = int(env_head_num_cpus)
                logger.info(
                    "Using head_num_cpus from environment variable: %s",
                    args.head_num_cpus,
                )
            except ValueError:
                logger.warning(
                    "Invalid head_num_cpus environment variable value: %s. Must be an integer.",
                    env_head_num_cpus,
                )

    # Handle head_num_gpus parameter from environment variable if not provided as argument
    if args.head_num_gpus is None:
        env_head_num_gpus = os.environ.get("head_num_gpus")
        if env_head_num_gpus is not None:
            try:
                args.head_num_gpus = int(env_head_num_gpus)
                logger.info(
                    "Using head_num_gpus from environment variable: %s",
                    args.head_num_gpus,
                )
            except ValueError:
                logger.warning(
                    "Invalid head_num_gpus environment variable value: %s. Must be an integer.",
                    env_head_num_gpus,
                )

    # Handle launch_prometheus parameter from environment variable if not provided as argument
    env_launch_prometheus = os.environ.get("launch_prometheus")
    if env_launch_prometheus is not None:
        try:
            # Convert string to boolean (handle common boolean representations)
            if env_launch_prometheus.lower() in ("true", "1", "yes", "on"):
                args.launch_prometheus = True
            elif env_launch_prometheus.lower() in ("false", "0", "no", "off"):
                args.launch_prometheus = False
            else:
                raise ValueError(f"Invalid boolean value: {env_launch_prometheus}")

            logger.info(
                "Using launch_prometheus from environment variable: %s",
                args.launch_prometheus,
            )
        except ValueError as e:
            logger.warning(
                "Invalid launch_prometheus environment variable value: %s. Must be a boolean (true/false, 1/0, yes/no, on/off). Error: %s",
                env_launch_prometheus,
                e,
            )

    # Handle prometheus_path parameter from environment variable if not provided as argument
    if args.prometheus_path is None:
        env_prometheus_path = os.environ.get("prometheus_path")
        if env_prometheus_path is not None:
            args.prometheus_path = env_prometheus_path
            logger.info(
                "Using prometheus_path from environment variable: %s",
                args.prometheus_path,
            )

    # Handle grafana_path parameter from environment variable if not provided as argument
    if args.grafana_path is None:
        env_grafana_path = os.environ.get("grafana_path")
        if env_grafana_path is not None:
            args.grafana_path = env_grafana_path
            logger.info(
                "Using grafana_path from environment variable: %s",
                args.grafana_path,
            )

    # Handle grafana_port parameter from environment variable if not provided as argument
    if args.grafana_port is None:
        env_grafana_port = os.environ.get("grafana_port")
        if env_grafana_port is not None:
            try:
                args.grafana_port = int(env_grafana_port)
                logger.info(
                    "Using grafana_port from environment variable: %s",
                    args.grafana_port,
                )
            except ValueError:
                logger.warning(
                    "Invalid grafana_port environment variable value: %s. Must be an integer.",
                    env_grafana_port,
                )

    # Apply the default only after both the flag and the env var were resolved,
    # so an explicit --grafana-port is never overridden by grafana_port.
    if args.grafana_port is None:
        args.grafana_port = DEFAULT_GRAFANA_PORT

    # Grafana starts after Ray, so a port Ray already owns leaves Grafana unable
    # to bind and dead for the whole job. Fall back to the default instead.
    if args.grafana_port in RESERVED_PORTS:
        logger.warning(
            "--grafana-port %s is already used by %s. Falling back to %s.",
            args.grafana_port,
            RESERVED_PORTS[args.grafana_port],
            DEFAULT_GRAFANA_PORT,
        )
        args.grafana_port = DEFAULT_GRAFANA_PORT

    # Grafana still runs without the Dashboard, it just cannot be embedded in the
    # Metrics tab (there is none). Reaching it then means port-forwarding directly.
    if args.grafana_path and not args.include_dashboard:
        logger.info(
            "The embedded Grafana will start without the Ray Dashboard, so there is "
            "no Metrics tab to embed its panels in. Reach it directly on port %s "
            "(e.g. through SSM port forwarding).",
            args.grafana_port,
        )

    # An external Grafana already serves the Metrics tab, and an explicit
    # RAY_GRAFANA_HOST wins over the embedded default, so starting the embedded
    # one too would leave a process nobody queries (plus a needless extraction).
    # A loopback value is treated as the user pointing at the embedded Grafana
    # itself, so it does not disable it.
    env_grafana_host = os.environ.get("RAY_GRAFANA_HOST")
    if (
        args.grafana_path
        and env_grafana_host
        and _is_remote_prometheus_host(env_grafana_host)
    ):
        logger.warning(
            "Ignoring grafana_path %s: RAY_GRAFANA_HOST=%s already points at an "
            "external Grafana, which takes precedence.",
            args.grafana_path,
            env_grafana_host,
        )
        args.grafana_path = None

    # Metrics collection is independent of the Dashboard: Ray always exposes the
    # per-node exporter, and the launcher generates its own scrape targets when
    # the Dashboard is not there to provide them.
    if args.launch_prometheus and not args.include_dashboard:
        logger.info(
            "Collecting metrics without the Ray Dashboard. Prometheus will scrape "
            "the Ray nodes using scrape targets generated by the launcher, since "
            "Ray only writes its own scrape config when the Dashboard runs."
        )

    if args.prometheus_path and not args.launch_prometheus:
        logger.warning(
            "prometheus_path %s will not be used because launch_prometheus is false.",
            args.prometheus_path,
        )

    # The embedded Grafana can only render what a Prometheus can serve it. With no
    # local Prometheus the datasource points at a dead loopback address, so Grafana
    # comes up with every panel empty.
    if args.grafana_path and not args.launch_prometheus:
        remote_prom = os.environ.get("RAY_PROMETHEUS_HOST")
        if not remote_prom:
            logger.warning(
                "The embedded Grafana is enabled but launch_prometheus is false and no "
                "RAY_PROMETHEUS_HOST is set: Grafana will start with a datasource "
                "pointing at a Prometheus that is not running, so all panels will be "
                "empty. Enable launch_prometheus or set RAY_PROMETHEUS_HOST."
            )
        elif _extract_amp_region(remote_prom):
            logger.info(
                "The embedded Grafana will query AMP at %s using SigV4 signed with "
                "the execution role, which therefore needs the AMP read actions "
                "(aps:QueryMetrics, aps:GetSeries, aps:GetLabels, "
                "aps:GetMetricMetadata). The job must also be able to reach AMP, "
                "e.g. through a VPC endpoint.",
                remote_prom,
            )

    # If entrypoint is provided, parse it and set environment variables
    if args.entrypoint:
        entrypoint_path = args.entrypoint

        # Remove leading "./" if present
        if entrypoint_path.startswith("./"):
            entrypoint_path = entrypoint_path[2:]

        # Split the path to get source_dir and entry_script
        if "/" in entrypoint_path:
            # Split into directory and filename
            path_parts = entrypoint_path.split("/")
            source_dir = "/".join(path_parts[:-1])  # All parts except the last
            entry_script = path_parts[-1]  # Last part (filename)
        else:
            # No directory, just filename
            source_dir = ""
            entry_script = entrypoint_path

        # Set environment variables
        os.environ["source_dir"] = source_dir
        os.environ["entry_script"] = entry_script

        logger.info("Entrypoint argument provided: %s", args.entrypoint)
        logger.info("Set source_dir=%s, entry_script=%s", source_dir, entry_script)

    return args, unknown


def _build_prometheus_command(prometheus_folder_name: str) -> str:
    """Build a safe prometheus command string.

    Args:
        prometheus_folder_name: Name of the prometheus folder

    Returns:
        Safely constructed prometheus command string
    """
    prometheus_binary_path = f"/opt/ml/code/{prometheus_folder_name}/prometheus"
    config_file_path = "/tmp/ray/session_latest/metrics/prometheus/prometheus.yml"
    return f"{shlex.quote(prometheus_binary_path)} --config.file={shlex.quote(config_file_path)}"


def _copy_prometheus_binary(prometheus_path: str) -> str:
    """Copy prometheus tar.gz file to /opt/ml/code directory and extract it.

    Args:
        prometheus_path: Path to the prometheus binary tar.gz file

    Returns:
        The name of the extracted folder

    Raises:
        FileNotFoundError: If the prometheus file doesn't exist
        Exception: If there are errors during file copy or extraction
    """
    import shutil
    import tarfile

    if not os.path.exists(prometheus_path):
        raise FileNotFoundError(f"Prometheus binary file not found: {prometheus_path}")

    # Ensure the destination directory exists
    destination_dir = "/opt/ml/code"
    os.makedirs(destination_dir, exist_ok=True)

    # Get the filename from the path
    filename = os.path.basename(prometheus_path)
    destination_path = os.path.join(destination_dir, filename)

    try:
        logger.info(
            "Copying prometheus binary from %s to %s", prometheus_path, destination_path
        )
        shutil.copy2(prometheus_path, destination_path)
        logger.info("Successfully copied prometheus binary to %s", destination_path)

        # Verify the file was copied successfully
        if os.path.exists(destination_path):
            original_size = os.path.getsize(prometheus_path)
            copied_size = os.path.getsize(destination_path)
            if original_size == copied_size:
                logger.info(
                    "File copy verification successful (size: %s bytes)", copied_size
                )
            else:
                logger.warning(
                    "File size mismatch: original=%s, copied=%s",
                    original_size,
                    copied_size,
                )
        else:
            raise Exception(
                f"File copy failed: destination file not found at {destination_path}"
            )

        # Extract the tar.gz file
        logger.info("Extracting prometheus binary from %s", destination_path)

        # Determine the folder name from the filename
        # Remove .tar.gz extension to get the folder name
        if filename.endswith(".tar.gz"):
            folder_name = filename[:-7]  # Remove '.tar.gz'
        elif filename.endswith(".tgz"):
            folder_name = filename[:-4]  # Remove '.tgz'
        else:
            # Fallback: remove common archive extensions
            folder_name = filename.rsplit(".", 1)[0]

        logger.info("Expected folder name after extraction: %s", folder_name)

        # Extract the tar.gz file in the destination directory with security validation
        with tarfile.open(destination_path, "r:gz") as tar:
            # Validate and extract members safely
            _safe_extract_all(tar, destination_dir)

        # Verify the extraction was successful
        extracted_folder_path = os.path.join(destination_dir, folder_name)
        if os.path.exists(extracted_folder_path):
            logger.info(
                "Successfully extracted prometheus binary to %s", extracted_folder_path
            )

            # Check if prometheus binary exists in the extracted folder
            prometheus_binary_path = os.path.join(extracted_folder_path, "prometheus")
            if os.path.exists(prometheus_binary_path):
                logger.info("Prometheus binary found at %s", prometheus_binary_path)
            else:
                logger.warning(
                    "Prometheus binary not found at expected path: %s",
                    prometheus_binary_path,
                )
        else:
            raise Exception(
                f"Extraction failed: folder not found at {extracted_folder_path}"
            )

        return folder_name

    except Exception as e:
        logger.error("Error copying or extracting prometheus binary: %s", e)
        raise


def _is_remote_prometheus_host(url: str) -> bool:
    """Check if the Prometheus host URL points to a remote server (not localhost)."""
    return not ("127.0.0.1" in url or "localhost" in url)


def _extract_amp_region(url: str) -> Optional[str]:
    """Extract AWS region from an Amazon Managed Prometheus URL.

    Args:
        url: AMP workspace URL (e.g., https://aps-workspaces.us-east-1.amazonaws.com/workspaces/ws-xxx)

    Returns:
        AWS region string if URL matches AMP pattern, None otherwise
    """
    match = re.search(r"aps-workspaces\.([a-z0-9-]+)\.amazonaws\.com", url)
    return match.group(1) if match else None


def _build_remote_write_url(remote_host: str, region: Optional[str] = None) -> str:
    """Build the full remote write URL from a host and optional region.

    AMP endpoints use /api/v1/remote_write, self-hosted Prometheus uses /api/v1/write.
    Handles hosts that may already include a subpath (e.g., https://example.com/prometheus).
    """
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(remote_host)
    rw_path = "/api/v1/remote_write" if region else "/api/v1/write"
    # Append the write path to the existing path, avoiding double slashes
    base_path = parsed.path.rstrip("/")
    new_path = f"{base_path}{rw_path}"
    return urlunparse(parsed._replace(path=new_path))


def _get_prometheus_config_path(use_ray_template: bool = False) -> str:
    """Get the Prometheus config file path.

    Args:
        use_ray_template: If True, return the Ray package template config path
            (used by `ray metrics launch-prometheus`). If False, return the
            session-specific config path (used by custom Prometheus binaries).

    Returns:
        Path to the Prometheus config file
    """
    if use_ray_template:
        from ray.dashboard.consts import PROMETHEUS_CONFIG_INPUT_PATH

        return PROMETHEUS_CONFIG_INPUT_PATH
    return "/tmp/ray/session_latest/metrics/prometheus/prometheus.yml"


def _write_prometheus_static_config(config_path: str, env: Any) -> bool:
    """Write a self-contained prometheus.yml that scrapes the Ray nodes directly.

    Ray's scrape configuration, and the service-discovery file it points at, are
    produced by the Dashboard. With `--include-dashboard false` there is no
    Dashboard, so Prometheus would come up with nothing to scrape even though the
    per-node metrics exporters are running. The launcher already resolves every
    node's IP (for the instance_type relabels), so it can enumerate the targets
    itself and drop the dependency on Ray's discovery altogether.

    The scrape job is deliberately named "ray" so `_inject_remote_write_config`
    and `_inject_sagemaker_relabels` keep working on it unchanged.

    Args:
        config_path: prometheus.yml to write
        env: SageMaker environment object, used to enumerate the cluster hosts

    Returns:
        True if a config was written
    """
    ip_type_map = _build_ip_instance_type_map(env)
    if not ip_type_map:
        logger.warning(
            "Could not resolve any Ray node IP, so no static Prometheus targets "
            "could be generated. Prometheus will start with whatever config is "
            "already present."
        )
        return False

    targets = sorted(f"{ip}:{RAY_METRICS_EXPORT_PORT}" for ip in ip_type_map)
    config = {
        "global": {"scrape_interval": "15s", "evaluation_interval": "15s"},
        "scrape_configs": [
            {"job_name": "ray", "static_configs": [{"targets": targets}]}
        ],
    }

    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(
            "Wrote a static Prometheus config at %s scraping %d Ray node(s): %s",
            config_path,
            len(targets),
            targets,
        )
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to write the static Prometheus config: %s", e)
        return False


def _inject_remote_write_config(
    remote_write_url: str,
    region: Optional[str] = None,
    config_path: Optional[str] = None,
    basic_auth: Optional[Dict[str, str]] = None,
) -> None:
    """Inject remote_write configuration into a prometheus.yml config file.

    Args:
        remote_write_url: Full remote write endpoint URL
        region: AWS region for SigV4 authentication (only for AMP endpoints)
        config_path: Path to the prometheus.yml file to modify
        basic_auth: Optional dict with 'username' and 'password' keys for basic auth
    """
    if config_path is None:
        config_path = "/tmp/ray/session_latest/metrics/prometheus/prometheus.yml"

    if not os.path.exists(config_path):
        logger.warning(
            "Prometheus config not found at %s, skipping remote_write injection",
            config_path,
        )
        return

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # yaml.safe_load returns None for an empty file; normalize to a dict.
        if config is None:
            config = {}

        # NOTE: the sagemaker_training_job_name label is NOT stamped here.
        # external_labels apply only to remote_write samples, which would leave
        # the label missing on the local Prometheus that backs the Ray
        # Dashboard. It is injected as a scrape-time relabel instead (see
        # _inject_sagemaker_relabels), so it is present in both.

        remote_write_entry = {
            "url": remote_write_url,
            "queue_config": {
                "max_samples_per_send": 1000,
                "max_shards": 200,
                "capacity": 2500,
            },
        }

        # Prometheus allows AT MOST ONE authentication mechanism per remote_write
        # entry; emitting both sigv4 and basic_auth makes the config invalid, so
        # Prometheus exits at startup and the job runs with no metrics at all.
        # An AMP endpoint is always SigV4, so it wins and basic auth is dropped.
        # (Reachable by leaving self-hosted credentials in the environment while
        # switching the host over to AMP.)
        if region:
            remote_write_entry["sigv4"] = {"region": region}
            if basic_auth:
                logger.warning(
                    "Both an AMP endpoint and RAY_PROMETHEUS_USERNAME/PASSWORD were "
                    "provided. AMP authenticates with SigV4, so basic auth is ignored; "
                    "unset the credentials to silence this warning."
                )
        elif basic_auth:
            remote_write_entry["basic_auth"] = {
                "username": basic_auth["username"],
                "password": basic_auth["password"],
            }

        if "remote_write" not in config:
            config["remote_write"] = []

        config["remote_write"].append(remote_write_entry)

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(
            "Injected remote_write config into %s (url: %s, sigv4: %s)",
            config_path,
            remote_write_url,
            region is not None,
        )

    except Exception as e:
        logger.error("Failed to inject remote_write config: %s", e)
        raise


def _resolve_ip_bounded(host: str, attempts: int = 3, delay: int = 2) -> Optional[str]:
    """Best-effort host -> IP with a SHORT, bounded retry.

    Deliberately NOT `_get_ip_from_host` (which retries 200x5s ~= 1000s): this
    runs before the Prometheus launch, so it must not block the head node for
    minutes if a worker's DNS entry lags. A node that does not resolve in time
    is simply left unlabeled.
    """
    import socket

    for _ in range(attempts):
        try:
            return socket.gethostbyname(host)
        except OSError:
            time.sleep(delay)
    return None


def _build_ip_instance_type_map(env: Any) -> Dict[str, str]:
    """Map each node IP -> its SageMaker instance type.

    The head knows the whole topology from the SageMaker environment, so it can
    label every node's series with its instance type (which differs per node on
    a heterogeneous cluster). Returns {ip: instance_type}; nodes that fail to
    resolve quickly are omitted (left unlabeled).
    """
    mapping: Dict[str, str] = {}
    try:
        if getattr(env, "is_hetero", False):
            for group in env.instance_groups_dict.values():
                itype = group.get("instance_type") or "unknown"
                for host in group.get("hosts", []):
                    ip = _resolve_ip_bounded(host)
                    if ip:
                        mapping[ip] = itype
                    else:
                        logger.warning(
                            "instance_type map: could not resolve host %s", host
                        )
        else:
            itype = getattr(env, "current_instance_type", None) or "unknown"
            for host in env.hosts:
                ip = _resolve_ip_bounded(host)
                if ip:
                    mapping[ip] = itype
                else:
                    logger.warning("instance_type map: could not resolve host %s", host)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed building IP->instance_type map: %s", e)
    return mapping


def _inject_sagemaker_relabels(
    config_path: Optional[str],
    ip_type_map: Dict[str, str],
    training_job_name: Optional[str] = None,
) -> None:
    """Add scrape-time relabel_configs for the SageMaker-specific labels.

    relabel_configs run at scrape time (before ingestion and remote_write), so
    the added labels reach both the local Prometheus that backs the Ray
    Dashboard and any remote_write destination. This is why they are used here
    instead of global external_labels, which only apply to remote_write.

    Two labels are added, both consumed by dashboard variables:
      - instance_type: per node, matched on the target's __address__
        (<ip>:<port>), since it differs per node on a heterogeneous cluster.
      - sagemaker_training_job_name: constant for the whole job.

    Must run before Prometheus reads its config (i.e. before launch).
    """
    if not config_path or not os.path.exists(config_path):
        return
    if not ip_type_map and not training_job_name:
        return
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

        jobs = config.get("scrape_configs", [])
        # Ray writes a single scrape job (job_name "ray"); fall back to the first.
        ray_job = next(
            (j for j in jobs if j.get("job_name") == "ray"),
            jobs[0] if jobs else None,
        )
        if ray_job is None:
            logger.warning("No scrape_configs found; skipping SageMaker relabels")
            return

        relabels = ray_job.setdefault("relabel_configs", [])
        for ip, itype in ip_type_map.items():
            relabels.append(
                {
                    "source_labels": ["__address__"],
                    "regex": "%s:.*" % re.escape(ip),
                    "target_label": "instance_type",
                    "replacement": itype,
                }
            )

        if training_job_name:
            # Matches every target, so the label lands on all scraped series.
            relabels.append(
                {
                    "source_labels": ["__address__"],
                    "regex": ".*",
                    "target_label": "sagemaker_training_job_name",
                    "replacement": training_job_name,
                }
            )

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(
            "Injected relabel_configs: instance_type for %d node(s), "
            "sagemaker_training_job_name=%s",
            len(ip_type_map),
            training_job_name or "<unset>",
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to inject SageMaker relabels: %s", e)


GRAFANA_INI_TEMPLATE = """[security]
allow_embedding = true

[auth.anonymous]
enabled = true
org_name = Main Org.
org_role = Viewer

[paths]
provisioning = {provisioning_dir}
"""


def _get_archive_root_folder(tar: tarfile.TarFile) -> str:
    """Return the single top level folder contained in a tar archive.

    Args:
        tar: The TarFile object to inspect

    Returns:
        Name of the top level folder

    Raises:
        ValueError: If the archive does not contain exactly one top level folder
    """
    # Archives written with pax headers carry a synthetic "pax_global_header"
    # member alongside the real content. It is metadata, not a top level folder,
    # so counting it would make a perfectly valid archive look ambiguous.
    ignored_roots = {".", "./", "pax_global_header"}
    roots = {
        name.split("/")[0]
        for name in tar.getnames()
        if name and not name.startswith("/") and name not in ignored_roots
    }
    roots -= ignored_roots
    if len(roots) != 1:
        raise ValueError(
            f"Expected a single top level folder in the archive, found: {sorted(roots)}"
        )
    return roots.pop()


def _copy_grafana_binary(grafana_path: str) -> str:
    """Copy the grafana tar.gz file to /opt/ml/code directory and extract it.

    Args:
        grafana_path: Path to the grafana tar.gz file

    Returns:
        The name of the extracted folder

    Raises:
        FileNotFoundError: If the grafana file doesn't exist
        Exception: If there are errors during file copy or extraction
    """
    import shutil

    if not os.path.exists(grafana_path):
        raise FileNotFoundError(f"Grafana binary file not found: {grafana_path}")

    destination_dir = "/opt/ml/code"
    os.makedirs(destination_dir, exist_ok=True)

    destination_path = os.path.join(destination_dir, os.path.basename(grafana_path))
    logger.info("Copying grafana archive from %s to %s", grafana_path, destination_path)
    shutil.copy2(grafana_path, destination_path)

    with tarfile.open(destination_path, "r:gz") as tar:
        # The Grafana archive name and its top level folder do not match
        # (grafana-12.0.1.linux-amd64.tar.gz extracts to grafana-v12.0.1), so the
        # folder name is read from the archive instead of derived from the file name.
        folder_name = _get_archive_root_folder(tar)
        _safe_extract_all(tar, destination_dir)

    extracted_folder_path = os.path.join(destination_dir, folder_name)
    if not os.path.isdir(extracted_folder_path):
        raise Exception(
            f"Extraction failed: folder not found at {extracted_folder_path}"
        )

    logger.info("Successfully extracted grafana to %s", extracted_folder_path)
    return folder_name


def _build_grafana_command(grafana_folder_name: str, config_path: str) -> str:
    """Build a safe grafana command string.

    Args:
        grafana_folder_name: Name of the grafana folder extracted in /opt/ml/code
        config_path: Path to the grafana.ini configuration file

    Returns:
        Safely constructed grafana command string

    Raises:
        FileNotFoundError: If no grafana binary is found in the extracted folder
    """
    grafana_home = f"/opt/ml/code/{grafana_folder_name}"
    bin_dir = os.path.join(grafana_home, "bin")
    # Grafana 10+ ships a single `grafana` binary with a `server` subcommand, while
    # earlier releases only ship `grafana-server`.
    if os.path.isfile(os.path.join(bin_dir, "grafana")):
        command = f"{shlex.quote(os.path.join(bin_dir, 'grafana'))} server"
    elif os.path.isfile(os.path.join(bin_dir, "grafana-server")):
        command = shlex.quote(os.path.join(bin_dir, "grafana-server"))
    else:
        raise FileNotFoundError(f"Grafana binary not found in {bin_dir}")

    return (
        f"{command} --homepath={shlex.quote(grafana_home)} "
        f"--config={shlex.quote(config_path)}"
    )


def _wait_for_ray_grafana_config(timeout: int = GRAFANA_CONFIG_WAIT_SECONDS) -> bool:
    """Wait for the Ray Dashboard to generate its Grafana configuration.

    Shortly after the head node starts, the Dashboard writes grafana.ini, the
    Prometheus datasource and the Ray dashboards (with the UIDs the Metrics tab
    embeds) under RAY_GRAFANA_CONFIG_DIR. Provisioning Grafana from those files is
    what makes the Metrics tab render without any manual dashboard import.

    Args:
        timeout: Maximum number of seconds to wait

    Returns:
        True if the configuration was generated, False otherwise
    """
    config_path = os.path.join(RAY_GRAFANA_CONFIG_DIR, "grafana.ini")
    deadline = time.time() + timeout

    while time.time() < deadline:
        if os.path.exists(config_path):
            return True
        time.sleep(2)

    return False


def _write_grafana_fallback_config(prometheus_host: str, prometheus_name: str) -> None:
    """Write a minimal Grafana configuration pointing at the local Prometheus.

    Only used when the Ray Dashboard did not generate its own Grafana configuration.
    Ray's own dashboards are not provisioned in that case; `_provision_repo_dashboard`
    still adds this repository's dashboard if it can be found.

    Args:
        prometheus_host: URL of the Prometheus server to use as datasource
        prometheus_name: Name of the Prometheus datasource in Grafana
    """
    provisioning_dir = os.path.join(RAY_GRAFANA_CONFIG_DIR, "provisioning")
    datasources_dir = os.path.join(provisioning_dir, "datasources")
    os.makedirs(datasources_dir, exist_ok=True)

    with open(os.path.join(RAY_GRAFANA_CONFIG_DIR, "grafana.ini"), "w") as f:
        f.write(GRAFANA_INI_TEMPLATE.format(provisioning_dir=provisioning_dir))

    datasource: Dict[str, Any] = {
        "name": prometheus_name,
        "type": "prometheus",
        "access": "proxy",
        "url": prometheus_host,
        "isDefault": True,
    }

    with open(os.path.join(datasources_dir, "default.yml"), "w") as f:
        yaml.dump(
            {"apiVersion": 1, "datasources": [datasource]}, f, default_flow_style=False
        )

    logger.info("Wrote fallback Grafana configuration in %s", RAY_GRAFANA_CONFIG_DIR)


def _enable_amp_sigv4_on_datasources() -> bool:
    """Add SigV4 signing to any provisioned datasource that points at AMP.

    AMP secures its query APIs with IAM, so an unsigned Prometheus datasource
    gets 403 on every panel. Applied to whatever provisioning is in place —
    the datasource Ray generates as well as the fallback written here — because
    Ray has no notion of AMP and emits a plain datasource.

    Grafana only honours `jsonData.sigV4Auth` when SigV4 support is switched on
    for the server itself, which `_launch_grafana` does via
    GF_AUTH_SIGV4_AUTH_ENABLED / AWS_SDK_LOAD_CONFIG. Signing then uses the
    container's credentials, i.e. the SageMaker execution role, which must hold
    the AMP read actions.

    Returns:
        True if at least one datasource was switched to SigV4
    """
    datasources_dir = os.path.join(
        RAY_GRAFANA_CONFIG_DIR, "provisioning", "datasources"
    )
    if not os.path.isdir(datasources_dir):
        return False

    patched = False
    for entry in sorted(os.listdir(datasources_dir)):
        if not entry.endswith((".yml", ".yaml")):
            continue
        path = os.path.join(datasources_dir, entry)
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f) or {}

            changed = False
            for datasource in config.get("datasources") or []:
                if datasource.get("type") != "prometheus":
                    continue
                region = _extract_amp_region(datasource.get("url") or "")
                if not region:
                    continue
                json_data = datasource.get("jsonData") or {}
                if json_data.get("sigV4Auth"):
                    continue
                json_data.update(
                    {
                        "sigV4Auth": True,
                        "sigV4AuthType": "default",
                        "sigV4Region": region,
                        "httpMethod": "POST",
                    }
                )
                datasource["jsonData"] = json_data
                changed = True
                logger.info(
                    "Datasource %s points at AMP in %s; enabling SigV4 signing",
                    datasource.get("name", "<unnamed>"),
                    region,
                )

            if changed:
                with open(path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                patched = True
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not enable SigV4 on datasource %s: %s", path, e)

    return patched


def _find_repo_dashboard() -> Optional[str]:
    """Locate this repository's Grafana dashboard JSON inside the container.

    Only `source_dir` is uploaded to the training job, so the dashboard is found
    only when the user shipped it alongside their code (or passed it on an input
    channel). Absence is normal and not an error.

    Returns:
        Path to the dashboard JSON, or None if it is not present
    """
    explicit = os.environ.get("grafana_dashboard_path")
    candidates = [explicit] if explicit else []

    code_dir = os.getcwd()
    for base in (code_dir, "/opt/ml/code", "/opt/ml/input/data/code"):
        candidates.append(os.path.join(base, REPO_DASHBOARD_FILENAME))
        candidates.append(
            os.path.join(base, "grafana-dashboards", REPO_DASHBOARD_FILENAME)
        )

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _provision_repo_dashboard() -> None:
    """Provision this repository's dashboard into the embedded Grafana.

    Ray provisions only its own dashboards, so the repo dashboard would otherwise
    have to be imported by hand — precisely what is impractical in an isolated
    environment. Its two SageMaker-specific variables (`TrainingJobName`,
    `InstanceType`) are backed by scrape-time relabels, so they resolve against
    the local Prometheus too.

    Best effort: any failure is logged and Grafana still starts.
    """
    dashboard_path = _find_repo_dashboard()
    if not dashboard_path:
        logger.info(
            "%s not found in the container; only Ray's own dashboards will be "
            "provisioned. Ship it in source_dir (or set grafana_dashboard_path) to "
            "have it provisioned automatically.",
            REPO_DASHBOARD_FILENAME,
        )
        return

    try:
        import shutil

        provisioning_dir = os.path.join(RAY_GRAFANA_CONFIG_DIR, "provisioning")
        dashboards_dir = os.path.join(provisioning_dir, "dashboards")
        target_dir = os.path.join(dashboards_dir, "sagemaker")
        os.makedirs(target_dir, exist_ok=True)

        shutil.copy2(
            dashboard_path, os.path.join(target_dir, REPO_DASHBOARD_FILENAME)
        )

        # A provider file is what makes Grafana load JSON from disk. Written under
        # a distinct name so it cannot clash with Ray's own provider.
        provider = {
            "apiVersion": 1,
            "providers": [
                {
                    "name": "sagemaker-ray",
                    "orgId": 1,
                    "folder": "SageMaker",
                    "type": "file",
                    "disableDeletion": False,
                    "editable": True,
                    "options": {"path": target_dir},
                }
            ],
        }
        with open(os.path.join(dashboards_dir, "sagemaker_provider.yml"), "w") as f:
            yaml.dump(provider, f, default_flow_style=False)

        logger.info(
            "Provisioned %s into the embedded Grafana from %s",
            REPO_DASHBOARD_FILENAME,
            dashboard_path,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not provision %s: %s", REPO_DASHBOARD_FILENAME, e)


def _extract_observability_binaries(args: argparse.Namespace) -> None:
    """Extract the Prometheus/Grafana archives, on the node that will run them.

    Called from the head and single-node paths only: workers never launch either
    process, so unpacking a few hundred megabytes there wastes disk and startup
    time. Sets the `prometheus_folder_name` / `grafana_folder_name` globals.

    A missing or corrupt Prometheus archive is fatal (it was requested
    explicitly), while Grafana stays best effort, matching the previous behaviour.

    Args:
        args: Command line arguments
    """
    global prometheus_folder_name, grafana_folder_name

    if args.prometheus_path and args.launch_prometheus:
        prometheus_folder_name = _copy_prometheus_binary(args.prometheus_path)
        logger.info("Prometheus folder name set to: %s", prometheus_folder_name)

    if args.grafana_path:
        try:
            grafana_folder_name = _copy_grafana_binary(args.grafana_path)
            logger.info("Grafana folder name set to: %s", grafana_folder_name)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not extract the grafana archive: %s", e)


def _launch_grafana(
    args: argparse.Namespace, runtime_env: Dict[str, Any]
) -> Optional[subprocess.Popen]:
    """Start the embedded Grafana server on the head node.

    When the Dashboard is running, Grafana is provisioned from the configuration Ray
    generates, so the Prometheus datasource and the dashboards embedded by the Metrics
    tab match the running cluster. Without the Dashboard, Ray writes no such config and
    a minimal one is written instead. Failures are logged and never fail the training
    job: metrics are still collected by Prometheus.

    Returns as soon as the process is spawned; call _wait_for_grafana_ready to
    confirm it came up.

    Args:
        args: Command line arguments
        runtime_env: Ray runtime environment configuration

    Returns:
        The Popen object of the Grafana process, or None if Grafana was not started
    """
    if not grafana_folder_name:
        return None

    try:
        # Only the Dashboard generates that config, so skip the wait entirely when
        # it is disabled rather than burning GRAFANA_CONFIG_WAIT_SECONDS on a file
        # that will never appear.
        ray_config_ready = args.include_dashboard and _wait_for_ray_grafana_config()
        if not ray_config_ready:
            if args.include_dashboard:
                logger.warning(
                    "Ray did not generate a Grafana configuration in %s, falling back to a minimal one",
                    RAY_GRAFANA_CONFIG_DIR,
                )
            else:
                logger.info(
                    "Dashboard disabled, so Ray generates no Grafana configuration; "
                    "writing a minimal one pointing at the local Prometheus."
                )
            _write_grafana_fallback_config(
                runtime_env.get("RAY_PROMETHEUS_HOST", "http://127.0.0.1:9090"),
                runtime_env.get("RAY_PROMETHEUS_NAME", "Prometheus"),
            )

        # Added on top of whichever config is in place, so the repo dashboard is
        # available alongside Ray's own.
        _provision_repo_dashboard()

        # Ray emits a plain Prometheus datasource, so an AMP URL needs SigV4
        # switched on both per-datasource and server-wide (below) to answer.
        needs_sigv4 = _enable_amp_sigv4_on_datasources()

        grafana_home = f"/opt/ml/code/{grafana_folder_name}"
        grafana_cmd = _build_grafana_command(
            grafana_folder_name, os.path.join(RAY_GRAFANA_CONFIG_DIR, "grafana.ini")
        )
        grafana_env = {
            "GF_PATHS_PROVISIONING": os.path.join(
                RAY_GRAFANA_CONFIG_DIR, "provisioning"
            ),
            "GF_PATHS_DATA": os.path.join(grafana_home, "data"),
            "GF_PATHS_LOGS": os.path.join(grafana_home, "data", "log"),
            "GF_PATHS_PLUGINS": os.path.join(grafana_home, "data", "plugins"),
            "GF_SERVER_HTTP_PORT": str(args.grafana_port),
            # The Ray Dashboard renders Grafana panels in iframes without
            # authenticating against Grafana, so embedding and anonymous read access
            # are both required. Grafana is only reachable from inside the training
            # container, or through SSM port forwarding.
            "GF_SECURITY_ALLOW_EMBEDDING": "true",
            "GF_AUTH_ANONYMOUS_ENABLED": "true",
            "GF_AUTH_ANONYMOUS_ORG_ROLE": "Viewer",
        }

        if needs_sigv4:
            # Grafana ships with SigV4 support OFF, so a datasource's sigV4Auth
            # flag is ignored until the server itself enables it. Both variables
            # are required; AWS_SDK_LOAD_CONFIG lets the bundled AWS SDK pick up
            # the container's credentials (the SageMaker execution role).
            grafana_env["GF_AUTH_SIGV4_AUTH_ENABLED"] = "true"
            grafana_env["AWS_SDK_LOAD_CONFIG"] = "true"
            logger.info(
                "Enabled server-side SigV4 support for the AMP datasource. The "
                "execution role needs the AMP read actions (aps:QueryMetrics, "
                "aps:GetSeries, aps:GetLabels, aps:GetMetricMetadata)."
            )

        logger.info("Starting embedded Grafana with command: %s", grafana_cmd)
        return _run_subprocess_command_async(
            grafana_cmd,
            wait_in_seconds=0,
            stdout_file="/tmp/grafana_stdout.log",
            stderr_file="/tmp/grafana_stderr.log",
            env_vars=grafana_env,
        )

    except Exception as e:  # noqa: BLE001
        logger.warning("Could not start the embedded Grafana server: %s", e)
        return None


def _wait_for_grafana_ready(
    process: Optional[subprocess.Popen], grafana_port: int
) -> Optional[subprocess.Popen]:
    """Poll the embedded Grafana until it answers on /api/health.

    Kept separate from _launch_grafana so this wait can overlap with work the
    head node has to do anyway (waiting for the workers to join). Grafana only
    needs to be up by the time someone opens the Dashboard Metrics tab, not
    before the cluster is assembled, so blocking the critical path on it would
    delay the start of every job that enables the flag.

    Args:
        process: The Grafana process returned by _launch_grafana
        grafana_port: Port the embedded Grafana listens on

    Returns:
        The process if it is running, None if it exited or was never started
    """
    if process is None:
        return None

    health_url = f"http://127.0.0.1:{grafana_port}/api/health"
    start_time = time.time()

    logger.info(
        "Waiting for Grafana to become ready (max %s seconds)...",
        GRAFANA_WAIT_SECONDS,
    )

    while time.time() - start_time < GRAFANA_WAIT_SECONDS:
        if process.poll() is not None:
            logger.warning("Grafana process exited with code %s", process.returncode)
            _read_and_log_process_logs("/tmp/grafana_stderr.log")
            return None

        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                logger.info(
                    "Grafana health check passed after %.1f seconds",
                    time.time() - start_time,
                )
                return process
        except Exception as e:
            logger.debug("Grafana not ready yet: %s", e)

        time.sleep(2)

    logger.warning(
        "Grafana did not become ready within %s seconds", GRAFANA_WAIT_SECONDS
    )
    _read_and_log_process_logs("/tmp/grafana_stderr.log")
    return process


def _shutdown_grafana() -> None:
    """Terminate the embedded Grafana server if it is running."""
    global grafana_process

    if grafana_process is None:
        return

    if grafana_process.poll() is None:
        logger.info("Shutting down grafana")
        grafana_process.terminate()
        try:
            grafana_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Grafana did not stop gracefully, killing it")
            grafana_process.kill()

    grafana_process = None


def _is_efa_device_present() -> bool:
    """Return True only if an EFA device is actually attached to THIS node.

    Instance-type capability (membership in SM_EFA_NCCL_INSTANCES) only tells us
    whether the type *can* have EFA, not whether EFA is actually attached for
    this job. The two can differ (e.g. a coordinator node, or topologies where
    SageMaker does not attach EFA), so we confirm against the real device.

    Detection order (first reliable signal wins):
      1. `fi_info -p efa` — authoritative: returns 0 only when libfabric can
         actually open the EFA provider. This is exactly the condition
         FI_PROVIDER=efa relies on.
      2. sysfs / device node fallback, used only if `fi_info` is unavailable.
    """
    try:
        result = subprocess.run(
            ["fi_info", "-p", "efa"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except FileNotFoundError:
        # fi_info not on PATH — fall back to inspecting the RDMA device tree.
        logger.info("fi_info not found; falling back to sysfs EFA detection")
    except Exception as e:  # noqa: BLE001
        logger.warning("EFA detection via fi_info failed (%s); using sysfs fallback", e)

    try:
        return any(os.scandir("/sys/class/infiniband"))
    except FileNotFoundError:
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning("EFA sysfs detection failed: %s", e)
        return False


def _create_runtime_environment(args: argparse.Namespace, env: Any) -> Dict[str, Any]:
    """
    Create the Ray runtime environment configuration based on instance type.
    Includes ALL current environment variables to ensure complete environment is available to Ray workers.

    Args:
        env: SageMaker environment object

    Returns:
        Dict containing ALL environment variables plus Ray-specific configurations
    """
    # Start with ALL current environment variables
    runtime_env = dict(os.environ)

    # Get the source_dir directory path for Ray workers
    source_dir = os.environ.get("source_dir", "")
    current_dir = os.getcwd()
    absolute_source_dir = (
        os.path.join(current_dir, source_dir) if source_dir else current_dir
    )

    # Get current PYTHONPATH and add our source_dir directory
    current_pythonpath = runtime_env.get("PYTHONPATH", "")
    if current_pythonpath:
        new_pythonpath = f"{absolute_source_dir}:{current_pythonpath}"
    else:
        new_pythonpath = absolute_source_dir

    # Override/add specific Ray and networking environment variables
    runtime_env.update(
        {
            "NCCL_SOCKET_IFNAME": str(env.network_interface_name),
            "NCCL_PROTO": "simple",
            "PYTHONPATH": new_pythonpath,  # Add source_dir directory to PYTHONPATH for Ray workers
        }
    )

    # Configure EFA/RDMA based on ACTUAL device presence, not just instance-type
    # capability. An instance type can be EFA-capable yet have no EFA device
    # attached for a given job (e.g. a coordinator-only head, or topologies where
    # SageMaker does not attach EFA). Forcing FI_PROVIDER=efa on such a node
    # points libfabric at a device that does not exist. We therefore require both
    # that the type is EFA-capable AND that an EFA device is actually present.
    #
    # An explicit value supplied via the ModelTrainer `environment` dict always
    # wins, so users can override the autodetection when they know better.
    efa_capable = env.current_instance_type in SM_EFA_NCCL_INSTANCES
    rdma_capable = env.current_instance_type in SM_EFA_RDMA_INSTANCES
    efa_present = _is_efa_device_present() if (efa_capable or rdma_capable) else False

    if "FI_PROVIDER" in os.environ:
        logger.info(
            "FI_PROVIDER set from environment: %s", os.environ["FI_PROVIDER"]
        )
    elif efa_capable and efa_present:
        runtime_env["FI_PROVIDER"] = "efa"
        logger.info("EFA device detected; setting FI_PROVIDER=efa")
    elif efa_capable:
        logger.info(
            "Instance type %s is EFA-capable but no EFA device detected; "
            "not setting FI_PROVIDER",
            env.current_instance_type,
        )

    if rdma_capable and efa_present:
        if "FI_EFA_USE_DEVICE_RDMA" not in os.environ:
            runtime_env["FI_EFA_USE_DEVICE_RDMA"] = "1"
        if "RDMAV_FORK_SAFE" not in os.environ:
            runtime_env["RDMAV_FORK_SAFE"] = "1"

    if args.launch_prometheus:
        # Configure Prometheus host - Ray Dashboard connects to local Prometheus
        runtime_env["RAY_PROMETHEUS_HOST"] = "http://127.0.0.1:9090"

        # If a remote Prometheus host is provided, store it for remote_write
        env_prometheus_host = os.environ.get("RAY_PROMETHEUS_HOST")
        if env_prometheus_host and _is_remote_prometheus_host(env_prometheus_host):
            # Store the remote URL for remote_write, Dashboard still uses localhost
            runtime_env["RAY_REMOTE_WRITE_PROMETHEUS_HOST"] = (
                env_prometheus_host.rstrip("/")
            )
            # Pass through basic auth credentials if provided
            rw_username = os.environ.get("RAY_PROMETHEUS_USERNAME")
            rw_password = os.environ.get("RAY_PROMETHEUS_PASSWORD")
            if rw_username and rw_password:
                runtime_env["RAY_PROMETHEUS_USERNAME"] = rw_username
                runtime_env["RAY_PROMETHEUS_PASSWORD"] = rw_password
            logger.info(
                "Detected remote Prometheus host: %s. Local Prometheus will remote_write to it.",
                env_prometheus_host,
            )
    else:
        if os.environ.get("RAY_PROMETHEUS_HOST") is not None:
            runtime_env["RAY_PROMETHEUS_HOST"] = os.environ.get("RAY_PROMETHEUS_HOST")

    if os.environ.get("RAY_PROMETHEUS_NAME") is not None:
        runtime_env["RAY_PROMETHEUS_NAME"] = os.environ.get("RAY_PROMETHEUS_NAME")

    # Configure Grafana environment variables for Ray Dashboard integration
    if os.environ.get("RAY_GRAFANA_HOST") is not None:
        runtime_env["RAY_GRAFANA_HOST"] = os.environ.get("RAY_GRAFANA_HOST")

    # RAY_GRAFANA_IFRAME_HOST: Used by browser to fetch Grafana panels
    if os.environ.get("RAY_GRAFANA_IFRAME_HOST") is not None:
        runtime_env["RAY_GRAFANA_IFRAME_HOST"] = os.environ.get(
            "RAY_GRAFANA_IFRAME_HOST"
        )
    elif os.environ.get("RAY_GRAFANA_HOST") is not None:
        runtime_env["RAY_GRAFANA_IFRAME_HOST"] = os.environ.get("RAY_GRAFANA_HOST")

    if args.grafana_path:
        # The embedded Grafana runs on the head node, next to the local Prometheus.
        # The Dashboard backend queries it over loopback, while the browser loads the
        # panel iframes through the SSM port forwarding, hence the different host.
        # Values explicitly provided by the user always win.
        runtime_env.setdefault(
            "RAY_GRAFANA_HOST", f"http://127.0.0.1:{args.grafana_port}"
        )
        runtime_env.setdefault(
            "RAY_GRAFANA_IFRAME_HOST", f"http://localhost:{args.grafana_port}"
        )

    if runtime_env.get("RAY_PROMETHEUS_HOST") is not None:
        logger.info(
            "Configured Prometheus host: %s", runtime_env.get("RAY_PROMETHEUS_HOST")
        )

    if runtime_env.get("RAY_GRAFANA_HOST") is not None:
        logger.info("Configured Grafana host: %s", runtime_env.get("RAY_GRAFANA_HOST"))

    logger.info(
        "Ray runtime environment contains %d total environment variables",
        len(runtime_env),
    )
    logger.info("Ray runtime environment: %s", runtime_env)

    logger.info("source_dir directory added to PYTHONPATH: %s", absolute_source_dir)
    return runtime_env


def _execute_entry_script(
    runtime_env: Dict[str, Any],
) -> None:
    """Execute the entry script dynamically based on environment variables.

    This function loads and executes the script based on its file extension:
    - .py files: Loaded as Python modules and executed
    - .sh files: Executed as bash scripts

    Args:
        runtime_env: Ray runtime environment configuration

    Raises:
        ValueError: If required environment variables are not set or unsupported file type
        FileNotFoundError: If the script file cannot be found
        Exception: If there are errors during script execution
    """
    global has_failure
    source_dir = os.environ.get("source_dir", "")
    entry_script = os.environ.get("entry_script")

    # source_dir can be empty if the script is in the same directory as launcher
    # but entry_script is always required
    if not entry_script:
        raise ValueError("entry_script environment variable is required")

    logger.info("Raw source_dir from env: '%s'", source_dir)
    logger.info("Raw entry_script from env: '%s'", entry_script)

    # Get current working directory and construct absolute paths
    current_dir = os.getcwd()  # This should be /opt/ml/input/data/code

    # Handle empty source_dir (script in same directory as launcher)
    if source_dir:
        absolute_source_dir = os.path.join(current_dir, source_dir)
    else:
        absolute_source_dir = current_dir

    script_path = os.path.join(absolute_source_dir, entry_script)

    logger.info("Current working directory: %s", current_dir)
    logger.info("Absolute source directory: %s", absolute_source_dir)
    logger.info("Script path: %s", script_path)

    # Debug: List contents of directories
    if os.path.exists(current_dir):
        logger.info("Contents of %s: %s", current_dir, os.listdir(current_dir))
    if os.path.exists(absolute_source_dir):
        logger.info(
            "Contents of %s: %s", absolute_source_dir, os.listdir(absolute_source_dir)
        )

    logger.info("Script path exists: %s", os.path.exists(script_path))

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Entry script not found: {script_path}")

    # Determine script type based on file extension
    script_extension = os.path.splitext(entry_script)[1].lower()

    try:
        if script_extension == ".py":
            _execute_python_script(script_path, absolute_source_dir)
        elif script_extension == ".sh":
            _execute_bash_script(script_path, absolute_source_dir, runtime_env)
        else:
            raise ValueError(
                f"Unsupported script type: {script_extension}. Only .py and .sh files are supported."
            )

        logger.info("Script execution completed successfully")

    except Exception as e:
        has_failure = True
        logger.error("Error executing entry script %s: %s", script_path, e)
        logger.error("Exception type: %s", type(e).__name__)
        import traceback

        logger.error("Traceback: %s", traceback.format_exc())
        raise
    finally:
        # Restore original working directory
        os.chdir(current_dir)
        logger.info("Restored working directory to: %s", current_dir)


def _execute_python_script(script_path: str, absolute_source_dir: str) -> None:
    """Execute a Python script using importlib.

    Args:
        script_path: Full path to the Python script
        absolute_source_dir: Absolute path to the source directory
    """
    # Change to the absolute source directory so relative imports work
    os.chdir(absolute_source_dir)
    logger.info("Changed working directory to: %s", absolute_source_dir)
    logger.info("Current working directory after change: %s", os.getcwd())
    logger.info("Contents of current directory: %s", os.listdir("."))

    # Add the absolute source directory to Python path if not already there
    if absolute_source_dir not in sys.path:
        sys.path.insert(0, absolute_source_dir)
        logger.info("Added %s to sys.path", absolute_source_dir)

    # Use importlib to load and execute the script
    # We are calling the module as __main__ to ensure it runs as a script
    logger.info("Loading and executing Python script using importlib...")

    spec = importlib.util.spec_from_file_location("__main__", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _execute_bash_script(
    script_path: str,
    absolute_source_dir: str,
    runtime_env: Dict[str, Any],
) -> None:
    """Execute a bash script using subprocess.

    Args:
        script_path: Full path to the bash script
        absolute_source_dir: Absolute path to the source directory
        runtime_env: Ray runtime environment configuration
    """
    # Change to the absolute source directory so relative paths in the script work
    os.chdir(absolute_source_dir)
    logger.info("Changed working directory to: %s", absolute_source_dir)
    logger.info("Current working directory after change: %s", os.getcwd())
    logger.info("Contents of current directory: %s", os.listdir("."))

    # Execute the bash script
    logger.info("Executing bash script: %s", script_path)

    # Use subprocess to run the bash script
    try:
        result = subprocess.run(
            ["bash", script_path],
            check=True,
            capture_output=True,
            text=True,
            env=runtime_env,
            cwd=absolute_source_dir,
        )

        # Log the output
        if result.stdout:
            logger.info("Script stdout:\n%s", result.stdout)
        if result.stderr:
            logger.info("Script stderr:\n%s", result.stderr)

        logger.info("Bash script completed with return code: %s", result.returncode)

    except subprocess.CalledProcessError as e:
        logger.error("Bash script failed with return code: %s", e.returncode)
        if e.stdout:
            logger.error("Script stdout:\n%s", e.stdout)
        if e.stderr:
            logger.error("Script stderr:\n%s", e.stderr)
        raise


def _get_ip_from_host(host):
    """Get the IP address from the current host."""
    import socket

    ip_wait_time = 200
    counter = 0
    ip = ""

    while counter < ip_wait_time and ip == "":
        try:
            ip = socket.gethostbyname(host)
            break
        except:
            counter += 1
            time.sleep(5)

    if counter == ip_wait_time and ip == "":
        raise Exception(
            "Exceeded max wait time of %ss for hostname resolution" % ip_wait_time
        )

    logger.info("IP address for %s is %s", host, ip)
    return ip


def _log_environment_debug_info() -> None:
    """Log environment variables for debugging."""
    source_dir = os.environ.get("source_dir", "Not set")
    entry_script = os.environ.get("entry_script", "Not set")
    logger.info("source_dir: %s", source_dir)
    logger.info("entry_script: %s", entry_script)


def _read_and_log_process_logs(log_file_path: str) -> None:
    """Read and log the contents of a background process log file.

    Args:
        log_file_path: Path to the log file to read and log
    """
    try:
        if not os.path.exists(log_file_path):
            logger.warning("Log file not found: %s", log_file_path)
            return

        with open(log_file_path, "r") as f:
            content = f.read()

        if content.strip():
            logger.info("Contents of %s:\n%s", log_file_path, content)
        else:
            logger.info("Log file %s is empty", log_file_path)

    except Exception as e:
        logger.error("Error reading log file %s: %s", log_file_path, e)


def _run_script(
    runtime_env: Dict[str, Any],
) -> None:
    """
    Execute the dynamically loaded entry script.

    Args:
        runtime_env: Ray runtime environment configuration
    """
    global has_failure
    try:
        _execute_entry_script(runtime_env)
        logger.info("Entry script execution complete")
    except Exception as e:
        has_failure = True
        logger.error("Error executing entry script: %s", e)
        raise


def _validate_command(args: List[str]) -> None:
    """Validate that command uses only allowed executables.

    Args:
        args: Command arguments list

    Raises:
        ValueError: If command is not allowed
    """
    if not args:
        raise ValueError("Empty command not allowed")

    executable = args[0]
    if (
        executable in ["ray", "bash"]
        or executable.startswith("./prometheus-")
        or executable.startswith("/opt/ml/code/prometheus-")
        or _is_grafana_executable(executable)
    ):
        return

    raise ValueError(f"Command not allowed: {executable}")


def _is_grafana_executable(executable: str) -> bool:
    """Check whether an executable is a Grafana binary extracted in /opt/ml/code.

    Matched against the folder `_copy_grafana_binary` actually extracted rather
    than a `grafana-` name prefix: the official archives happen to unpack to
    `grafana-v<version>`, but a repackaged or renamed archive (`grafana`,
    `my-grafana`, ...) is equally valid and was previously refused here, which
    surfaced only as "Could not start the embedded Grafana server". Comparing the
    exact `bin` directory is also stricter than the old prefix test.

    Args:
        executable: Path of the executable to check

    Returns:
        True if the executable is an allowed Grafana binary
    """
    if os.path.basename(executable) not in ("grafana", "grafana-server"):
        return False

    if grafana_folder_name:
        expected_bin_dir = os.path.join("/opt/ml/code", grafana_folder_name, "bin")
        return os.path.dirname(executable) == expected_bin_dir

    # Nothing extracted yet (e.g. validation before extraction): fall back to
    # requiring the binary to live under a grafana* folder in /opt/ml/code.
    return executable.startswith("/opt/ml/code/grafana")


def _run_subprocess_command_with_env(
    command: str, env_vars: Dict[str, str], check: bool = True
) -> Tuple[int, str, str]:
    """
    Run a shell command with custom environment variables and return the result.

    Args:
        command: Shell command to execute (will be parsed safely)
        env_vars: Dictionary of environment variables to set for the process
        check: Whether to raise an exception on non-zero exit status

    Returns:
        Tuple of (return_code, stdout, stderr)

    Raises:
        subprocess.CalledProcessError: If check is True and the command returns non-zero exit status
    """
    try:
        # Parse command string into arguments to avoid shell injection
        args = shlex.split(command)
        _validate_command(args)

        result = subprocess.run(
            args, shell=False, check=check, capture_output=True, text=True, env=env_vars
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        logger.error("Command '%s' failed with exit status %s", command, e.returncode)
        logger.error("STDOUT: %s", e.stdout)
        logger.error("STDERR: %s", e.stderr)
        if check:
            raise
        return e.returncode, e.stdout, e.stderr


def _run_subprocess_command(command: str, check: bool = True) -> Tuple[int, str, str]:
    """
    Run a shell command and return the result.

    Args:
        command: Shell command to execute (will be parsed safely)
        check: Whether to raise an exception on non-zero exit status

    Returns:
        Tuple of (return_code, stdout, stderr)

    Raises:
        subprocess.CalledProcessError: If check is True and the command returns non-zero exit status
    """
    try:
        # Parse command string into arguments to avoid shell injection
        args = shlex.split(command)
        _validate_command(args)

        result = subprocess.run(
            args, shell=False, check=check, capture_output=True, text=True
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        logger.error("Command '%s' failed with exit status %s", command, e.returncode)
        logger.error("STDOUT: %s", e.stdout)
        logger.error("STDERR: %s", e.stderr)
        if check:
            raise
        return e.returncode, e.stdout, e.stderr


def _run_subprocess_command_async(
    command: str,
    stdout_file: Optional[str] = None,
    stderr_file: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
    wait_in_seconds: int = 0,
) -> subprocess.Popen:
    """
    Run a shell command asynchronously using subprocess.Popen without waiting for completion.

    Args:
        command: Shell command to execute (will be parsed safely)
        stdout_file: Optional file path to redirect stdout (defaults to subprocess.PIPE)
        stderr_file: Optional file path to redirect stderr (defaults to subprocess.PIPE)
        env_vars: Optional dictionary of environment variables to set for the process
        wait_in_seconds: Number of seconds to wait after starting the process (0 = no wait)

    Returns:
        subprocess.Popen object representing the running process

    Example:
        # Start a long-running process without waiting
        process = _run_subprocess_command_async("sleep 60")
        logger.info("Process started with PID: %s", process.pid)

        # Start a process and wait 5 seconds before returning
        process = _run_subprocess_command_async("ray start --head", wait_in_seconds=5)

        # You can check if it's still running
        if process.poll() is None:
            logger.info("Process is still running")

        # Or terminate it later if needed
        # process.terminate()
    """
    try:
        # Parse command string into arguments to avoid shell injection
        args = shlex.split(command)
        _validate_command(args)

        # Validate and set up stdout redirection
        if stdout_file:
            # Validate file path to prevent directory traversal
            if ".." in stdout_file or stdout_file.startswith("/"):
                if not stdout_file.startswith("/tmp/"):
                    raise ValueError(f"Invalid stdout file path: {stdout_file}")
            stdout = open(stdout_file, "w")
        else:
            stdout = subprocess.PIPE

        # Validate and set up stderr redirection
        if stderr_file:
            # Validate file path to prevent directory traversal
            if ".." in stderr_file or stderr_file.startswith("/"):
                if not stderr_file.startswith("/tmp/"):
                    raise ValueError(f"Invalid stderr file path: {stderr_file}")
            stderr = open(stderr_file, "w")
        else:
            stderr = subprocess.PIPE

        # Set up environment variables
        process_env = os.environ.copy()
        if env_vars:
            process_env.update(env_vars)

        logger.info("Starting async command: %s", command)

        # Start the process without waiting for completion
        process = subprocess.Popen(
            args,
            shell=False,
            stdout=stdout,
            stderr=stderr,
            env=process_env,
            text=True,
        )

        logger.info("Async process started with PID: %s", process.pid)

        # Close file handles immediately after Popen since they're now owned by the subprocess
        if stdout_file and stdout != subprocess.PIPE:
            stdout.close()
        if stderr_file and stderr != subprocess.PIPE:
            stderr.close()

        # Wait for specified seconds if requested
        if wait_in_seconds > 0:
            logger.info("Waiting %s seconds after starting process...", wait_in_seconds)
            time.sleep(wait_in_seconds)

            # Check if process is still running after wait
            if process.poll() is None:
                logger.info(
                    "Process is still running after %s seconds wait", wait_in_seconds
                )
            else:
                logger.info(
                    "Process completed during %s seconds wait with return code: %s",
                    wait_in_seconds,
                    process.returncode,
                )

        return process

    except Exception as e:
        logger.error("Error starting async command '%s': %s", command, e)
        # Clean up file handles if they were opened
        if stdout_file and "stdout" in locals() and hasattr(stdout, "close"):
            stdout.close()
        if stderr_file and "stderr" in locals() and hasattr(stderr, "close"):
            stderr.close()
        raise


def _safe_extract_all(tar: tarfile.TarFile, path: str) -> None:
    """Safely extract all members from a tar file, preventing directory traversal attacks.

    Args:
        tar: The TarFile object to extract from
        path: The destination directory path

    Raises:
        ValueError: If a member path is unsafe (contains directory traversal)
        Exception: If extraction fails for any other reason
    """

    def _is_safe_path(member_path: str, destination: str) -> bool:
        """Check if the member path is safe for extraction."""
        # Resolve the full path and check if it's within the destination directory
        full_path = os.path.realpath(os.path.join(destination, member_path))
        destination_path = os.path.realpath(destination)
        return (
            full_path.startswith(destination_path + os.sep)
            or full_path == destination_path
        )

    def _is_safe_member(member: tarfile.TarInfo) -> bool:
        """Check if a tar member is safe to extract."""
        # Check for absolute paths
        if os.path.isabs(member.name):
            return False

        # Check for directory traversal patterns
        if ".." in member.name:
            return False

        # Check for null bytes
        if "\x00" in member.name:
            return False

        # Additional check using path resolution
        return _is_safe_path(member.name, path)

    # Validate all members before extraction
    unsafe_members = []
    for member in tar.getmembers():
        if not _is_safe_member(member):
            unsafe_members.append(member.name)

    if unsafe_members:
        raise ValueError(
            f"Unsafe tar members detected (potential directory traversal): {unsafe_members}"
        )

    # Extract all members (now that we've validated they're safe)
    for member in tar.getmembers():
        try:
            tar.extract(member, path)
        except Exception as e:
            logger.error("Failed to extract member %s: %s", member.name, e)
            raise


def _setup_head_node(
    hosts: List[str],
    runtime_env: Dict[str, Any],
    args: argparse.Namespace,
    env: Any,
) -> int:
    """Configure and run the Ray head node.

    Args:
        hosts: List of all hosts in the cluster
        runtime_env: Ray runtime environment configuration
        args: Command line arguments
        env: SageMaker environment variables object

    Returns:
        Return code from Ray stop command
    """
    global ray_initialized, has_failure, prometheus_folder_name, grafana_process

    # This node runs Prometheus/Grafana, so it is the one that unpacks them.
    _extract_observability_binaries(args)

    try:
        num_cpus = (
            args.head_num_cpus if args.head_num_cpus is not None else env.num_cpus
        )
        num_gpus = (
            args.head_num_gpus if args.head_num_gpus is not None else env.num_gpus
        )

        logger.info("CPUs for the head node: %s", num_cpus)
        logger.info("GPUs for the head node: %s", num_gpus)

        # Build Ray start command safely (no runtime-env option available in CLI)
        ray_cmd = f"ray start --head --num-cpus={shlex.quote(str(num_cpus))} --num-gpus={shlex.quote(str(num_gpus))} --port={DEFAULT_RAY_PORT}"
        ray_init_kwargs = {
            "address": "auto",
            "include_dashboard": args.include_dashboard,
            "runtime_env": {"env_vars": runtime_env},
        }

        # The metrics endpoint is always exposed on a known port. It is served by
        # the per-node agent, not the Dashboard UI, so metrics collection does not
        # require --include-dashboard.
        ray_cmd += f" --metrics-export-port={RAY_METRICS_EXPORT_PORT}"

        if args.include_dashboard:
            ray_cmd += (
                f" --dashboard-host=0.0.0.0 --dashboard-port={RAY_DASHBOARD_PORT}"
            )
            ray_init_kwargs["dashboard_host"] = "0.0.0.0"
            ray_init_kwargs["dashboard_port"] = RAY_DASHBOARD_PORT

        # Set environment variables for the Ray process
        env_for_ray = os.environ.copy()
        env_for_ray.update(runtime_env)
        _run_subprocess_command_with_env(ray_cmd, env_for_ray, check=True)

        ray.init(**ray_init_kwargs)

        if args.launch_prometheus:
            # Determine config path and whether we use custom or Ray-managed Prometheus
            use_custom_prometheus = args.prometheus_path and prometheus_folder_name
            remote_host = runtime_env.get("RAY_REMOTE_WRITE_PROMETHEUS_HOST")
            # ray metrics launch-prometheus reads the Ray package template;
            # custom Prometheus uses the session config.
            config_path = _get_prometheus_config_path(
                use_ray_template=not use_custom_prometheus
            )

            # With no Dashboard, Ray writes neither the scrape config nor the
            # service-discovery file it references, so enumerate the targets here.
            if not args.include_dashboard:
                _write_prometheus_static_config(config_path, env)

            # Inject remote_write config BEFORE launching Prometheus so the
            # process starts with the correct configuration already in place.
            if remote_host:
                region = _extract_amp_region(remote_host)
                remote_write_url = _build_remote_write_url(remote_host, region)
                rw_user = runtime_env.get("RAY_PROMETHEUS_USERNAME")
                rw_pass = runtime_env.get("RAY_PROMETHEUS_PASSWORD")
                basic_auth = (
                    {"username": rw_user, "password": rw_pass}
                    if rw_user and rw_pass
                    else None
                )
                _inject_remote_write_config(
                    remote_write_url,
                    region,
                    config_path=config_path,
                    basic_auth=basic_auth,
                )

            # Add the instance_type and sagemaker_training_job_name labels via
            # scrape-time relabeling so the dashboard can filter by instance type
            # (e.g. ml.g4dn.12xlarge) and by training job. This runs regardless of
            # remote_write and must happen before launch, since Prometheus reads
            # its config only at startup.
            _inject_sagemaker_relabels(
                config_path,
                _build_ip_instance_type_map(env),
                training_job_name=os.environ.get("TRAINING_JOB_NAME"),
            )

            logger.info("Launching prometheus")
            if use_custom_prometheus:
                prometheus_cmd = _build_prometheus_command(prometheus_folder_name)
                logger.info(
                    "Starting custom Prometheus with command: %s", prometheus_cmd
                )
                prometheus_process = _run_subprocess_command_async(
                    prometheus_cmd,
                    wait_in_seconds=0,
                    stdout_file="/tmp/prometheus_stdout.log",
                    stderr_file="/tmp/prometheus_stderr.log",
                )
            else:
                prometheus_process = _run_subprocess_command_async(
                    "ray metrics launch-prometheus",
                    wait_in_seconds=0,
                    stdout_file="/tmp/prometheus_stdout.log",
                    stderr_file="/tmp/prometheus_stderr.log",
                )

            # Poll for Prometheus readiness
            start_time = time.time()
            prometheus_ready = False

            logger.info(
                "Waiting for Prometheus to become ready (max %s seconds)...",
                PROMETHEUS_WAIT_SECONDS,
            )

            while time.time() - start_time < PROMETHEUS_WAIT_SECONDS:
                # Fail fast if the Prometheus process has already exited
                if prometheus_process.poll() is not None:
                    logger.warning(
                        "Prometheus process exited with code %s. "
                        "This may indicate missing internet access for downloading the binary. "
                        "Use --prometheus-path to provide a pre-downloaded binary.",
                        prometheus_process.returncode,
                    )
                    break

                try:
                    response = requests.get(
                        f"{runtime_env['RAY_PROMETHEUS_HOST']}/-/healthy",
                        timeout=5,
                    )
                    if response.status_code == 200:
                        elapsed_time = time.time() - start_time
                        logger.info(
                            "Prometheus health check passed after %.1f seconds",
                            elapsed_time,
                        )
                        prometheus_ready = True
                        break
                except Exception as e:
                    logger.warning(str(e))
                    pass

                time.sleep(2)

            if not prometheus_ready:
                logger.warning(
                    "Prometheus did not become ready within %s seconds",
                    PROMETHEUS_WAIT_SECONDS,
                )

            _read_and_log_process_logs("/tmp/prometheus_stderr.log")

        if args.grafana_path:
            grafana_process = _launch_grafana(args, runtime_env)

        ray_initialized = True

        # Wait for all worker nodes to connect
        cluster_size = len(hosts)
        connected_nodes = 1
        start_time = time.time()

        while connected_nodes < cluster_size:
            if time.time() - start_time > RAY_CONNECTION_TIMEOUT:
                logger.warning(
                    "Timed out waiting for all nodes to connect after %s seconds",
                    RAY_CONNECTION_TIMEOUT,
                )
                logger.warning(
                    "Proceeding with %s/%s nodes", connected_nodes, cluster_size
                )
                break

            time.sleep(1)
            resources = ray.available_resources().keys()
            # Ray exposes a synthetic "node:__internal_head__" resource in
            # addition to the per-node "node:<ip>" entries; excluding it keeps
            # the count equal to the real number of nodes, otherwise the head is
            # counted twice and the loop exits one worker early.
            curr_nodes = [
                r
                for r in resources
                if r.startswith("node:") and r != "node:__internal_head__"
            ]
            connected_nodes = len(curr_nodes)

            if connected_nodes < cluster_size and (time.time() - start_time) % 30 < 1:
                logger.info(
                    "Waiting for nodes to connect: %s/%s", connected_nodes, cluster_size
                )
                logger.info("Currently connected nodes: %s", curr_nodes)

        logger.info("All nodes connected to the Ray cluster!")

        # Confirm Grafana came up only now: its startup overlapped the wait for
        # the workers above, so this usually returns immediately.
        if args.grafana_path:
            grafana_process = _wait_for_grafana_ready(
                grafana_process, args.grafana_port
            )

        _run_script(runtime_env)

    except Exception as e:
        has_failure = True
        logger.error("Error in head node setup or script execution: %s", e)
        raise
    finally:
        if not has_failure:
            _wait_before_shutdown(args.wait_shutdown)

        _shutdown_grafana()

        if args.launch_prometheus:
            logger.info("Shutting down prometheus")
            _run_subprocess_command("ray metrics shutdown-prometheus", check=False)

        _shutdown_ray_safely()
        returncode, _, _ = _run_subprocess_command("ray stop", check=False)
        return returncode


def _setup_worker_node(
    head: str,
    runtime_env: Dict[str, Any],
) -> int:
    """Configure and run a Ray worker node.

    Args:
        head: Hostname of the head node
        runtime_env: Ray runtime environment configuration

    Returns:
        Return code from Ray stop command
    """
    master_ip = _get_ip_from_host(head)
    # Connect to the head node - construct command safely (no runtime-env option available in CLI)
    ray_address = f"{master_ip}:{DEFAULT_RAY_PORT}"
    ray_start_cmd = f"ray start --address={shlex.quote(ray_address)}"

    # Set environment variables for the Ray process
    env_for_ray = os.environ.copy()
    env_for_ray.update(runtime_env)
    _run_subprocess_command_with_env(ray_start_cmd, env_for_ray, check=True)

    # Keep worker node alive until head node completes
    poll_count = 0

    while _is_ray_alive():
        time.sleep(RAY_WORKER_POLL_INTERVAL)
        poll_count += 1
        if poll_count % 6 == 0:  # Log every ~60 seconds
            logger.info("Worker node still connected to Ray cluster")

    logger.info("Head node is down, shutting down worker node")
    returncode, _, _ = _run_subprocess_command("ray stop", check=False)
    return returncode


def _is_ray_alive() -> bool:
    """
    Check if the Ray cluster is still running.

    Returns:
        True if Ray is running, False otherwise
    """
    try:
        returncode, _, _ = _run_subprocess_command("ray status", check=False)
        return returncode == 0
    except Exception as e:
        logger.warning("Error checking Ray status: %s", e)
        return False


def _get_cluster_configuration(
    args: argparse.Namespace, env: Any
) -> Tuple[List[str], str, int]:
    """
    Get cluster configuration for both homogeneous and heterogeneous setups.

    Args:
        args: Command line arguments
        env: SageMaker environment variables

    Returns:
        Tuple of (all_hosts, head_host, compute_host_count)

        compute_host_count counts only the hosts that CONTRIBUTE COMPUTE, so it
        excludes a coordinator-only head. It is reported for observability and
        must NOT be used to decide the cluster topology: use len(all_hosts),
        which is the number of instances in the job. The two differ whenever the
        head is coordinator-only, and a job with a coordinator head plus a
        single worker has compute_host_count == 1 while genuinely being a
        two-instance cluster.
    """
    if env.is_hetero:
        return _get_heterogeneous_cluster_config(args, env)
    else:
        return _get_homogeneous_cluster_config(env)


def _get_homogeneous_cluster_config(env: Any) -> Tuple[List[str], str, int]:
    """Get configuration for homogeneous cluster.

    Every host runs the same instance type and contributes compute, so
    compute_host_count is simply the number of hosts.
    """
    hosts = env.hosts
    head_host = hosts[0] if hosts else ""
    return hosts, head_host, len(hosts)


def _get_heterogeneous_cluster_config(
    args: argparse.Namespace, env: Any
) -> Tuple[List[str], str, int]:
    """Get configuration for heterogeneous cluster."""
    all_hosts = []
    head_host = ""
    compute_host_count = 0

    # Find head instance group and collect all hosts
    for instance_group in env.instance_groups_dict.values():
        group_hosts = instance_group["hosts"]
        all_hosts.extend(group_hosts)

        if instance_group["instance_group_name"] == args.head_instance_group:
            head_host = group_hosts[0]  # First host in head group becomes head node

            # Check if head node should participate in computation
            if args.head_num_cpus is not None:
                head_num_cpus = args.head_num_cpus
            else:
                head_num_cpus = env.num_cpus

            if args.head_num_gpus is not None:
                head_num_gpus = args.head_num_gpus
            else:
                head_num_gpus = env.num_gpus

            if head_num_cpus == 0 and head_num_gpus == 0:
                # The head itself contributes no compute. Any OTHER host in the
                # head group still joins as a normal worker with full resources,
                # so only the head is excluded.
                compute_host_count += len(group_hosts) - 1
                logger.info("Head node configured as coordinator only (0 CPUs, 0 GPUs)")
            else:
                # Head node participates in computation
                compute_host_count += len(group_hosts)
        else:
            # All hosts in non-head groups are workers
            compute_host_count += len(group_hosts)

    if not head_host:
        raise ValueError(
            "Head instance group '%s' not found" % args.head_instance_group
        )

    return all_hosts, head_host, compute_host_count


def _setup_single_node_ray(
    args: argparse.Namespace, runtime_env: Dict[str, Any], env: Any
) -> int:
    """
    Set up Ray for single-node execution.

    Args:
        args: Command line arguments
        runtime_env: Ray runtime environment configuration
        env: SageMaker environment variables object (for instance_type labeling)
    """
    global ray_initialized, has_failure, prometheus_folder_name, grafana_process

    logger.info("Found a single host, initializing Ray as a single node")

    # On a one-instance job the single node has to do the work, so it always
    # starts with Ray's autodetected resources. Reserving 0 CPUs/GPUs here would
    # leave nothing able to run a task, so an explicit head reservation is
    # deliberately ignored rather than honored - say so instead of doing it
    # silently.
    if args.head_num_cpus == 0 or args.head_num_gpus == 0:
        logger.warning(
            "Ignoring head_num_cpus=%s / head_num_gpus=%s: this job has a single "
            "instance, which must contribute compute. A coordinator-only head "
            "only makes sense on a multi-instance cluster.",
            args.head_num_cpus,
            args.head_num_gpus,
        )

    # This node runs Prometheus/Grafana, so it is the one that unpacks them.
    _extract_observability_binaries(args)

    try:
        # Build Ray start command (no runtime-env option available in CLI)
        ray_cmd = f"ray start --head --port={DEFAULT_RAY_PORT}"
        ray_init_kwargs = {
            "address": "auto",
            "include_dashboard": args.include_dashboard,
            "runtime_env": {"env_vars": runtime_env},
        }

        # Always expose the metrics endpoint; it does not depend on the Dashboard.
        ray_cmd += f" --metrics-export-port={RAY_METRICS_EXPORT_PORT}"

        if args.include_dashboard:
            ray_cmd += (
                f" --dashboard-host=0.0.0.0 --dashboard-port={RAY_DASHBOARD_PORT}"
            )
            ray_init_kwargs["dashboard_host"] = "0.0.0.0"
            ray_init_kwargs["dashboard_port"] = RAY_DASHBOARD_PORT

        # Set environment variables for the Ray process
        env_for_ray = os.environ.copy()
        env_for_ray.update(runtime_env)
        _run_subprocess_command_with_env(ray_cmd, env_for_ray, check=True)

        ray.init(**ray_init_kwargs)

        if args.launch_prometheus:
            use_custom_prometheus = args.prometheus_path and prometheus_folder_name
            remote_host = runtime_env.get("RAY_REMOTE_WRITE_PROMETHEUS_HOST")
            config_path = _get_prometheus_config_path(
                use_ray_template=not use_custom_prometheus
            )

            # With no Dashboard, Ray writes no scrape config; enumerate targets here.
            if not args.include_dashboard:
                _write_prometheus_static_config(config_path, env)

            if remote_host:
                region = _extract_amp_region(remote_host)
                remote_write_url = _build_remote_write_url(remote_host, region)
                rw_user = runtime_env.get("RAY_PROMETHEUS_USERNAME")
                rw_pass = runtime_env.get("RAY_PROMETHEUS_PASSWORD")
                basic_auth = (
                    {"username": rw_user, "password": rw_pass}
                    if rw_user and rw_pass
                    else None
                )
                _inject_remote_write_config(
                    remote_write_url,
                    region,
                    config_path=config_path,
                    basic_auth=basic_auth,
                )

            # Add the instance_type and sagemaker_training_job_name labels via
            # scrape-time relabeling (single-node is homogeneous). Must happen
            # before launch; Prometheus reads its config only at startup.
            _inject_sagemaker_relabels(
                config_path,
                _build_ip_instance_type_map(env),
                training_job_name=os.environ.get("TRAINING_JOB_NAME"),
            )

            logger.info("Launching prometheus")
            if use_custom_prometheus:
                prometheus_cmd = _build_prometheus_command(prometheus_folder_name)
                logger.info(
                    "Starting custom Prometheus with command: %s", prometheus_cmd
                )
                prometheus_process = _run_subprocess_command_async(
                    prometheus_cmd,
                    wait_in_seconds=0,
                    stdout_file="/tmp/prometheus_stdout.log",
                    stderr_file="/tmp/prometheus_stderr.log",
                )
            else:
                prometheus_process = _run_subprocess_command_async(
                    "ray metrics launch-prometheus",
                    wait_in_seconds=0,
                    stdout_file="/tmp/prometheus_stdout.log",
                    stderr_file="/tmp/prometheus_stderr.log",
                )

            # Poll for Prometheus readiness
            start_time = time.time()
            prometheus_ready = False

            logger.info(
                "Waiting for Prometheus to become ready (max %s seconds)...",
                PROMETHEUS_WAIT_SECONDS,
            )

            while time.time() - start_time < PROMETHEUS_WAIT_SECONDS:
                if prometheus_process.poll() is not None:
                    logger.warning(
                        "Prometheus process exited with code %s. "
                        "This may indicate missing internet access for downloading the binary. "
                        "Use --prometheus-path to provide a pre-downloaded binary.",
                        prometheus_process.returncode,
                    )
                    break

                try:
                    response = requests.get(
                        f"{runtime_env['RAY_PROMETHEUS_HOST']}/-/healthy",
                        timeout=5,
                    )
                    if response.status_code == 200:
                        elapsed_time = time.time() - start_time
                        logger.info(
                            "Prometheus health check passed after %.1f seconds",
                            elapsed_time,
                        )
                        prometheus_ready = True
                        break
                except Exception as e:
                    logger.warning(str(e))
                    pass

                time.sleep(2)

            if not prometheus_ready:
                logger.warning(
                    "Prometheus did not become ready within %s seconds",
                    PROMETHEUS_WAIT_SECONDS,
                )

            _read_and_log_process_logs("/tmp/prometheus_stderr.log")

        if args.grafana_path:
            # No workers to wait for on a single node, so the readiness poll
            # follows the launch directly.
            grafana_process = _wait_for_grafana_ready(
                _launch_grafana(args, runtime_env), args.grafana_port
            )

        ray_initialized = True
        _run_script(runtime_env)
    except Exception as e:
        has_failure = True
        logger.error("Error in single-node setup or script execution: %s", e)
        raise
    finally:
        if not has_failure:
            _wait_before_shutdown(args.wait_shutdown)

        _shutdown_grafana()

        if args.launch_prometheus:
            logger.info("Shutting down prometheus")
            _run_subprocess_command("ray metrics shutdown-prometheus", check=False)

        _shutdown_ray_safely()
        returncode, _, _ = _run_subprocess_command("ray stop", check=False)
        return returncode


def _setup_multi_node_ray(
    all_hosts: List[str],
    head_host: str,
    runtime_env: Dict[str, Any],
    args: argparse.Namespace,
    env: Any,
) -> int:
    """
    Set up Ray for multi-node execution.

    Args:
        all_hosts: List of all hosts in the cluster
        head_host: Hostname of the head node
        runtime_env: Ray runtime environment configuration
        args: Command line arguments
        env: SageMaker environment variables

    Returns:
        Return code from Ray stop command
    """
    logger.info("Found multiple hosts, initializing Ray as a multi-node cluster")
    logger.info("Head node: %s, Current host: %s", head_host, env.current_host)

    if env.current_host == head_host:
        return _setup_head_node(all_hosts, runtime_env, args, env)
    else:
        return _setup_worker_node(head_host, runtime_env)


def _wait_before_shutdown(wait_shutdown: Optional[int]) -> None:
    """Wait for the specified number of seconds before shutting down Ray.

    Args:
        wait_shutdown: Number of seconds to wait, or None to skip waiting
    """
    if wait_shutdown is not None and wait_shutdown > 0:
        logger.info(
            "Waiting %s seconds before shutting down Ray server...", wait_shutdown
        )
        time.sleep(wait_shutdown)
        logger.info("Wait period completed, proceeding with shutdown")


def _shutdown_ray_safely() -> None:
    """Safely shutdown Ray with proper error handling."""
    global ray_initialized

    if ray_initialized:
        try:
            logger.info("Shutting down Ray...")
            ray.shutdown()
            ray_initialized = False
            logger.info("Ray shutdown completed successfully")
        except Exception as e:
            logger.warning("Error during Ray shutdown: %s", e)


def _setup_ray_environment_homogeneous_cluster(
    args: argparse.Namespace,
    env: Any,
) -> Optional[int]:
    """
    Set up the Ray execution environment for homogeneous distributed workload.

    Args:
        args: Command line arguments
        env: SageMaker environment variables

    Returns:
        Return code from Ray stop command (for multi-node setups) or None
    """
    # Create runtime environment based on instance type
    runtime_env = _create_runtime_environment(args, env)

    # Log environment variables for debugging
    _log_environment_debug_info()

    # Get cluster configuration
    all_hosts, head_host, compute_host_count = _get_cluster_configuration(args, env)

    logger.info(
        "Homogeneous cluster configuration: %s instances, %s contributing compute",
        len(all_hosts),
        compute_host_count,
    )
    logger.info("All hosts: %s", all_hosts)

    # Single-node workload scenario. Decided on the number of INSTANCES in the
    # job (see _get_cluster_configuration on why not compute_host_count).
    if len(all_hosts) == 1:
        return _setup_single_node_ray(args, runtime_env, env)

    # Multi-node workload scenario
    return _setup_multi_node_ray(all_hosts, head_host, runtime_env, args, env)


def _setup_ray_environment_heterogeneous_cluster(
    args: argparse.Namespace,
    env: Any,
) -> Optional[int]:
    """
    Set up the Ray execution environment for heterogeneous distributed workload.

    Args:
        args: Command line arguments
        env: SageMaker environment variables

    Returns:
        Return code from Ray stop command (for multi-node setups) or None
    """
    # Validate required arguments for heterogeneous setup
    if not args.head_instance_group:
        raise ValueError("--head-instance-group is required for heterogeneous clusters")

    # Create runtime environment based on instance type
    runtime_env = _create_runtime_environment(args, env)

    # Log environment variables for debugging
    _log_environment_debug_info()

    # Get cluster configuration
    all_hosts, head_host, compute_host_count = _get_cluster_configuration(args, env)

    logger.info(
        "Heterogeneous cluster configuration: %s instances, %s contributing compute",
        len(all_hosts),
        compute_host_count,
    )
    logger.info("Head instance group: %s", args.head_instance_group)
    logger.info("Head host: %s", head_host)
    logger.info("All hosts: %s", all_hosts)

    # Single-node workload scenario. Decided on the number of INSTANCES in the
    # job, never on compute_host_count: with a coordinator-only head the latter
    # is 1 for a real two-instance cluster, which would send BOTH instances down
    # the single-node path so each starts its own Ray cluster and runs the entry
    # script (see _get_cluster_configuration).
    if len(all_hosts) == 1:
        return _setup_single_node_ray(args, runtime_env, env)

    # Multi-node workload scenario
    return _setup_multi_node_ray(all_hosts, head_host, runtime_env, args, env)


def _write_failure_reason_file(failure_msg: str) -> None:
    """Create a file 'failure' with failure reason if Ray initialization failed.

    Args:
        failure_msg: The content of file to be written.
    """
    if not os.path.exists(FAILURE_REASON_PATH):
        with open(FAILURE_REASON_PATH, "w") as f:
            f.write("RayRuntimeError: " + failure_msg)


def main() -> int:
    """Main entry point for the launcher script.

    Args:
        sys_args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    global has_failure, prometheus_folder_name, grafana_folder_name
    try:
        # Parse only the arguments we care about and ignore the rest
        args, unknown = _parse_args()

        if unknown:
            logger.info("Ignoring unknown arguments: %s", unknown)

        # Get SageMaker environment information
        env = sagemaker_training.environment.Environment()
        logger.info(
            "Found SageMaker environment with hosts: %s", env.instance_groups_dict
        )
        logger.info("Current host: %s", env.current_host)

        # The Prometheus/Grafana archives are extracted lazily by the head (or
        # single) node in _extract_observability_binaries. Doing it here would
        # unpack a few hundred MB on every worker, none of which ever runs them.

        # Set up Ray environment and run the specified script
        if env.is_hetero:
            exit_code = (
                _setup_ray_environment_heterogeneous_cluster(args, env)
                or SUCCESS_EXIT_CODE
            )
        else:
            exit_code = (
                _setup_ray_environment_homogeneous_cluster(args, env)
                or SUCCESS_EXIT_CODE
            )

        # Check if the job failed and raise an exception to ensure SageMaker marks it as failed
        if exit_code == DEFAULT_FAILURE_CODE or has_failure:
            has_failure = True
            failure_reason = "Unknown failure"
            if os.path.exists(FAILURE_REASON_PATH):
                try:
                    with open(FAILURE_REASON_PATH, "r") as f:
                        failure_reason = f.read().strip()
                    logger.error("Job failed with reason: %s", failure_reason)
                except Exception as e:
                    logger.warning("Could not read failure reason file: %s", e)

            # Raise an exception to ensure SageMaker marks the job as failed
            raise RuntimeError("Training job failed: %s" % failure_reason)

        return exit_code

    except Exception as e:
        has_failure = True
        logger.exception("Error encountered while running Ray launcher: %s", e)
        _write_failure_reason_file(str(e))
        raise


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error("Fatal error in launcher: %s", e)
        sys.exit(DEFAULT_FAILURE_CODE)

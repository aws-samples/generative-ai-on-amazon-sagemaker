"""
Utility functions for the SageMaker AI Benchmarking & Inference Recommendations Workshop.

These helpers simplify common operations across all labs:
- Resolving the SageMaker execution role
- Retrieving and storing HuggingFace tokens in Secrets Manager
- Downloading and extracting benchmark output tarballs from S3
- Parsing benchmark results from S3
- Comparing metrics across benchmark runs
- Formatting results as tables
"""

import json
import os
import boto3
import io
import tarfile
import pandas as pd
from typing import Dict, List, Optional
from tabulate import tabulate


def get_execution_role() -> str:
    """
    Resolve the SageMaker execution role ARN.

    Resolution order:
    1. SAGEMAKER_EXECUTION_ROLE environment variable
    2. AWS_SAGEMAKER_ROLE environment variable
    3. Derive from the current caller identity (assumed-role or role ARN)

    Returns:
        The IAM role ARN to use for SageMaker operations.

    Raises:
        ValueError: If the role cannot be determined automatically.
    """
    # 1. Check environment variables (set in SageMaker Studio or manually)
    role = os.environ.get("SAGEMAKER_EXECUTION_ROLE") or os.environ.get("AWS_SAGEMAKER_ROLE")
    if role:
        return role

    # 2. Fallback: derive from current caller identity
    sts = boto3.client("sts")
    identity = sts.get_caller_identity()
    arn = identity["Arn"]
    account = identity["Account"]

    if ":assumed-role/" in arn:
        role_name = arn.split(":assumed-role/")[1].split("/")[0]
        # Look up actual role path — handles service-role/ prefix
        iam = boto3.client("iam")
        try:
            role_info = iam.get_role(RoleName=role_name)
            return role_info["Role"]["Arn"]
        except Exception:
            return f"arn:aws:iam::{account}:role/{role_name}"

    if ":role/" in arn:
        return arn

    raise ValueError(
        "Cannot auto-detect execution role. Set SAGEMAKER_EXECUTION_ROLE env var.\n"
        "Example: export SAGEMAKER_EXECUTION_ROLE=arn:aws:iam::123456789012:role/MyRole"
    )


def parse_benchmark_results(s3_output_location: str, region: str = "us-west-2") -> Dict:
    """
    Download and parse benchmark results from S3.
    
    The benchmark service writes results as output.tar.gz to the S3 output location.
    This function downloads the tarball, extracts it, and parses the JSON results inside.
    
    Args:
        s3_output_location: S3 URI where benchmark results were written. Can be:
            - Direct path to output.tar.gz
            - A prefix (the function will search for output.tar.gz recursively)
        region: AWS region
        
    Returns:
        Dictionary containing all parsed benchmark data (metrics, config, etc.)
    """
    s3_client = boto3.client("s3", region_name=region)
    
    # Parse S3 URI
    s3_path = s3_output_location.replace("s3://", "")
    bucket = s3_path.split("/")[0]
    prefix = "/".join(s3_path.split("/")[1:])
    
    # If the path points directly to a tar.gz, use it
    if prefix.endswith(".tar.gz") or prefix.endswith(".tgz"):
        tar_key = prefix
    else:
        # Search for output.tar.gz under the prefix
        if not prefix.endswith("/"):
            prefix += "/"
        
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        tar_key = None
        
        if "Contents" in response:
            for obj in response["Contents"]:
                key = obj["Key"]
                if key.endswith("output.tar.gz") or key.endswith(".tar.gz"):
                    tar_key = key
                    break
        
        if tar_key is None:
            print(f"⚠️  No .tar.gz found at {s3_output_location}")
            print(f"   Objects found: {[o['Key'] for o in response.get('Contents', [])]}")
            return {}
    
    print(f"📥 Downloading: s3://{bucket}/{tar_key}")
    
    # Download the tarball into memory
    obj_response = s3_client.get_object(Bucket=bucket, Key=tar_key)
    tar_bytes = obj_response["Body"].read()
    
    # Extract and parse all JSON files from the tarball
    results = {}
    all_files = []
    
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            all_files.append(member.name)
            if member.name.endswith(".json") and member.isfile():
                f = tar.extractfile(member)
                if f:
                    content = f.read().decode("utf-8")
                    try:
                        parsed = json.loads(content)
                        # Use the filename (without extension) as key, or merge if single file
                        name = member.name.rsplit("/", 1)[-1].replace(".json", "")
                        results[name] = parsed
                    except json.JSONDecodeError:
                        print(f"   ⚠️  Could not parse {member.name}")
    
    print(f"📦 Extracted {len(all_files)} files from tarball:")
    for f in sorted(all_files):
        print(f"      {f}")
    print(f"📊 Parsed {len(results)} JSON result file(s)")
    
    # If there's only one JSON result, return it directly for convenience
    if len(results) == 1:
        return list(results.values())[0]
    
    # If profile_export_aiperf exists, return it as the primary result
    # (this is the main metrics file from AIPerf benchmarking)
    if "profile_export_aiperf" in results:
        return results["profile_export_aiperf"]
    
    return results


def extract_metrics(results: Dict) -> Dict:
    """
    Extract key metrics from raw benchmark results into a flat dictionary.
    
    Handles the AIPerf output format where each metric is a top-level key
    containing a dict with 'unit', 'avg', 'p50', 'p90', 'p99', etc.
    
    Args:
        results: Raw benchmark results dictionary from parse_benchmark_results()
        
    Returns:
        Dictionary with key metric names and values
    """
    metrics = {}
    
    # AIPerf output schema: top-level keys are metric categories,
    # each containing {unit, avg, p50, p90, p99, min, max, std, count, sum}
    metric_categories = {
        "time_to_first_token": "ttft",
        "inter_token_latency": "itl",
        "time_to_second_token": "ttst",
        "request_latency": "request_latency",
        "output_token_throughput": "output_throughput",
        "output_token_throughput_per_user": "output_throughput_per_user",
        "request_throughput": "request_throughput",
        "output_sequence_length": "output_seq_len",
        "request_count": "request_count",
    }
    
    stats_to_extract = ["avg", "p50", "p90", "p99", "min", "max"]
    
    for raw_key, short_name in metric_categories.items():
        if raw_key in results and isinstance(results[raw_key], dict):
            category_data = results[raw_key]
            unit = category_data.get("unit", "")
            
            for stat in stats_to_extract:
                if stat in category_data:
                    metric_key = f"{short_name}_{stat}"
                    metrics[metric_key] = category_data[stat]
            
            # Store unit for reference
            metrics[f"{short_name}_unit"] = unit
    
    # Also extract schema metadata
    if "aiperf_version" in results:
        metrics["_aiperf_version"] = results["aiperf_version"]
    if "benchmark_id" in results:
        metrics["_benchmark_id"] = results["benchmark_id"]
    
    return metrics


def format_metrics_table(metrics: Dict, title: str = "Benchmark Results") -> str:
    """
    Format benchmark metrics as a readable table.
    
    Args:
        metrics: Dictionary of metric names to values
        title: Table title
        
    Returns:
        Formatted string table
    """
    # Group metrics by category based on the new key naming
    categories = {
        "⏱️  Time to First Token (TTFT)": [k for k in metrics if k.startswith("ttft_") and not k.endswith("_unit")],
        "⚡ Inter-Token Latency (ITL)": [k for k in metrics if k.startswith("itl_") and not k.endswith("_unit")],
        "📨 Request Latency": [k for k in metrics if k.startswith("request_latency_") and not k.endswith("_unit")],
        "🚀 Throughput": [k for k in metrics if ("throughput" in k or "request_throughput" in k) and not k.endswith("_unit") and not k.startswith("request_latency")],
        "📏 Sequence Length": [k for k in metrics if "seq_len" in k and not k.endswith("_unit")],
    }
    
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")
    
    for category, keys in categories.items():
        if not keys:
            continue
        rows = []
        for key in sorted(keys):
            if key in metrics:
                value = metrics[key]
                if isinstance(value, (int, float)):
                    # Determine unit from the metric name context
                    display_name = key.replace("_", " ").replace("ttft", "TTFT").replace("itl", "ITL")
                    
                    # Look up the unit
                    prefix = key.rsplit("_", 1)[0]
                    unit = metrics.get(f"{prefix}_unit", "")
                    
                    if "ms" in unit or "latency" in key or "ttft" in key or "itl" in key or "ttst" in key:
                        rows.append([display_name, f"{value:.2f} ms"])
                    elif "tokens/sec" in unit or "throughput" in key:
                        rows.append([display_name, f"{value:.2f} tokens/s"])
                    elif "requests/sec" in unit:
                        rows.append([display_name, f"{value:.2f} req/s"])
                    elif "tokens" in unit:
                        rows.append([display_name, f"{value:.1f} tokens"])
                    else:
                        rows.append([display_name, f"{value:.2f}"])
        
        if rows:
            print(f"  {category}")
            print(f"  {'-'*40}")
            print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))
            print()
    
    return ""


def compare_results(
    results_list: List[Dict],
    labels: List[str],
    metrics_to_compare: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare metrics across multiple benchmark runs.
    
    Args:
        results_list: List of metric dictionaries (output of extract_metrics())
        labels: List of labels for each run (e.g., ["Baseline", "Optimized"])
        metrics_to_compare: Optional list of specific metrics to include. 
                           If None, includes all common metrics.
    
    Returns:
        DataFrame with runs as columns and metrics as rows
    """
    if metrics_to_compare is None:
        # Find metrics common to all runs
        if results_list:
            common_keys = set(results_list[0].keys())
            for r in results_list[1:]:
                common_keys &= set(r.keys())
            metrics_to_compare = sorted(common_keys)
        else:
            metrics_to_compare = []
    
    data = {}
    for label, metrics in zip(labels, results_list):
        data[label] = {m: metrics.get(m, None) for m in metrics_to_compare}
    
    df = pd.DataFrame(data, index=metrics_to_compare)
    return df


def print_comparison_table(
    results_list: List[Dict],
    labels: List[str],
    title: str = "Benchmark Comparison"
):
    """
    Print a formatted comparison table across multiple benchmark runs.
    
    Args:
        results_list: List of metric dictionaries
        labels: Labels for each run
        title: Table title
    """
    df = compare_results(results_list, labels)
    
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")
    
    # Calculate delta if exactly 2 runs
    if len(results_list) == 2:
        df["Delta"] = ""
        df["Improvement"] = ""
        for idx in df.index:
            baseline = df.iloc[:, 0][idx]
            optimized = df.iloc[:, 1][idx]
            if baseline and optimized and isinstance(baseline, (int, float)) and isinstance(optimized, (int, float)):
                delta = optimized - baseline
                pct = ((optimized - baseline) / baseline) * 100
                # For latency metrics, negative delta = improvement
                if "latency" in idx or "ttft" in idx or "itl" in idx:
                    improvement = "✅" if delta < 0 else "⚠️"
                else:
                    improvement = "✅" if delta > 0 else "⚠️"
                df.at[idx, "Delta"] = f"{delta:+.2f}"
                df.at[idx, "Improvement"] = f"{pct:+.1f}% {improvement}"
    
    print(tabulate(df, headers="keys", tablefmt="grid", floatfmt=".2f"))
    print()


def wait_for_endpoint(endpoint_name: str, region: str = "us-west-2", timeout_minutes: int = 30):
    """
    Wait for a SageMaker endpoint to reach InService status.
    
    Args:
        endpoint_name: Name of the endpoint
        region: AWS region
        timeout_minutes: Maximum time to wait
    """
    import time
    
    sm_client = boto3.client("sagemaker", region_name=region)
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    print(f"Waiting for endpoint '{endpoint_name}' to be InService...")
    
    while True:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        elapsed = time.time() - start_time
        
        print(f"  Status: {status} ({elapsed/60:.1f} min elapsed)")
        
        if status == "InService":
            print(f"✅ Endpoint is InService! (took {elapsed/60:.1f} minutes)")
            return response
        elif status == "Failed":
            reason = response.get("FailureReason", "Unknown")
            raise RuntimeError(f"❌ Endpoint creation failed: {reason}")
        elif elapsed > timeout_seconds:
            raise TimeoutError(f"⏰ Endpoint did not reach InService within {timeout_minutes} minutes")
        
        time.sleep(30)


def wait_for_benchmark_job(
    job_name: str, 
    region: str = "us-west-2", 
    timeout_minutes: int = 30
) -> Dict:
    """
    Wait for an AI Benchmark Job to complete.
    
    Args:
        job_name: Name of the benchmark job
        region: AWS region
        timeout_minutes: Maximum time to wait
        
    Returns:
        Final describe response
    """
    import time
    
    sm_client = boto3.client("sagemaker", region_name=region)
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    print(f"Waiting for benchmark job '{job_name}' to complete...")
    
    while True:
        response = sm_client.describe_ai_benchmark_job(
            AIBenchmarkJobName=job_name
        )
        status = response["AIBenchmarkJobStatus"]
        elapsed = time.time() - start_time
        
        print(f"  Status: {status} ({elapsed/60:.1f} min elapsed)")
        
        if status == "Completed":
            print(f"✅ Benchmark job completed! (took {elapsed/60:.1f} minutes)")
            return response
        elif status == "Failed":
            reason = response.get("FailureReason", "Unknown")
            raise RuntimeError(f"❌ Benchmark job failed: {reason}")
        elif status == "Stopped":
            raise RuntimeError("⛔ Benchmark job was stopped")
        elif elapsed > timeout_seconds:
            raise TimeoutError(f"⏰ Benchmark job did not complete within {timeout_minutes} minutes")
        
        time.sleep(30)


def wait_for_recommendation_job(
    job_name: str, 
    region: str = "us-west-2", 
    timeout_minutes: int = 120
) -> Dict:
    """
    Wait for an AI Recommendation Job to complete.
    
    Note: Recommendation jobs take longer than benchmark jobs because they
    include model analysis, optimization, and multi-configuration benchmarking.
    
    Args:
        job_name: Name of the recommendation job
        region: AWS region
        timeout_minutes: Maximum time to wait (default 120 min for recommendations)
        
    Returns:
        Final describe response
    """
    import time
    
    sm_client = boto3.client("sagemaker", region_name=region)
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    print(f"Waiting for recommendation job '{job_name}' to complete...")
    print(f"  (Recommendation jobs typically take 30-90 minutes)")
    
    while True:
        response = sm_client.describe_ai_recommendation_job(
            AIRecommendationJobName=job_name
        )
        status = response["AIRecommendationJobStatus"]
        elapsed = time.time() - start_time
        
        print(f"  Status: {status} ({elapsed/60:.1f} min elapsed)")
        
        if status == "Completed":
            print(f"✅ Recommendation job completed! (took {elapsed/60:.1f} minutes)")
            return response
        elif status == "Failed":
            reason = response.get("FailureReason", "Unknown")
            raise RuntimeError(f"❌ Recommendation job failed: {reason}")
        elif status == "Stopped":
            raise RuntimeError("⛔ Recommendation job was stopped")
        elif elapsed > timeout_seconds:
            raise TimeoutError(f"⏰ Recommendation job did not complete within {timeout_minutes} minutes")
        
        time.sleep(60)  # Longer polling interval for recommendations


def format_recommendation(recommendation: Dict, index: int = 0) -> str:
    """
    Format a single recommendation from a recommendation job into readable output.
    
    Args:
        recommendation: A single recommendation dict from the Recommendations array
        index: Rank/index of this recommendation
        
    Returns:
        Formatted string
    """
    deploy_config = recommendation.get("DeploymentConfiguration", {})
    performance = recommendation.get("ExpectedPerformance", {})
    optimization = recommendation.get("OptimizationDetails", {})
    
    output = []
    output.append(f"\n{'─'*60}")
    output.append(f"  Recommendation #{index + 1}")
    output.append(f"{'─'*60}")
    
    # Deployment config
    output.append(f"\n  📦 Deployment Configuration:")
    output.append(f"     Instance Type:     {deploy_config.get('InstanceType', 'N/A')}")
    output.append(f"     Instance Count:    {deploy_config.get('InstanceCount', 1)}")
    output.append(f"     Copies/Instance:   {deploy_config.get('CopyCountPerInstance', 1)}")
    output.append(f"     Container Image:   {deploy_config.get('ImageUri', 'N/A')[:80]}...")
    
    # Environment variables (select key ones)
    env_vars = deploy_config.get("EnvironmentVariables", {})
    if env_vars:
        output.append(f"\n  🔧 Key Environment Variables:")
        for key in ["TENSOR_PARALLEL_DEGREE", "MAX_MODEL_LEN", "SM_NUM_GPUS"]:
            if key in env_vars:
                output.append(f"     {key}: {env_vars[key]}")
    
    # Performance metrics
    if performance:
        output.append(f"\n  📊 Expected Performance:")
        for metric in performance:
            name = metric.get("Metric", metric.get("Name", "unknown"))
            value = metric.get("Value", "N/A")
            stat = metric.get("Stat", "")
            unit = metric.get("Unit", "")
            output.append(f"     {name} ({stat}): {value} {unit}")
    
    # Optimization details
    if optimization:
        output.append(f"\n  ⚡ Optimizations Applied:")
        if isinstance(optimization, dict):
            for technique, details in optimization.items():
                output.append(f"     • {technique}: {details}")
        elif isinstance(optimization, list):
            for opt in optimization:
                output.append(f"     • {opt}")
    
    return "\n".join(output)


def cleanup_resources(
    endpoint_names: List[str] = None,
    model_names: List[str] = None,
    workload_config_names: List[str] = None,
    region: str = "us-west-2"
):
    """
    Clean up SageMaker resources created during the workshop.
    
    Args:
        endpoint_names: Endpoints to delete (also deletes associated endpoint configs)
        model_names: Models to delete
        workload_config_names: AI Workload Configs to delete
        region: AWS region
    """
    sm_client = boto3.client("sagemaker", region_name=region)
    
    # Delete endpoints (and their configs)
    if endpoint_names:
        for name in endpoint_names:
            try:
                sm_client.delete_endpoint(EndpointName=name)
                print(f"  ✅ Deleted endpoint: {name}")
            except sm_client.exceptions.ClientError as e:
                print(f"  ⚠️  Could not delete endpoint '{name}': {e}")
            
            try:
                sm_client.delete_endpoint_config(EndpointConfigName=name)
                print(f"  ✅ Deleted endpoint config: {name}")
            except sm_client.exceptions.ClientError as e:
                print(f"  ⚠️  Could not delete endpoint config '{name}': {e}")
    
    # Delete models
    if model_names:
        for name in model_names:
            try:
                sm_client.delete_model(ModelName=name)
                print(f"  ✅ Deleted model: {name}")
            except sm_client.exceptions.ClientError as e:
                print(f"  ⚠️  Could not delete model '{name}': {e}")
    
    # Delete workload configs
    if workload_config_names:
        for name in workload_config_names:
            try:
                sm_client.delete_ai_workload_config(AIWorkloadConfigName=name)
                print(f"  ✅ Deleted workload config: {name}")
            except sm_client.exceptions.ClientError as e:
                print(f"  ⚠️  Could not delete workload config '{name}': {e}")
    
    print("\n🧹 Cleanup complete!")

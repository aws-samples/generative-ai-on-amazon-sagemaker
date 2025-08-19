from opentel_utils import QueueSpanExporter
from flotorch_eval.agent_eval.core.converter import TraceConverter
import pandas as pd
from IPython.display import display, HTML
from flotorch_eval.agent_eval.metrics.base import BaseMetric
from flotorch_eval.agent_eval.core.evaluator import Evaluator
from flotorch_eval.agent_eval.metrics.base import MetricResult
from typing import List, Any, Dict
import textwrap
import json
from pathlib import Path
from flotorch_eval.agent_eval.core.schemas import Trajectory

def get_all_spans(exporter: QueueSpanExporter)->list:
    """Extracts all spans from the queue exporter."""
    spans = []
    while not exporter.spans.empty():
        spans.append(exporter.spans.get())
    return spans

def create_trajectory(spans:list):
    """Converts a list of spans into a structured Trajectory object."""
    converter = TraceConverter()
    trajectory = converter.from_spans(spans)
    return trajectory

def _format_latency_breakdown_recursive(items: List[Dict], level: int) -> List[str]:
    """A helper function to recursively format the nested latency breakdown."""
    lines = []
    
    indent = "&nbsp;&nbsp;&nbsp;&nbsp;" * level
    for item in items:
        step_name = item.get('step_name', 'Unknown Step')
        latency = item.get('latency_ms', 'N/A')
        lines.append(f"{indent}- {step_name}: {latency} ms")
        
        # If there are children, recurse
        if item.get('children'):
            lines.extend(_format_latency_breakdown_recursive(item['children'], level + 1))
    return lines


def display_evaluation_results(results: Any):
    """
    Displays evaluation results as a clean, styled HTML table,
    correctly rendering newlines and hierarchical indentation in the details column.
    """
    if not results or not getattr(results, 'scores', None):
        print("No evaluation results were generated.")
        return

    data = []
    for metric in results.scores:
        details_dict = metric.details.copy() if metric.details else {}
        display_parts = []

        if 'error' in details_dict:
            error_message = f"Error: {details_dict.pop('error')}"
            display_parts.append(textwrap.fill(error_message, width=80))

        elif 'comment' in details_dict:
            comment = details_dict.pop('comment')
            display_parts.append(textwrap.fill(comment, width=80))

        elif 'total_latency_ms' in details_dict and 'latency_breakdown' in details_dict:
            latency_summary = []
            
            breakdown_data = details_dict.pop('latency_breakdown', [])
            total_latency = details_dict.pop('total_latency_ms')
            avg_latency = details_dict.pop('average_step_latency_ms', None)
            
            latency_summary.append(f"Total Latency (Root Steps): {total_latency} ms")
            if avg_latency is not None:
                latency_summary.append(f"Average Root Step Latency: {avg_latency} ms")
            
            if breakdown_data:
                latency_summary.append("Latency Breakdown:")
                # Call the recursive helper function (which now uses &nbsp;)
                formatted_lines = _format_latency_breakdown_recursive(breakdown_data, level=1)
                latency_summary.extend(formatted_lines)
            
            display_parts.append("\n".join(latency_summary))
        
        other_details = []
        if details_dict:
            if 'cost_breakdown' in details_dict and isinstance(details_dict['cost_breakdown'], list):
                details_dict['cost_breakdown'] = "\n" + "\n".join([f"    - {item}" for item in details_dict['cost_breakdown']])
            for key, value in details_dict.items():
                other_details.append(f"- {key}: {value}")

        if other_details:
            if display_parts:
                display_parts.append("\n" + "-"*20)
            display_parts.append("\n".join(other_details))
        
        score_display = f"{metric.score:.2f}" if isinstance(metric.score, (int, float)) else "N/A"

        data.append({
            "Metric": metric.name,
            "Score": score_display,
            "Details": "\n".join(display_parts)
        })

    df = pd.DataFrame(data)
    pd.set_option('display.max_colwidth', None)

    styled_df = df.style.set_properties(
        subset=['Details'],
        **{'white-space': 'pre-wrap', 'text-align': 'left'}
    ).set_table_styles(
        [dict(selector="th", props=[("text-align", "left")])]
    ).hide(axis="index")

    display(styled_df)
    
def initialize_evaluator(metrics: List[BaseMetric]) -> Evaluator:
    """Initializes the Evaluator with a given list of metric objects."""
    return Evaluator(metrics=metrics)


def save_trajectory_to_json(trajectory: Trajectory, output_path: str | Path) -> None:
    """
    Save a Trajectory object as a pretty-printed JSON file.
    
    Args:
        trajectory: The Trajectory object to save
        output_path: Path where the JSON file should be saved
    """
    # Convert trajectory to dict and then to JSON with pretty printing
    json_str = trajectory.model_dump_json(indent=2)
    
    # Ensure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(json_str) 
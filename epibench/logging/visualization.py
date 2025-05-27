"""
EpiBench Logging Visualization Foundations

Provides basic utilities for transforming and displaying log data in a structured, text-based format.
Designed for CLI output and as a foundation for future graphical/interactive visualizations (Task 39).
"""

from typing import List, Dict, Any, Optional
import json

class LogDataAdapter:
    """
    Adapter to transform raw log data into flat, display-friendly structures.
    Designed for extension with richer visualization backends.
    """
    def __init__(self, logs: List[Dict[str, Any]]):
        self.logs = logs

    def to_flat_table(self, fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Flattens log data for tabular display.
        Args:
            fields: List of fields to include (default: common fields)
        Returns:
            List of dicts, each representing a row.
        """
        default_fields = [
            "execution_id", "sample_id", "status", "start_time", "duration", "r_squared", "mse"
        ]
        fields = fields or default_fields
        table = []
        for log in self.logs:
            row = {}
            exec_meta = log.get("execution_metadata", {})
            perf = log.get("performance_metrics", {})
            row["execution_id"] = exec_meta.get("execution_id", "N/A")
            row["sample_id"] = log.get("input_configuration", {}).get("sample_id", "N/A")
            row["status"] = exec_meta.get("execution_status", "unknown")
            row["start_time"] = exec_meta.get("timestamp_start", "N/A")
            row["duration"] = log.get("runtime_information", {}).get("duration_seconds", 0)
            row["r_squared"] = perf.get("r_squared", None)
            row["mse"] = perf.get("mse", None)
            # Add more fields as needed
            table.append({k: row.get(k, "") for k in fields})
        return table

    def extract_key_metrics(self) -> List[Dict[str, Any]]:
        """
        Extracts key metrics for summary display.
        Returns:
            List of dicts with execution_id, sample_id, r_squared, mse, status
        """
        return [
            {
                "execution_id": log.get("execution_metadata", {}).get("execution_id", "N/A"),
                "sample_id": log.get("input_configuration", {}).get("sample_id", "N/A"),
                "r_squared": log.get("performance_metrics", {}).get("r_squared", None),
                "mse": log.get("performance_metrics", {}).get("mse", None),
                "status": log.get("execution_metadata", {}).get("execution_status", "unknown"),
            }
            for log in self.logs
        ]

# --- Simple Text-Based Visualization Utilities ---
def ascii_table(rows: List[Dict[str, Any]], headers: Optional[List[str]] = None) -> str:
    """
    Render a simple ASCII table from a list of dicts.
    Args:
        rows: List of dicts (each row)
        headers: List of column headers (optional)
    Returns:
        String containing the ASCII table
    """
    if not rows:
        return "(No data)"
    headers = headers or list(rows[0].keys())
    col_widths = {h: max(len(str(h)), max(len(str(row.get(h, ""))) for row in rows)) for h in headers}
    sep = "+" + "+".join("-" * (col_widths[h] + 2) for h in headers) + "+"
    header_row = "| " + " | ".join(f"{h:<{col_widths[h]}}" for h in headers) + " |"
    lines = [sep, header_row, sep]
    for row in rows:
        line = "| " + " | ".join(f"{str(row.get(h, '')):<{col_widths[h]}}" for h in headers) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def log_summary(log: Dict[str, Any]) -> str:
    """
    Render a simple summary of a single log.
    Args:
        log: Log dict
    Returns:
        String summary
    """
    exec_meta = log.get("execution_metadata", {})
    perf = log.get("performance_metrics", {})
    lines = [
        f"Execution ID: {exec_meta.get('execution_id', 'N/A')}",
        f"Sample ID: {log.get('input_configuration', {}).get('sample_id', 'N/A')}",
        f"Status: {exec_meta.get('execution_status', 'unknown')}",
        f"Start Time: {exec_meta.get('timestamp_start', 'N/A')}",
        f"Duration: {log.get('runtime_information', {}).get('duration_seconds', 0)} seconds",
        f"R²: {perf.get('r_squared', '-')}",
        f"MSE: {perf.get('mse', '-')}",
    ]
    return "\n".join(lines)

# --- Extension Notes ---
"""
This module is designed for extension in Task 39:
- Add graphical/interactive visualizations (matplotlib, seaborn, plotly, etc.)
- Integrate with CLI and web dashboards
- Support advanced data adapters for grouping, filtering, and aggregation
- Add export to HTML, PDF, or image formats
""" 
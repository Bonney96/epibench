"""Log query and analysis API for EpiBench logging system."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
import re
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv

from .log_manager import LogManager
from .metrics_utils import compare_metrics, aggregate_sample_metrics

logger = logging.getLogger(__name__)


class LogQuery:
    """Query and analyze EpiBench logs with filtering, aggregation, and export capabilities."""
    
    def __init__(self, log_directory: Union[str, Path], cache_size: int = 100):
        """
        Initialize LogQuery with a directory containing log files.
        
        Args:
            log_directory: Path to directory containing log JSON files
            cache_size: Number of queries to cache for performance
        """
        self.log_directory = Path(log_directory)
        if not self.log_directory.exists():
            raise ValueError(f"Log directory does not exist: {log_directory}")
        
        self._logs_cache = {}
        self._query_cache = {}
        self._cache_size = cache_size
        self._filters = []
        self._sort_by = None
        self._sort_order = 'asc'
        self._limit = None
        self._offset = 0
        
        # Load all logs lazily
        self._log_files = list(self.log_directory.glob("*.json"))
        logger.info(f"Initialized LogQuery with {len(self._log_files)} log files")
    
    def _load_log(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single log file with caching."""
        if file_path in self._logs_cache:
            return self._logs_cache[file_path]
        
        try:
            with open(file_path, 'r') as f:
                log_data = json.load(f)
            
            # Add file path for reference
            log_data['_file_path'] = str(file_path)
            
            # Cache the loaded log
            if len(self._logs_cache) < self._cache_size:
                self._logs_cache[file_path] = log_data
            
            return log_data
        except Exception as e:
            logger.error(f"Failed to load log file {file_path}: {e}")
            return None
    
    def filter_by_date_range(self, start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None) -> 'LogQuery':
        """
        Filter logs by date range.
        
        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
        
        Returns:
            Self for method chaining
        """
        def date_filter(log: Dict[str, Any]) -> bool:
            try:
                # Parse log timestamp
                log_start = log.get('execution_metadata', {}).get('timestamp', {}).get('start')
                if not log_start:
                    return False
                
                log_date = datetime.fromisoformat(log_start.replace('Z', '+00:00'))
                
                if start_date and log_date < start_date:
                    return False
                if end_date and log_date > end_date:
                    return False
                return True
            except Exception:
                return False
        
        self._filters.append(date_filter)
        return self
    
    def filter_by_sample_id(self, sample_ids: Union[str, List[str]]) -> 'LogQuery':
        """
        Filter logs by sample ID(s).
        
        Args:
            sample_ids: Single sample ID or list of sample IDs
        
        Returns:
            Self for method chaining
        """
        if isinstance(sample_ids, str):
            sample_ids = [sample_ids]
        
        sample_id_set = set(sample_ids)
        
        def sample_filter(log: Dict[str, Any]) -> bool:
            log_sample_id = log.get('input_configuration', {}).get('sample_id')
            return log_sample_id in sample_id_set
        
        self._filters.append(sample_filter)
        return self
    
    def filter_by_metric(self, metric_name: str, condition: str, threshold: float) -> 'LogQuery':
        """
        Filter logs by performance metric threshold.
        
        Args:
            metric_name: Name of the metric (e.g., 'mse', 'r_squared')
            condition: Comparison operator ('>', '<', '>=', '<=', '==', '!=')
            threshold: Threshold value
        
        Returns:
            Self for method chaining
        """
        operators = {
            '>': lambda x, y: x > y,
            '<': lambda x, y: x < y,
            '>=': lambda x, y: x >= y,
            '<=': lambda x, y: x <= y,
            '==': lambda x, y: x == y,
            '!=': lambda x, y: x != y
        }
        
        if condition not in operators:
            raise ValueError(f"Invalid condition: {condition}")
        
        op = operators[condition]
        
        def metric_filter(log: Dict[str, Any]) -> bool:
            metrics = log.get('performance_metrics', {})
            
            # Check main metrics
            if metric_name in metrics:
                value = metrics[metric_name]
                if value is not None:
                    return op(value, threshold)
            
            # Check correlations
            if metric_name in ['pearson', 'spearman']:
                corr_value = metrics.get('correlations', {}).get(metric_name, {}).get('r')
                if corr_value is not None:
                    return op(corr_value, threshold)
            
            # Check sample metrics
            sample_metrics = metrics.get('sample_metrics', {})
            for sample_data in sample_metrics.values():
                if metric_name in sample_data:
                    value = sample_data[metric_name]
                    if value is not None and op(value, threshold):
                        return True
            
            return False
        
        self._filters.append(metric_filter)
        return self
    
    def filter_by_status(self, status: Union[str, List[str]]) -> 'LogQuery':
        """
        Filter logs by execution status.
        
        Args:
            status: Status or list of statuses to filter by
        
        Returns:
            Self for method chaining
        """
        if isinstance(status, str):
            status = [status]
        
        status_set = set(status)
        
        def status_filter(log: Dict[str, Any]) -> bool:
            log_status = log.get('execution_metadata', {}).get('status')
            return log_status in status_set
        
        self._filters.append(status_filter)
        return self
    
    def filter_by_config(self, config_path: str, value: Any) -> 'LogQuery':
        """
        Filter logs by configuration parameter.
        
        Args:
            config_path: Dot-separated path to config parameter (e.g., 'model.batch_size')
            value: Value to match (can be regex pattern for strings)
        
        Returns:
            Self for method chaining
        """
        def get_nested_value(data: Dict[str, Any], path: str) -> Any:
            """Get value from nested dictionary using dot notation."""
            keys = path.split('.')
            current = data
            
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
            return current
        
        def config_filter(log: Dict[str, Any]) -> bool:
            # Check in effective config
            effective_config = log.get('configuration_parameters', {}).get('effective_config', {})
            config_value = get_nested_value(effective_config, config_path)
            
            if config_value is not None:
                if isinstance(value, str) and isinstance(config_value, str):
                    # Try regex matching for strings
                    try:
                        return bool(re.match(value, config_value))
                    except re.error:
                        return config_value == value
                else:
                    return config_value == value
            
            # Check in temp configs
            temp_configs = log.get('configuration_parameters', {}).get('temp_configs', {})
            temp_value = get_nested_value(temp_configs, config_path)
            
            if temp_value is not None:
                if isinstance(value, str) and isinstance(temp_value, str):
                    try:
                        return bool(re.match(value, temp_value))
                    except re.error:
                        return temp_value == value
                else:
                    return temp_value == value
            
            return False
        
        self._filters.append(config_filter)
        return self
    
    def sort_by(self, field: str, order: str = 'asc') -> 'LogQuery':
        """
        Sort results by a field.
        
        Args:
            field: Field to sort by (supports dot notation)
            order: Sort order ('asc' or 'desc')
        
        Returns:
            Self for method chaining
        """
        self._sort_by = field
        self._sort_order = order
        return self
    
    def limit(self, limit: int) -> 'LogQuery':
        """
        Limit number of results.
        
        Args:
            limit: Maximum number of results to return
        
        Returns:
            Self for method chaining
        """
        self._limit = limit
        return self
    
    def offset(self, offset: int) -> 'LogQuery':
        """
        Set result offset for pagination.
        
        Args:
            offset: Number of results to skip
        
        Returns:
            Self for method chaining
        """
        self._offset = offset
        return self
    
    def execute(self) -> List[Dict[str, Any]]:
        """
        Execute the query and return matching logs.
        
        Returns:
            List of matching log entries
        """
        # Create cache key from current query state
        cache_key = (
            tuple(self._filters),
            self._sort_by,
            self._sort_order,
            self._limit,
            self._offset
        )
        
        # Check cache
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        # Load and filter logs
        results = []
        
        # Use parallel loading for better performance
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(self._load_log, f): f for f in self._log_files}
            
            for future in as_completed(future_to_file):
                log = future.result()
                if log is None:
                    continue
                
                # Apply all filters
                if all(f(log) for f in self._filters):
                    results.append(log)
        
        # Sort if requested
        if self._sort_by:
            def get_sort_value(log: Dict[str, Any]) -> Any:
                keys = self._sort_by.split('.')
                current = log
                for key in keys:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        return None
                return current
            
            results.sort(
                key=get_sort_value,
                reverse=(self._sort_order == 'desc')
            )
        
        # Apply pagination
        if self._offset:
            results = results[self._offset:]
        if self._limit:
            results = results[:self._limit]
        
        # Cache results
        if len(self._query_cache) < self._cache_size:
            self._query_cache[cache_key] = results
        
        # Reset query state for next query
        self._reset()
        
        return results
    
    def _reset(self):
        """Reset query state."""
        self._filters = []
        self._sort_by = None
        self._sort_order = 'asc'
        self._limit = None
        self._offset = 0
    
    def count(self) -> int:
        """
        Count matching logs without loading all data.
        
        Returns:
            Number of matching logs
        """
        # Execute query without pagination
        original_limit = self._limit
        original_offset = self._offset
        self._limit = None
        self._offset = 0
        
        results = self.execute()
        count = len(results)
        
        # Restore pagination settings
        self._limit = original_limit
        self._offset = original_offset
        
        return count
    
    @staticmethod
    def compare_logs(log1: Dict[str, Any], log2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two logs and identify differences.
        
        Args:
            log1: First log
            log2: Second log
        
        Returns:
            Dictionary containing comparison results
        """
        comparison = {
            'config_differences': {},
            'metric_comparison': {},
            'performance_delta': {},
            'execution_time_diff': None
        }
        
        # Compare configurations
        config1 = log1.get('configuration_parameters', {}).get('effective_config', {})
        config2 = log2.get('configuration_parameters', {}).get('effective_config', {})
        
        comparison['config_differences'] = _diff_nested_dicts(config1, config2)
        
        # Compare metrics
        metrics1 = log1.get('performance_metrics', {})
        metrics2 = log2.get('performance_metrics', {})
        
        if metrics1 and metrics2:
            comparison['metric_comparison'] = compare_metrics(metrics1, metrics2)
        
        # Calculate performance delta
        for metric in ['mse', 'r_squared', 'mae']:
            if metric in metrics1 and metric in metrics2:
                val1 = metrics1[metric]
                val2 = metrics2[metric]
                if val1 is not None and val2 is not None:
                    comparison['performance_delta'][metric] = val2 - val1
        
        # Compare execution times
        duration1 = log1.get('runtime_information', {}).get('duration_seconds')
        duration2 = log2.get('runtime_information', {}).get('duration_seconds')
        
        if duration1 is not None and duration2 is not None:
            comparison['execution_time_diff'] = duration2 - duration1
        
        return comparison
    
    def export_to_csv(self, logs: List[Dict[str, Any]], output_path: Union[str, Path],
                     fields: Optional[List[str]] = None) -> None:
        """
        Export logs to CSV format.
        
        Args:
            logs: List of logs to export
            output_path: Path for output CSV file
            fields: List of fields to export (uses all if None)
        """
        if not logs:
            logger.warning("No logs to export")
            return
        
        output_path = Path(output_path)
        
        # Flatten logs for CSV export
        flattened_logs = []
        for log in logs:
            flat_log = _flatten_dict(log)
            flattened_logs.append(flat_log)
        
        # Determine fields to export
        if fields is None:
            # Get all unique fields
            all_fields = set()
            for log in flattened_logs:
                all_fields.update(log.keys())
            fields = sorted(all_fields)
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            
            for log in flattened_logs:
                # Fill missing fields with empty string
                row = {field: log.get(field, '') for field in fields}
                writer.writerow(row)
        
        logger.info(f"Exported {len(logs)} logs to {output_path}")
    
    def export_to_json(self, logs: List[Dict[str, Any]], output_path: Union[str, Path],
                      pretty: bool = True) -> None:
        """
        Export logs to JSON format.
        
        Args:
            logs: List of logs to export
            output_path: Path for output JSON file
            pretty: Whether to format JSON for readability
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            if pretty:
                json.dump(logs, f, indent=2)
            else:
                json.dump(logs, f)
        
        logger.info(f"Exported {len(logs)} logs to {output_path}")
    
    def export_to_excel(self, logs: List[Dict[str, Any]], output_path: Union[str, Path],
                       include_summary: bool = True) -> None:
        """
        Export logs to Excel format with multiple sheets.
        
        Args:
            logs: List of logs to export
            output_path: Path for output Excel file
            include_summary: Whether to include summary statistics sheet
        """
        try:
            import openpyxl
        except ImportError:
            logger.error("openpyxl not installed. Install with: pip install openpyxl")
            return
        
        output_path = Path(output_path)
        
        # Create main data frame
        flattened_logs = [_flatten_dict(log) for log in logs]
        df_main = pd.DataFrame(flattened_logs)
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write main data
            df_main.to_excel(writer, sheet_name='Logs', index=False)
            
            if include_summary and logs:
                # Create metrics summary
                metrics_data = []
                for log in logs:
                    metrics = log.get('performance_metrics', {})
                    if metrics:
                        metric_row = {
                            'execution_id': log.get('execution_metadata', {}).get('execution_id'),
                            'sample_id': log.get('input_configuration', {}).get('sample_id'),
                            'mse': metrics.get('mse'),
                            'r_squared': metrics.get('r_squared'),
                            'mae': metrics.get('mae')
                        }
                        metrics_data.append(metric_row)
                
                if metrics_data:
                    df_metrics = pd.DataFrame(metrics_data)
                    df_metrics.to_excel(writer, sheet_name='Metrics Summary', index=False)
                    
                    # Add statistics sheet
                    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns
                    if not numeric_cols.empty:
                        stats_df = df_metrics[numeric_cols].describe()
                        stats_df.to_excel(writer, sheet_name='Statistics')
        
        logger.info(f"Exported {len(logs)} logs to {output_path}")
    
    def aggregate_metrics(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple logs.
        
        Args:
            logs: List of logs to aggregate
        
        Returns:
            Dictionary containing aggregated statistics
        """
        if not logs:
            return {}
        
        # Extract metrics from all logs
        all_metrics = []
        for log in logs:
            metrics = log.get('performance_metrics', {})
            if metrics:
                # Create a flat metrics dict
                flat_metrics = {
                    'mse': metrics.get('mse'),
                    'r_squared': metrics.get('r_squared'),
                    'mae': metrics.get('mae')
                }
                
                # Add correlation metrics
                if 'correlations' in metrics:
                    if 'pearson' in metrics['correlations']:
                        flat_metrics['pearson_r'] = metrics['correlations']['pearson'].get('r')
                    if 'spearman' in metrics['correlations']:
                        flat_metrics['spearman_r'] = metrics['correlations']['spearman'].get('r')
                
                all_metrics.append(flat_metrics)
        
        # Use metrics_utils aggregation function
        return aggregate_sample_metrics(all_metrics)


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary for CSV export."""
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to string representation
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    
    return dict(items)


def _diff_nested_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any], 
                      path: str = '') -> Dict[str, Any]:
    """Find differences between two nested dictionaries."""
    differences = {}
    
    # Check keys in dict1
    for key, value1 in dict1.items():
        current_path = f"{path}.{key}" if path else key
        
        if key not in dict2:
            differences[current_path] = {'only_in': 'first', 'value': value1}
        elif isinstance(value1, dict) and isinstance(dict2[key], dict):
            nested_diff = _diff_nested_dicts(value1, dict2[key], current_path)
            differences.update(nested_diff)
        elif value1 != dict2[key]:
            differences[current_path] = {
                'first': value1,
                'second': dict2[key]
            }
    
    # Check keys only in dict2
    for key, value2 in dict2.items():
        current_path = f"{path}.{key}" if path else key
        if key not in dict1:
            differences[current_path] = {'only_in': 'second', 'value': value2}
    
    return differences 
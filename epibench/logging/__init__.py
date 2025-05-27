# EpiBench Logging Module
"""
Comprehensive logging system for EpiBench to ensure reproducibility
and enable analysis of experimental results.
"""

from .log_manager import LogManager
from .log_schema import LogSchema, LOG_SCHEMA
from .config_aggregator import ConfigurationAggregator
from .metrics_utils import (
    calculate_derived_metrics,
    calculate_performance_score,
    compare_metrics,
    aggregate_sample_metrics
)
from .log_query import LogQuery
from .log_analysis import (
    LogAnalyzer,
    analyze_experiment_series,
    compare_experiment_groups
)

__all__ = [
    'LogSchema',
    'LOG_SCHEMA',
    'LogManager',
    'ConfigurationAggregator',
    'calculate_derived_metrics',
    'calculate_performance_score',
    'compare_metrics',
    'aggregate_sample_metrics',
    'LogQuery',
    'LogAnalyzer',
    'analyze_experiment_series',
    'compare_experiment_groups'
] 
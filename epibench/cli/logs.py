# -*- coding: utf-8 -*-
"""CLI commands for EpiBench log management."""

import argparse
import logging
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
from tabulate import tabulate
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON
from rich.syntax import Syntax
from rich import box

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from epibench.logging import LogQuery, LogAnalyzer, analyze_experiment_series, compare_experiment_groups
from epibench.utils.logging import LoggerManager
from epibench.logging.visualization import LogDataAdapter, ascii_table, log_summary

logger = logging.getLogger(__name__)
console = Console()


def setup_logs_parser(parser: argparse.ArgumentParser):
    """Adds arguments specific to the logs command."""
    subparsers = parser.add_subparsers(dest='log_command', title='Log Management Commands',
                                     help='Available log management subcommands', required=True)
    
    # List logs command
    list_parser = subparsers.add_parser(
        'list',
        help='List available logs with summary information',
        description='Display a table of all available logs with key metadata'
    )
    list_parser.add_argument(
        '-d', '--log-dir',
        type=str,
        default='logs',
        help='Directory containing log files (default: logs)'
    )
    list_parser.add_argument(
        '--status',
        type=str,
        choices=['completed', 'failed', 'running'],
        help='Filter logs by execution status'
    )
    list_parser.add_argument(
        '--sample',
        type=str,
        help='Filter logs by sample ID'
    )
    list_parser.add_argument(
        '--days',
        type=int,
        help='Show logs from the last N days'
    )
    list_parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Maximum number of logs to display (default: 20)'
    )
    list_parser.add_argument(
        '--sort',
        type=str,
        default='timestamp',
        choices=['timestamp', 'duration', 'sample_id', 'status'],
        help='Sort logs by field (default: timestamp)'
    )
    list_parser.add_argument(
        '--reverse',
        action='store_true',
        help='Reverse sort order'
    )
    list_parser.add_argument(
        '--format',
        type=str,
        default='table',
        choices=['table', 'json', 'csv', 'ascii'],
        help='Output format (default: table)'
    )
    list_parser.set_defaults(func=list_logs)
    
    # Show log command
    show_parser = subparsers.add_parser(
        'show',
        help='View detailed information for a specific log',
        description='Display comprehensive details of a single log entry'
    )
    show_parser.add_argument(
        'log_id',
        type=str,
        help='Execution ID or log filename to display'
    )
    show_parser.add_argument(
        '-d', '--log-dir',
        type=str,
        default='logs',
        help='Directory containing log files (default: logs)'
    )
    show_parser.add_argument(
        '--section',
        type=str,
        choices=['all', 'metadata', 'config', 'performance', 'runtime', 'pipeline'],
        default='all',
        help='Specific section to display (default: all)'
    )
    show_parser.add_argument(
        '--format',
        type=str,
        default='rich',
        choices=['rich', 'json', 'yaml', 'ascii'],
        help='Output format (default: rich)'
    )
    show_parser.set_defaults(func=show_log)
    
    # Search logs command
    search_parser = subparsers.add_parser(
        'search',
        help='Search logs matching specific criteria',
        description='Find logs using flexible query parameters'
    )
    search_parser.add_argument(
        '-d', '--log-dir',
        type=str,
        default='logs',
        help='Directory containing log files (default: logs)'
    )
    search_parser.add_argument(
        '--metric',
        type=str,
        help='Filter by metric value (e.g., "r_squared>0.8")'
    )
    search_parser.add_argument(
        '--config',
        type=str,
        help='Filter by config parameter (e.g., "model.batch_size=32")'
    )
    search_parser.add_argument(
        '--status',
        type=str,
        choices=['completed', 'failed', 'running'],
        help='Filter by execution status'
    )
    search_parser.add_argument(
        '--sample',
        type=str,
        help='Filter by sample ID (supports wildcards)'
    )
    search_parser.add_argument(
        '--date-from',
        type=str,
        help='Start date for search (ISO format: YYYY-MM-DD)'
    )
    search_parser.add_argument(
        '--date-to',
        type=str,
        help='End date for search (ISO format: YYYY-MM-DD)'
    )
    search_parser.add_argument(
        '--limit',
        type=int,
        default=50,
        help='Maximum number of results (default: 50)'
    )
    search_parser.add_argument(
        '--format',
        type=str,
        default='table',
        choices=['table', 'json', 'csv', 'ascii'],
        help='Output format (default: table)'
    )
    search_parser.set_defaults(func=search_logs)
    
    # Compare logs command
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare two or more logs to show differences',
        description='Display side-by-side comparison of multiple log entries'
    )
    compare_parser.add_argument(
        'log_ids',
        type=str,
        nargs='+',
        help='Execution IDs or log filenames to compare'
    )
    compare_parser.add_argument(
        '-d', '--log-dir',
        type=str,
        default='logs',
        help='Directory containing log files (default: logs)'
    )
    compare_parser.add_argument(
        '--focus',
        type=str,
        choices=['all', 'config', 'metrics', 'performance'],
        default='all',
        help='Focus comparison on specific aspects (default: all)'
    )
    compare_parser.add_argument(
        '--format',
        type=str,
        default='table',
        choices=['table', 'json'],
        help='Output format (default: table)'
    )
    compare_parser.set_defaults(func=compare_logs)
    
    # Export logs command
    export_parser = subparsers.add_parser(
        'export',
        help='Export logs in various formats',
        description='Save filtered logs to files in different formats'
    )
    export_parser.add_argument(
        '-d', '--log-dir',
        type=str,
        default='logs',
        help='Directory containing log files (default: logs)'
    )
    export_parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output file path'
    )
    export_parser.add_argument(
        '--format',
        type=str,
        required=True,
        choices=['csv', 'json', 'excel'],
        help='Export format'
    )
    export_parser.add_argument(
        '--status',
        type=str,
        choices=['completed', 'failed', 'running'],
        help='Filter logs by status'
    )
    export_parser.add_argument(
        '--sample',
        type=str,
        help='Filter logs by sample ID'
    )
    export_parser.add_argument(
        '--fields',
        type=str,
        nargs='+',
        help='Specific fields to export (default: all)'
    )
    export_parser.add_argument(
        '--include-summary',
        action='store_true',
        help='Include summary statistics (for Excel export)'
    )
    export_parser.set_defaults(func=export_logs)
    
    # Analyze logs command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Run statistical analysis on logs',
        description='Perform time series analysis, correlations, and generate insights'
    )
    analyze_parser.add_argument(
        '-d', '--log-dir',
        type=str,
        default='logs',
        help='Directory containing log files (default: logs)'
    )
    analyze_parser.add_argument(
        '--analysis-type',
        type=str,
        choices=['summary', 'time-series', 'correlations', 'groups', 'clustering'],
        default='summary',
        help='Type of analysis to perform (default: summary)'
    )
    analyze_parser.add_argument(
        '--metric',
        type=str,
        default='r_squared',
        help='Metric to analyze (default: r_squared)'
    )
    analyze_parser.add_argument(
        '--group-by',
        type=str,
        help='Configuration parameter to group by (for group analysis)'
    )
    analyze_parser.add_argument(
        '--output',
        type=str,
        help='Save analysis results to file'
    )
    analyze_parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate visualization plots'
    )
    analyze_parser.set_defaults(func=analyze_logs)


def list_logs(args):
    """List available logs with summary information."""
    try:
        query = LogQuery(args.log_dir)
        
        # Apply filters
        if args.status:
            query.filter_by_status(args.status)
        
        if args.sample:
            query.filter_by_sample_id(args.sample)
        
        if args.days:
            start_date = datetime.now() - timedelta(days=args.days)
            query.filter_by_date_range(start_date=start_date)
        
        # Sort and limit
        sort_order = 'desc' if args.reverse else 'asc'
        query.sort_by(args.sort, sort_order).limit(args.limit)
        
        # Execute query
        logs = query.execute()
        
        if not logs:
            console.print("[yellow]No logs found matching the criteria.[/yellow]")
            return
        
        # Format output
        if args.format == 'ascii':
            adapter = LogDataAdapter(logs)
            table = adapter.to_flat_table()
            print(ascii_table(table))
        elif args.format == 'json':
            # Simplified JSON output
            output = []
            for log in logs:
                output.append({
                    'execution_id': log.get('execution_metadata', {}).get('execution_id'),
                    'sample_id': log.get('input_configuration', {}).get('sample_id'),
                    'status': log.get('execution_metadata', {}).get('status'),
                    'timestamp': log.get('execution_metadata', {}).get('timestamp', {}).get('start'),
                    'duration': log.get('runtime_information', {}).get('duration_seconds')
                })
            console.print_json(data=output)
            
        elif args.format == 'csv':
            # CSV output
            df = _logs_to_dataframe(logs)
            print(df.to_csv(index=False))
            
        else:  # table format
            # Create rich table
            table = Table(title=f"EpiBench Logs (showing {len(logs)} of {query.count()} total)")
            table.add_column("Execution ID", style="cyan")
            table.add_column("Sample ID", style="green")
            table.add_column("Status", style="bold")
            table.add_column("Start Time")
            table.add_column("Duration (s)")
            table.add_column("R²", justify="right")
            table.add_column("MSE", justify="right")
            
            for log in logs:
                exec_meta = log.get('execution_metadata', {})
                perf_metrics = log.get('performance_metrics', {})
                
                status = exec_meta.get('status', 'unknown')
                status_style = {
                    'completed': '[green]✓ completed[/green]',
                    'failed': '[red]✗ failed[/red]',
                    'running': '[yellow]⟳ running[/yellow]'
                }.get(status, status)
                
                table.add_row(
                    exec_meta.get('execution_id', 'N/A')[:8],  # Truncate ID
                    log.get('input_configuration', {}).get('sample_id', 'N/A'),
                    status_style,
                    _format_timestamp(exec_meta.get('timestamp', {}).get('start')),
                    f"{log.get('runtime_information', {}).get('duration_seconds', 0):.1f}",
                    f"{perf_metrics.get('r_squared', 0):.3f}" if perf_metrics.get('r_squared') else '-',
                    f"{perf_metrics.get('mse', 0):.3f}" if perf_metrics.get('mse') else '-'
                )
            
            console.print(table)
            
    except Exception as e:
        logger.error(f"Error listing logs: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def show_log(args):
    """Show detailed information for a specific log."""
    try:
        # Find log file
        log_file = _find_log_file(args.log_dir, args.log_id)
        if not log_file:
            console.print(f"[red]Log not found: {args.log_id}[/red]")
            sys.exit(1)
        
        # Load log
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        if args.format == 'ascii':
            print(log_summary(log_data))
        elif args.format == 'json':
            if args.section != 'all':
                log_data = log_data.get(args.section, {})
            console.print_json(data=log_data)
            
        elif args.format == 'yaml':
            import yaml
            if args.section != 'all':
                log_data = log_data.get(args.section, {})
            print(yaml.dump(log_data, default_flow_style=False))
            
        else:  # rich format
            # Display with rich formatting
            _display_log_rich(log_data, args.section)
            
    except Exception as e:
        logger.error(f"Error showing log: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def search_logs(args):
    """Search logs matching specific criteria."""
    try:
        query = LogQuery(args.log_dir)
        
        # Apply filters
        if args.status:
            query.filter_by_status(args.status)
        
        if args.sample:
            query.filter_by_sample_id(args.sample)
        
        if args.date_from:
            start_date = datetime.fromisoformat(args.date_from)
            query.filter_by_date_range(start_date=start_date)
        
        if args.date_to:
            end_date = datetime.fromisoformat(args.date_to)
            query.filter_by_date_range(end_date=end_date)
        
        if args.metric:
            # Parse metric filter (e.g., "r_squared>0.8")
            metric_name, condition, threshold = _parse_metric_filter(args.metric)
            query.filter_by_metric(metric_name, condition, float(threshold))
        
        if args.config:
            # Parse config filter (e.g., "model.batch_size=32")
            config_path, value = args.config.split('=', 1)
            # Try to convert value to appropriate type
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string
            query.filter_by_config(config_path, value)
        
        # Limit results
        query.limit(args.limit)
        
        # Execute query
        logs = query.execute()
        
        if not logs:
            console.print("[yellow]No logs found matching the search criteria.[/yellow]")
            return
        
        console.print(f"[green]Found {len(logs)} matching logs[/green]")
        
        # Format output (reuse list_logs formatting)
        if args.format == 'ascii':
            adapter = LogDataAdapter(logs)
            table = adapter.to_flat_table()
            print(ascii_table(table))
        elif args.format == 'json':
            output = []
            for log in logs:
                output.append({
                    'execution_id': log.get('execution_metadata', {}).get('execution_id'),
                    'sample_id': log.get('input_configuration', {}).get('sample_id'),
                    'status': log.get('execution_metadata', {}).get('status'),
                    'timestamp': log.get('execution_metadata', {}).get('timestamp', {}).get('start'),
                    'r_squared': log.get('performance_metrics', {}).get('r_squared'),
                    'mse': log.get('performance_metrics', {}).get('mse')
                })
            console.print_json(data=output)
        else:
            # Table format
            _display_logs_table(logs)
            
    except Exception as e:
        logger.error(f"Error searching logs: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def compare_logs(args):
    """Compare multiple logs to show differences."""
    try:
        # Load specified logs
        logs = []
        for log_id in args.log_ids:
            log_file = _find_log_file(args.log_dir, log_id)
            if not log_file:
                console.print(f"[red]Log not found: {log_id}[/red]")
                sys.exit(1)
            
            with open(log_file, 'r') as f:
                logs.append(json.load(f))
        
        if len(logs) < 2:
            console.print("[red]Need at least 2 logs to compare[/red]")
            sys.exit(1)
        
        # Perform comparison
        if args.format == 'json':
            # JSON output with full comparison
            comparisons = []
            for i in range(1, len(logs)):
                comp = LogQuery.compare_logs(logs[0], logs[i])
                comp['log1_id'] = logs[0].get('execution_metadata', {}).get('execution_id')
                comp['log2_id'] = logs[i].get('execution_metadata', {}).get('execution_id')
                comparisons.append(comp)
            console.print_json(data=comparisons)
            
        else:  # table format
            # Create comparison table
            _display_comparison_table(logs, args.focus)
            
    except Exception as e:
        logger.error(f"Error comparing logs: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def export_logs(args):
    """Export logs in various formats."""
    try:
        query = LogQuery(args.log_dir)
        
        # Apply filters
        if args.status:
            query.filter_by_status(args.status)
        
        if args.sample:
            query.filter_by_sample_id(args.sample)
        
        # Execute query
        logs = query.execute()
        
        if not logs:
            console.print("[yellow]No logs found to export.[/yellow]")
            return
        
        # Export based on format
        output_path = Path(args.output)
        
        if args.format == 'csv':
            query.export_to_csv(logs, output_path, fields=args.fields)
            
        elif args.format == 'json':
            query.export_to_json(logs, output_path, pretty=True)
            
        elif args.format == 'excel':
            query.export_to_excel(logs, output_path, include_summary=args.include_summary)
        
        console.print(f"[green]✓ Exported {len(logs)} logs to {output_path}[/green]")
        
    except Exception as e:
        logger.error(f"Error exporting logs: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def analyze_logs(args):
    """Run statistical analysis on logs."""
    try:
        if args.analysis_type == 'summary':
            # Quick summary analysis
            results = analyze_experiment_series(args.log_dir, output_format='dict')
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                console.print(f"[green]✓ Analysis saved to {args.output}[/green]")
            else:
                # Display summary
                _display_analysis_summary(results)
                
        elif args.analysis_type == 'time-series':
            # Time series analysis
            query = LogQuery(args.log_dir)
            logs = query.execute()
            
            if not logs:
                console.print("[yellow]No logs found for analysis.[/yellow]")
                return
            
            analyzer = LogAnalyzer(logs)
            results = analyzer.time_series_analysis(args.metric, window_size=5)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
            else:
                _display_time_series_analysis(results)
                
        elif args.analysis_type == 'correlations':
            # Correlation analysis
            query = LogQuery(args.log_dir)
            logs = query.execute()
            
            analyzer = LogAnalyzer(logs)
            results = analyzer.correlation_analysis(target_metric=args.metric)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
            else:
                _display_correlation_analysis(results)
                
        elif args.analysis_type == 'groups':
            # Group comparison
            if not args.group_by:
                console.print("[red]--group-by parameter required for group analysis[/red]")
                sys.exit(1)
            
            results = compare_experiment_groups(args.log_dir, args.group_by, metrics=[args.metric])
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
            else:
                _display_group_comparison(results)
                
        elif args.analysis_type == 'clustering':
            # Clustering analysis
            query = LogQuery(args.log_dir)
            logs = query.execute()
            
            analyzer = LogAnalyzer(logs)
            results = analyzer.experiment_clustering(n_clusters=3)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
            else:
                _display_clustering_results(results)
        
        if args.plot:
            console.print("[yellow]Plot generation not yet implemented[/yellow]")
            
    except Exception as e:
        logger.error(f"Error analyzing logs: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# Helper functions
def _find_log_file(log_dir: str, log_id: str) -> Optional[Path]:
    """Find log file by execution ID or filename."""
    log_path = Path(log_dir)
    
    # Try direct filename
    if log_id.endswith('.json'):
        full_path = log_path / log_id
        if full_path.exists():
            return full_path
    
    # Search by execution ID
    for log_file in log_path.glob("*.json"):
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
                if data.get('execution_metadata', {}).get('execution_id', '').startswith(log_id):
                    return log_file
        except:
            continue
    
    return None


def _format_timestamp(timestamp: Optional[str]) -> str:
    """Format ISO timestamp for display."""
    if not timestamp:
        return '-'
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return timestamp[:16]  # Fallback to simple truncation


def _parse_metric_filter(metric_filter: str) -> tuple:
    """Parse metric filter string like 'r_squared>0.8'."""
    import re
    match = re.match(r'(\w+)([><=!]+)([\d.]+)', metric_filter)
    if not match:
        raise ValueError(f"Invalid metric filter format: {metric_filter}")
    return match.groups()


def _logs_to_dataframe(logs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert logs to pandas DataFrame."""
    data = []
    for log in logs:
        row = {
            'execution_id': log.get('execution_metadata', {}).get('execution_id'),
            'sample_id': log.get('input_configuration', {}).get('sample_id'),
            'status': log.get('execution_metadata', {}).get('status'),
            'start_time': log.get('execution_metadata', {}).get('timestamp', {}).get('start'),
            'duration_seconds': log.get('runtime_information', {}).get('duration_seconds'),
            'r_squared': log.get('performance_metrics', {}).get('r_squared'),
            'mse': log.get('performance_metrics', {}).get('mse'),
            'mae': log.get('performance_metrics', {}).get('mae')
        }
        data.append(row)
    return pd.DataFrame(data)


def _display_log_rich(log_data: Dict[str, Any], section: str):
    """Display log with rich formatting."""
    exec_meta = log_data.get('execution_metadata', {})
    
    # Header panel
    header = Panel(
        f"[bold cyan]Execution ID:[/bold cyan] {exec_meta.get('execution_id', 'N/A')}\n"
        f"[bold green]Sample ID:[/bold green] {log_data.get('input_configuration', {}).get('sample_id', 'N/A')}\n"
        f"[bold]Status:[/bold] {exec_meta.get('status', 'unknown')}\n"
        f"[bold]Start Time:[/bold] {_format_timestamp(exec_meta.get('timestamp', {}).get('start'))}",
        title="Log Details",
        expand=False
    )
    console.print(header)
    
    # Display requested sections
    if section in ['all', 'metadata']:
        console.print("\n[bold]Execution Metadata:[/bold]")
        console.print(JSON(json.dumps(exec_meta, indent=2)))
    
    if section in ['all', 'config']:
        config = log_data.get('configuration_parameters', {})
        if config:
            console.print("\n[bold]Configuration Parameters:[/bold]")
            console.print(JSON(json.dumps(config, indent=2)))
    
    if section in ['all', 'performance']:
        perf = log_data.get('performance_metrics', {})
        if perf:
            console.print("\n[bold]Performance Metrics:[/bold]")
            console.print(JSON(json.dumps(perf, indent=2)))
    
    if section in ['all', 'runtime']:
        runtime = log_data.get('runtime_information', {})
        if runtime:
            console.print("\n[bold]Runtime Information:[/bold]")
            console.print(JSON(json.dumps(runtime, indent=2)))
    
    if section in ['all', 'pipeline']:
        pipeline = log_data.get('pipeline_information', {})
        if pipeline:
            console.print("\n[bold]Pipeline Information:[/bold]")
            console.print(JSON(json.dumps(pipeline, indent=2)))


def _display_logs_table(logs: List[Dict[str, Any]]):
    """Display logs in a formatted table."""
    table = Table(title=f"Search Results ({len(logs)} logs)")
    table.add_column("Execution ID", style="cyan")
    table.add_column("Sample ID", style="green")
    table.add_column("Status", style="bold")
    table.add_column("Start Time")
    table.add_column("R²", justify="right")
    table.add_column("MSE", justify="right")
    
    for log in logs:
        exec_meta = log.get('execution_metadata', {})
        perf_metrics = log.get('performance_metrics', {})
        
        status = exec_meta.get('status', 'unknown')
        status_style = {
            'completed': '[green]completed[/green]',
            'failed': '[red]failed[/red]',
            'running': '[yellow]running[/yellow]'
        }.get(status, status)
        
        table.add_row(
            exec_meta.get('execution_id', 'N/A')[:12],
            log.get('input_configuration', {}).get('sample_id', 'N/A'),
            status_style,
            _format_timestamp(exec_meta.get('timestamp', {}).get('start')),
            f"{perf_metrics.get('r_squared', 0):.3f}" if perf_metrics.get('r_squared') else '-',
            f"{perf_metrics.get('mse', 0):.3f}" if perf_metrics.get('mse') else '-'
        )
    
    console.print(table)


def _display_comparison_table(logs: List[Dict[str, Any]], focus: str):
    """Display comparison table for multiple logs."""
    table = Table(title="Log Comparison", box=box.ROUNDED)
    
    # Add columns
    table.add_column("Property", style="bold")
    for i, log in enumerate(logs):
        exec_id = log.get('execution_metadata', {}).get('execution_id', 'N/A')[:8]
        table.add_column(f"Log {i+1}\n{exec_id}", justify="center")
    
    # Add rows based on focus
    if focus in ['all', 'metrics', 'performance']:
        # Performance metrics
        for metric in ['r_squared', 'mse', 'mae']:
            values = []
            for log in logs:
                val = log.get('performance_metrics', {}).get(metric)
                values.append(f"{val:.3f}" if val is not None else '-')
            table.add_row(metric.replace('_', ' ').title(), *values)
    
    if focus in ['all', 'config']:
        # Key configuration parameters
        all_params = set()
        for log in logs:
            key_params = log.get('configuration_parameters', {}).get('key_parameters', {})
            all_params.update(key_params.keys())
        
        for param in sorted(all_params):
            values = []
            for log in logs:
                val = log.get('configuration_parameters', {}).get('key_parameters', {}).get(param)
                values.append(str(val) if val is not None else '-')
            table.add_row(f"Config: {param}", *values)
    
    if focus in ['all']:
        # Runtime information
        table.add_row("Duration (s)", *[
            f"{log.get('runtime_information', {}).get('duration_seconds', 0):.1f}"
            for log in logs
        ])
        
        table.add_row("Status", *[
            log.get('execution_metadata', {}).get('status', 'unknown')
            for log in logs
        ])
    
    console.print(table)


def _display_analysis_summary(results: Dict[str, Any]):
    """Display analysis summary results."""
    summary = results.get('summary', {})
    
    # Overview panel
    overview = summary.get('overview', {})
    panel = Panel(
        f"[bold]Total Experiments:[/bold] {overview.get('total_experiments', 0)}\n"
        f"[bold]Unique Samples:[/bold] {overview.get('unique_samples', 0)}\n"
        f"[bold]Completion Rate:[/bold] {overview.get('completion_rate', 0):.1%}\n"
        f"[bold]Date Range:[/bold] {overview.get('date_range', {}).get('start', 'N/A')} to "
        f"{overview.get('date_range', {}).get('end', 'N/A')}",
        title="Experiment Overview",
        expand=False
    )
    console.print(panel)
    
    # Performance summary table
    perf_summary = summary.get('performance_summary', {})
    if perf_summary:
        table = Table(title="Performance Summary")
        table.add_column("Metric", style="bold")
        table.add_column("Mean", justify="right")
        table.add_column("Std Dev", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        
        for metric, stats in perf_summary.items():
            table.add_row(
                metric.replace('_', ' ').title(),
                f"{stats['mean']:.3f}",
                f"{stats['std']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}"
            )
        
        console.print("\n")
        console.print(table)
    
    # Best configuration
    best_config = summary.get('best_configurations', {}).get('highest_r_squared')
    if best_config:
        console.print("\n[bold]Best Configuration (Highest R²):[/bold]")
        console.print(f"  Execution ID: {best_config['execution_id']}")
        console.print(f"  R² Score: {best_config['r_squared']:.3f}")
        console.print("  Key Parameters:")
        for param, value in best_config.get('key_params', {}).items():
            console.print(f"    {param}: {value}")


def _display_time_series_analysis(results: Dict[str, Any]):
    """Display time series analysis results."""
    console.print(Panel(f"[bold]Time Series Analysis: {results.get('metric')}[/bold]", expand=False))
    
    # Basic info
    console.print(f"\n[bold]Data Points:[/bold] {results.get('data_points', 0)}")
    console.print(f"[bold]Time Range:[/bold] {results.get('time_range', {}).get('start')} to "
                 f"{results.get('time_range', {}).get('end')}")
    
    # Statistics
    stats = results.get('statistics', {})
    if stats:
        console.print("\n[bold]Statistics:[/bold]")
        console.print(f"  Mean: {stats.get('mean', 0):.3f}")
        console.print(f"  Std Dev: {stats.get('std', 0):.3f}")
        console.print(f"  Min: {stats.get('min', 0):.3f}")
        console.print(f"  Max: {stats.get('max', 0):.3f}")
        
        trend = stats.get('trend', {})
        if trend:
            console.print(f"\n[bold]Trend Analysis:[/bold]")
            console.print(f"  Direction: {trend.get('trend_direction', 'unknown')}")
            console.print(f"  Slope: {trend.get('slope', 0):.6f}")
            console.print(f"  R²: {trend.get('r_squared', 0):.3f}")
            console.print(f"  P-value: {trend.get('p_value', 0):.4f}")
    
    # Anomalies
    anomalies = results.get('anomalies', {})
    if anomalies.get('count', 0) > 0:
        console.print(f"\n[bold yellow]Anomalies Detected:[/bold yellow] {anomalies['count']}")


def _display_correlation_analysis(results: Dict[str, Any]):
    """Display correlation analysis results."""
    console.print(Panel(
        f"[bold]Correlation Analysis[/bold]\n"
        f"Target Metric: {results.get('target_metric')}\n"
        f"Method: {results.get('method')}\n"
        f"Sample Size: {results.get('sample_size')}",
        expand=False
    ))
    
    # Top correlations table
    top_corr = results.get('top_correlations', {})
    if top_corr:
        table = Table(title="Top Correlations")
        table.add_column("Parameter", style="bold")
        table.add_column("Correlation", justify="right")
        table.add_column("P-value", justify="right")
        table.add_column("Significant", justify="center")
        
        for param, stats in list(top_corr.items())[:10]:
            sig_marker = "✓" if stats['significant'] else ""
            table.add_row(
                param.replace('config_', ''),
                f"{stats['correlation']:.3f}",
                f"{stats['p_value']:.4f}",
                f"[green]{sig_marker}[/green]" if sig_marker else ""
            )
        
        console.print("\n")
        console.print(table)


def _display_group_comparison(results: Dict[str, Any]):
    """Display group comparison results."""
    for metric, comparison in results.items():
        console.print(Panel(f"[bold]Group Comparison: {metric}[/bold]", expand=False))
        
        # Group info
        console.print(f"\n[bold]Grouping Variable:[/bold] {comparison.get('group_by')}")
        console.print(f"[bold]Number of Groups:[/bold] {comparison.get('n_groups')}")
        
        # Group sizes
        console.print("\n[bold]Group Sizes:[/bold]")
        for group, size in comparison.get('group_sizes', {}).items():
            console.print(f"  {group}: {size} samples")
        
        # Test results
        test = comparison.get('test', {})
        if test:
            console.print(f"\n[bold]Statistical Test:[/bold] {test.get('type')}")
            console.print(f"[bold]Test Statistic:[/bold] {test.get('statistic', 0):.3f}")
            console.print(f"[bold]P-value:[/bold] {test.get('p_value', 0):.4f}")
            
            if test.get('significant'):
                console.print("[bold green]Result: Statistically significant difference[/bold green]")
            else:
                console.print("[yellow]Result: No significant difference[/yellow]")
            
            if 'effect_size' in test:
                console.print(f"[bold]Effect Size (Cohen's d):[/bold] {test['effect_size']:.3f}")


def _display_clustering_results(results: Dict[str, Any]):
    """Display clustering analysis results."""
    console.print(Panel("[bold]Experiment Clustering Analysis[/bold]", expand=False))
    
    console.print(f"\n[bold]Number of Clusters:[/bold] {results.get('n_clusters')}")
    console.print(f"[bold]Features Used:[/bold] {len(results.get('features_used', []))}")
    
    if 'silhouette_score' in results:
        console.print(f"[bold]Silhouette Score:[/bold] {results['silhouette_score']:.3f}")
    
    # Cluster sizes
    console.print("\n[bold]Cluster Sizes:[/bold]")
    for cluster_id, size in results.get('cluster_sizes', {}).items():
        console.print(f"  Cluster {cluster_id}: {size} experiments")
    
    # Cluster characteristics
    centers = results.get('cluster_centers', {})
    if centers:
        console.print("\n[bold]Cluster Characteristics:[/bold]")
        for cluster_name, features in centers.items():
            console.print(f"\n  {cluster_name}:")
            for feat, value in list(features.items())[:5]:  # Show top 5 features
                console.print(f"    {feat}: {value:.3f}")


def logs_main(args):
    """Main function for the logs command."""
    # Setup logging
    LoggerManager.setup_logger(default_log_level=logging.INFO)
    
    # Dispatch to appropriate subcommand
    if hasattr(args, 'func'):
        args.func(args)
    else:
        console.print("[red]No subcommand specified. Use 'epibench logs --help' for usage.[/red]")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EpiBench log management CLI")
    setup_logs_parser(parser)
    args = parser.parse_args()
    logs_main(args) 
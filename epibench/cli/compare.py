# -*- coding: utf-8 -*-
"""CLI command for running comparative analyses of EpiBench models."""

import argparse
import logging
import sys
import os
import json

# Add project root to sys.path (if running as script)
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# Assuming necessary imports - adjust based on actual locations
from epibench.config import ConfigManager
from epibench.analysis.comparative import ComparativeAnalyzer # Needed later
from epibench.util.logging_utils import setup_logging
from epibench.util.io_utils import ensure_dir

# Setup basic logger for initial messages
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_compare_parser(parser: argparse.ArgumentParser):
    """Adds arguments specific to the compare command."""
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to the main comparative analysis configuration file (YAML/JSON)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Directory to save the comparative analysis results (reports, plots)."
    )
    # Add other specific arguments if needed for the compare command itself,
    # although most settings should likely be in the config file.
    parser.add_argument(
        "--metrics",
        nargs='+', # Allows specifying multiple metrics
        default=None,
        help="List of metrics to compute during evaluation (e.g., mse r2 pearson). Overrides config if set."
    )
    parser.add_argument(
        "--comparison-metric",
        type=str,
        default=None,
        help="Metric to use for statistical comparisons. Overrides config if set."
    )
    parser.add_argument(
        "--report-format",
        type=str,
        default='json',
        choices=['json', 'csv', 'md'],
        help="Format for the generated comparative report."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Disable generation of comparative visualization plots."
    )

def compare_main(args):
    """Main function for the compare command."""
    # Initial log with basic config
    logger.info("Starting EpiBench comparative analysis...")
    logger.info(f"Arguments received: {args}")

    # --- Load Config (Subtask 15.2) --- 
    compare_config = {}
    try:
        logger.info(f"Loading comparative configuration from: {args.config}")
        config_manager = ConfigManager(args.config)
        compare_config = config_manager.get_config()
        logger.info("Comparative configuration loaded successfully.")
        
        # Basic validation of structure
        if 'model_configs' not in compare_config or not isinstance(compare_config['model_configs'], list):
            raise ValueError("Configuration missing required 'model_configs' list.")
        if 'sample_groups' not in compare_config or not isinstance(compare_config['sample_groups'], dict):
             raise ValueError("Configuration missing required 'sample_groups' dictionary.")
             
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except ValueError as ve:
         logger.error(f"Configuration error in {args.config}: {ve}")
         sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration from {args.config}: {e}", exc_info=True)
        sys.exit(1)

    # --- Setup Logging (using loaded config) ---
    try:
        log_level = compare_config.get('logging', {}).get('level', 'INFO')
        # Allow output dir to be used for log file if specified
        log_file_config = compare_config.get('logging', {}).get('file')
        log_file = os.path.join(args.output_dir, log_file_config) if log_file_config else None
        
        setup_logging(level=log_level, log_file=log_file)
        logger = logging.getLogger(__name__) # Re-get logger after setup
        logger.info("Logging configured.")
        logger.debug(f"Full comparative configuration: {json.dumps(compare_config, indent=2)}")
    except Exception as e:
         # Log error with basic config if logging setup fails
         logging.error(f"Error setting up logging based on config: {e}", exc_info=True)
         # Continue with basic logging setup

    # --- Ensure Output Directory --- 
    try:
        ensure_dir(args.output_dir)
        logger.info(f"Output directory ensured: {args.output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {args.output_dir}: {e}", exc_info=True)
        sys.exit(1)

    # --- Initialize ComparativeAnalyzer (Subtask 15.4 Placeholder) ---
    logger.info("--- Initializing Comparative Analyzer (Placeholder) ---")
    analyzer = ComparativeAnalyzer(
        model_configs=compare_config.get('model_configs', []),
        sample_group_data=compare_config.get('sample_groups', {}),
        base_output_dir=args.output_dir
    )

    # --- Run Analysis (Subtask 15.4 Placeholder) ---
    logger.info("--- Running Comparative Analysis (Placeholder) ---")
    metrics = args.metrics or compare_config.get('evaluation_metrics', ['mse', 'r2']) # Example precedence
    comp_metric = args.comparison_metric or compare_config.get('statistical_comparison_metric', 'mse')
    
    if analyzer:
        analyzer.run_analysis(
            metrics_to_compute=metrics,
            comparison_metric=comp_metric,
            report_format=args.report_format,
            generate_plots=not args.no_plots
        )
    else:
         logger.error("Analyzer initialization failed. Cannot run analysis.")

    logger.info("EpiBench comparative analysis finished (Placeholder).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run comparative analysis for EpiBench models.")
    setup_compare_parser(parser)
    args = parser.parse_args()
    compare_main(args) 
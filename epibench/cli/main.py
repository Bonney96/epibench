# -*- coding: utf-8 -*-
"""Main CLI entry point for the EpiBench toolkit."""

import argparse
import logging
import sys
import os
import importlib.metadata # Import for version retrieval

# Add project root to sys.path to allow running script directly during development
# This assumes the script is in epibench/cli/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# Import subcommand setup functions and main functions
from .process_data import setup_process_data_parser, process_data_main
from .train import setup_arg_parser as setup_train_parser, main as train_main
from .evaluate import setup_evaluate_parser, evaluate_main
from .predict import setup_predict_parser, predict_main
from .interpret import setup_interpret_parser, interpret_main
from .compare import setup_compare_parser, compare_main
from .logs import setup_logs_parser, logs_main

# Basic logger setup for the main entry point
# Logging will be potentially reconfigured by subcommands based on their configs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('epibench') # Get root logger for the package

def main():
    """Main function to parse arguments and dispatch subcommands."""
    
    # --- Get Package Version --- 
    try:
        package_version = importlib.metadata.version('epibench')
    except importlib.metadata.PackageNotFoundError:
        package_version = 'unknown' # Fallback if package not installed
        logger.warning("Could not determine package version. Is EpiBench installed?")
        
    parser = argparse.ArgumentParser(
        description="EpiBench: A toolkit for epigenetic benchmarking and analysis.",
        epilog="Use 'epibench <command> --help' for more information on a specific command."
    )
    
    # Add global arguments if any (e.g., --version, --verbose)
    parser.add_argument(
         '--version',
         action='version',
         version=f'%(prog)s {package_version}', # Display package version
         help="Show program's version number and exit."
     )
    parser.add_argument(
         "-v", "--verbose",
         action="store_true",
         help="Increase output verbosity (set logging level to DEBUG)."
    )
    
    # --- Setup Subparsers (Subtask 18.2) --- 
    subparsers = parser.add_subparsers(dest='command', title='Available Commands', 
                                       help='Subcommand to execute', required=True)
    
    # Process Data Command
    process_parser = subparsers.add_parser(
        'process-data', 
        help='Preprocess raw genomic data into required matrix format.',
        description='Reads raw data (e.g., bed, bigwig), processes it, performs chromosome-based splitting, and saves output matrices.'
    )
    setup_process_data_parser(process_parser)
    process_parser.set_defaults(func=process_data_main)
    
    # Train Command
    train_parser = subparsers.add_parser(
        'train', 
        help='Train a model or run hyperparameter optimization.',
        description='Trains the specified model using the provided configuration. Can optionally perform hyperparameter optimization using Optuna.'
    )
    setup_train_parser(train_parser)
    train_parser.set_defaults(func=train_main)
    
    # Evaluate Command
    evaluate_parser = subparsers.add_parser(
        'evaluate', 
        help='Evaluate a trained model on test data.',
        description='Loads a trained model checkpoint and evaluates its performance on a test dataset using specified metrics.'
    )
    setup_evaluate_parser(evaluate_parser)
    evaluate_parser.set_defaults(func=evaluate_main)
    
    # Predict Command
    predict_parser = subparsers.add_parser(
        'predict', 
        help='Generate predictions using a trained model.',
        description='Loads a trained model checkpoint and generates predictions for a given input dataset.'
    )
    setup_predict_parser(predict_parser)
    predict_parser.set_defaults(func=predict_main)
    
    # Interpret Command
    interpret_parser = subparsers.add_parser(
        'interpret', 
        help='Interpret model predictions using methods like Integrated Gradients.',
        description='Applies model interpretation techniques (e.g., Integrated Gradients) to understand feature importance for model predictions.'
    )
    setup_interpret_parser(interpret_parser)
    interpret_parser.set_defaults(func=interpret_main)
    
    # Compare Command
    compare_parser = subparsers.add_parser(
        'compare', 
        help='Run comparative analysis between different models or sample groups.',
        description='Performs comparative analysis by evaluating multiple model configurations across different sample groups and generating reports/visualizations.'
    )
    setup_compare_parser(compare_parser)
    compare_parser.set_defaults(func=compare_main)
    
    # Logs Command
    logs_parser = subparsers.add_parser(
        'logs',
        help='Manage and analyze EpiBench execution logs.',
        description='View, search, compare, export, and analyze logs from EpiBench runs to track experiments and analyze performance trends.'
    )
    setup_logs_parser(logs_parser)
    logs_parser.set_defaults(func=logs_main)
    
    logger.debug("Parsing command line arguments...")
    args = parser.parse_args()
    
    # Set verbosity based on global flag *before* calling subcommand function
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose mode enabled (DEBUG logging).")
        # Note: Subcommand-specific logging config might override this later

    # --- Dispatch to Subcommand Function --- 
    logger.debug(f"Dispatching command: {args.command}")
    # Check if the selected command has an associated function and call it
    if hasattr(args, 'func'):
        try:
            args.func(args) # Call the function associated with the subcommand
            logger.info(f"Command '{args.command}' executed successfully.")
        except Exception as e:
            logger.error(f"Error executing command '{args.command}': {e}", exc_info=True)
            sys.exit(1)
    else:
        # This should not happen if subparsers are required=True and set_defaults is used
        logger.error(f"No function associated with command '{args.command}'. This indicates an internal setup error.")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main() 
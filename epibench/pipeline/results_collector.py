import logging
from pathlib import Path
import pandas as pd
import numpy as np # Ensure numpy is imported
from typing import List, Dict, Any, Optional
import json
# import yaml # Removed unused import
import glob # Import glob for finding files
from pandas.io.formats.style import Styler # Import Styler for type hinting
import io # For plot encoding
import base64 # For plot encoding
from datetime import datetime # For timestamping reports

# Plotting libraries (add to requirements)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from jinja2 import Environment, FileSystemLoader, select_autoescape
import copy # For deep copying config dictionaries

# PDF Generation library (add to requirements)
import weasyprint

# Removed unused imports
# from epibench.pipeline.pipeline_executor import PipelineExecutor 
# from epibench.validation.config_validator import ProcessConfig

logger = logging.getLogger(__name__)

# Helper class for JSON serialization of numpy types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Handle NaN/Inf specifically
            if np.isnan(obj) or np.isinf(obj):
                return None 
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp): # Handle pandas Timestamp
             return obj.isoformat()
        # Let the base class default method raise the TypeError for other types
        return super(NumpyJSONEncoder, self).default(obj)

# Constants for default config and presets
DEFAULT_REPORT_CONFIG = {
    "output_formats": ["html", "pdf"],
    "report_preset": "detailed", # Presets: "detailed", "summary", "custom"
    "included_sections": ["summary_stats", "plots", "detailed_metrics", "failed_samples"], 
    "included_metrics": "all", # List of metric keys or "all"
    "included_plots": "all", # List of plot keys or "all"
    "theme": { 
        "primary_color": "#0d6efd", # Default Bootstrap blue
        "font_family": "sans-serif"
    },
    "report_title": "EpiBench Evaluation Report"
}

REPORT_PRESETS = {
    "detailed": {
        # Uses most defaults
        "included_sections": ["summary_stats", "plots", "detailed_metrics", "failed_samples"], 
        "included_metrics": "all",
        "included_plots": "all",
    },
    "summary": {
        "output_formats": ["pdf", "html"], # Example override
        "included_sections": ["summary_stats", "plots"], # Fewer sections
        "included_metrics": ["accuracy", "mse", "r2", "mae"], # Key overall metrics (adjust as needed)
        "included_plots": ["accuracy_bar", "mse_r2_scatter"], # Key overall plots (adjust as needed)
        "theme": { "primary_color": "#6c757d" } # Example theme override
    }
    # Add more presets if needed
}

class ResultsCollector:
    """
    Collects, aggregates, and reports results from pipeline execution across multiple samples.
    """

    def __init__(self, base_output_directory: Path, checkpoint_data: Dict[str, Any], report_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the ResultsCollector.

        Args:
            base_output_directory: The root directory where all pipeline outputs are stored.
            checkpoint_data: Dictionary containing the status of each processed sample.
            report_config: Dictionary to customize report generation (optional).
        """
        # self.main_config = main_config # Removed main_config
        self.checkpoint_data = checkpoint_data
        # --- Use base_output_directory directly ---
        self.base_output_directory = Path(base_output_directory) # Ensure it's a Path object
        self.jinja_env = self._setup_jinja_env() # Setup Jinja2 environment
        self.report_config = self._process_report_config(report_config) # Process and store config
        logger.info(f"ResultsCollector initialized. Output Dir: {self.base_output_directory}. Report Config: {self.report_config}")

    def _process_report_config(self, user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes user config, applies presets, sets defaults, and validates."""
        config = copy.deepcopy(DEFAULT_REPORT_CONFIG)
        preset_name = "detailed" # Default preset

        if user_config:
            preset_name = user_config.get("report_preset", preset_name)
            
            # Apply preset overrides first if not "custom"
            if preset_name != "custom" and preset_name in REPORT_PRESETS:
                 preset_settings = copy.deepcopy(REPORT_PRESETS[preset_name])
                 # Deep merge preset into config (simple dict update for theme)
                 for key, value in preset_settings.items():
                     if isinstance(value, dict) and isinstance(config.get(key), dict):
                          config[key].update(value)
                     else:
                         config[key] = value
                
            # Now, apply specific user overrides over preset/defaults
            for key, value in user_config.items():
                if key == "report_preset": continue # Already handled
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                     config[key].update(value) # Merge theme dict
                elif key in config: # Only update keys present in default config
                    config[key] = value
                else:
                     logger.warning(f"Ignoring unknown report configuration key: '{key}'")
                     
        # --- Basic Validation --- 
        if not isinstance(config.get("output_formats"), list):
            logger.warning("'output_formats' must be a list. Using default.")
            config["output_formats"] = DEFAULT_REPORT_CONFIG["output_formats"]
        else:
            config["output_formats"] = [fmt for fmt in config["output_formats"] if fmt in ["html", "pdf"]]
            if not config["output_formats"]:
                 logger.warning("No valid 'output_formats' specified. Defaulting to HTML.")
                 config["output_formats"] = ["html"]

        if not isinstance(config.get("included_sections"), list):
             logger.warning("'included_sections' must be a list. Using default.")
             config["included_sections"] = DEFAULT_REPORT_CONFIG["included_sections"]
             
        if not isinstance(config.get("included_metrics"), (list, str)) or (isinstance(config.get("included_metrics"), str) and config["included_metrics"] != "all"):
             logger.warning("'included_metrics' must be a list or 'all'. Using default.")
             config["included_metrics"] = DEFAULT_REPORT_CONFIG["included_metrics"]
             
        if not isinstance(config.get("included_plots"), (list, str)) or (isinstance(config.get("included_plots"), str) and config["included_plots"] != "all"):
             logger.warning("'included_plots' must be a list or 'all'. Using default.")
             config["included_plots"] = DEFAULT_REPORT_CONFIG["included_plots"]
             
        if not isinstance(config.get("theme"), dict):
             logger.warning("'theme' must be a dictionary. Using default.")
             config["theme"] = DEFAULT_REPORT_CONFIG["theme"]
             
        return config

    def _setup_jinja_env(self) -> Optional[Environment]: # Added Optional
        """Initialize and configure Jinja2 environment."""
        try:
            template_dir = Path(__file__).parent / "templates"
            if not template_dir.is_dir():
                 logger.warning(f"Template directory not found: {template_dir}. HTML reports may fail.")
                 # Return a dummy environment or raise error? For now, log warning.
                 return None 
            
            env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True
            )
            # Add custom filters if needed later
            # env.filters['format_number'] = lambda x: f"{x:.3f}" 
            return env
        except Exception as e:
             logger.error(f"Failed to initialize Jinja2 environment: {e}", exc_info=True)
             return None

    def collect_all(self) -> Dict[str, Any]:
        """
        Orchestrates the collection, aggregation, and reporting process for all samples.
        Returns the aggregated data dictionary (or the report context for now).
        """
        logger.info("Starting results collection...")
        
        # 1. Identify successfully completed samples
        completed_samples = self._find_completed_samples()
        if not completed_samples:
            logger.warning("No completed samples found based on checkpoint data. Nothing to collect.")
            # Return empty structure consistent with successful run with no completed samples
            return {"metrics": {}, "prediction_files": {}, "training_logs": {}}

        # 2. Define summary directory path using base_output_directory
        summary_dir = self.base_output_directory / "summary_reports"
        try:
            summary_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured summary directory exists: {summary_dir}")
        except OSError as e:
            logger.error(f"Failed to create summary directory {summary_dir}: {e}")
            # Return error info in the expected structure
            return {"error": f"Failed to create summary directory: {e}", "metrics": {}, "prediction_files": {}, "training_logs": {}} 

        # 3. Aggregate results from each completed sample
        aggregated_data = self._aggregate_sample_results(completed_samples)
        
        # 4. Prepare data for reporting
        report_context = self._prepare_report_data(aggregated_data)

        # 5. Generate reports using the prepared context
        self._generate_reports(report_context, summary_dir)

        # 6. Summarize success/failure
        self._summarize_run_status(summary_dir)

        logger.info("Results collection finished.")
        # Return the aggregated data for now, might return report_context later
        return aggregated_data 

    def _find_completed_samples(self) -> List[str]:
        """Identifies sample IDs marked as 'completed' in the checkpoint data."""
        completed = [
            sample_id 
            for sample_id, data in self.checkpoint_data.items() 
            if data.get('status') == 'completed'
        ]
        logger.info(f"Found {len(completed)} completed samples: {completed}")
        return completed

    def _aggregate_sample_results(self, completed_samples: List[str]) -> Dict[str, Any]:
        """
        Loads and aggregates key results/metrics from each completed sample's output directory.
        
        Assumes standard output structure: 
        <output_dir>/<sample_id>/evaluation_output/evaluation_metrics.json
        <output_dir>/<sample_id>/prediction_output/predictions.csv
        
        Args:
            completed_samples: List of sample IDs that completed successfully.

        Returns:
            A dictionary containing aggregated data (e.g., metrics dict, paths to predictions).
        """
        logger.info(f"Aggregating results for {len(completed_samples)} samples...")
        # Store collected metrics per sample and paths to main output files
        all_results = {"metrics": {}, "prediction_files": {}, "training_logs": {}} 

        for sample_id in completed_samples:
            # --- Use base_output_directory ---
            sample_output_dir = self.base_output_directory / sample_id 
            eval_dir = sample_output_dir / "evaluation_output" # Default subdir name
            predict_dir = sample_output_dir / "prediction_output" # Default subdir name
            
            logger.debug(f"Checking results for sample {sample_id} in {eval_dir} and {predict_dir}")
            
            # 1. Load Evaluation Metrics
            metrics_file = eval_dir / "evaluation_metrics.json" 
            if metrics_file.is_file():
                try:
                    with open(metrics_file, 'r') as f:
                        # Add sample_id to each metric dict for easier DataFrame creation later
                        metric_dict = json.load(f)
                        metric_dict['sample_id'] = sample_id # Add sample identifier
                        all_results["metrics"][sample_id] = metric_dict 
                        logger.debug(f"Loaded metrics for {sample_id} from {metrics_file}")
                except (IOError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to load or parse metrics file {metrics_file} for sample {sample_id}: {e}")
                    all_results["metrics"][sample_id] = {"error": "failed_to_load", "path": str(metrics_file), "sample_id": sample_id}
            else:
                logger.warning(f"Metrics file not found for completed sample {sample_id} at {metrics_file}")
                all_results["metrics"][sample_id] = {"error": "not_found", "path": str(metrics_file), "sample_id": sample_id}

            # 2. Find Predictions File Path
            predictions_file = predict_dir / "predictions.csv" # Assumed file name
            if predictions_file.is_file():
                 all_results["prediction_files"][sample_id] = str(predictions_file)
                 logger.debug(f"Found predictions file for {sample_id} at {predictions_file}")
            else:
                 logger.warning(f"Predictions file not found for completed sample {sample_id} at {predictions_file}")
                 all_results["prediction_files"][sample_id] = {"error": "not_found", "path": str(predictions_file)}

            # 3. Training Log Info (Simplified)
            train_log_dir = sample_output_dir / 'training_output' # Assume default name
            logger.warning(f"Skipping detailed training log collection for {sample_id}. "
                           f"Check TensorBoard logs potentially within '{train_log_dir}' for training progress.")
            all_results["training_logs"][sample_id] = {"status": "skipped", "reason": "TensorBoard format not parsed", "assumed_dir": str(train_log_dir)}

        # Consolidate metrics into a single JSON file (useful for debugging/manual analysis)
        summary_dir = self.base_output_directory / "summary_reports" 
        self._save_consolidated_metrics(all_results.get("metrics", {}), summary_dir)

        logger.info("Finished aggregating results.")
        return all_results

    def _save_consolidated_metrics(self, metrics_data: Dict[str, Any], summary_dir: Path):
        """
        Saves consolidated metrics data to a JSON file in the summary directory.
        
        Args:
            metrics_data: Dictionary containing metrics to save
            summary_dir: Directory where the summary file should be saved
        """
        metrics_file = summary_dir / "consolidated_metrics.json"
        logger.info(f"Saving consolidated metrics to {metrics_file}")
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2, cls=NumpyJSONEncoder)
            logger.info(f"Successfully saved consolidated metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save consolidated metrics: {e}")

    def _prepare_report_data(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares all necessary data structures for report generation, considering config.

        Args:
            aggregated_data: The dictionary returned by _aggregate_sample_results.

        Returns:
            A dictionary containing structured data ready for report templates.
        """
        logger.info("Preparing data for reports...")
        report_context = {}
        included_metrics_config = self.report_config.get("included_metrics", "all")

        # Filter out samples where metrics loading failed and create DataFrame
        all_metrics_dict = aggregated_data.get("metrics", {})
        valid_metrics_list = [
            m for m in all_metrics_dict.values() 
            if isinstance(m, dict) and "error" not in m
        ]
        
        metrics_df = pd.DataFrame()
        if valid_metrics_list:
             try:
                 metrics_df = pd.DataFrame(valid_metrics_list)
                 # Filter columns based on config
                 if isinstance(included_metrics_config, list):
                      cols_to_keep = [col for col in included_metrics_config if col in metrics_df.columns] + ['sample_id'] # Always keep sample_id
                      metrics_df = metrics_df[list(set(cols_to_keep))] # Use set to avoid duplicates
                      logger.info(f"Filtered metrics DataFrame to include: {list(metrics_df.columns)}")
                 elif included_metrics_config != "all":
                      logger.warning(f"Invalid 'included_metrics' config: {included_metrics_config}. Using all available metrics.")
                 
                 logger.info(f"Created DataFrame with {len(metrics_df)} valid metric records.")
             except Exception as e:
                  logger.error(f"Failed to create or filter DataFrame from valid metrics: {e}", exc_info=True)
                  metrics_df = pd.DataFrame() # Ensure it's empty on error
        else:
            logger.warning("No valid metric records found to create DataFrame.")

        # 1. Prepare Summary Statistics (uses filtered df)
        report_context['summary_stats_dict'] = self._prepare_summary_stats(metrics_df) 

        # 2. Prepare Data for Visualizations (uses filtered df, filters plots)
        report_context['visualization_data'] = self._prepare_visualization_data(metrics_df)

        # 3. Prepare Styled Tables (uses filtered df)
        report_context['styled_tables_html'] = self._prepare_styled_tables(metrics_df, report_context['summary_stats_dict']) 

        # 4. Add Run Metadata (Example)
        report_context['run_metadata'] = {
            'total_samples_in_checkpoint': len(self.checkpoint_data),
            'completed_samples_count': len(self._find_completed_samples()), 
            'timestamp': datetime.now(), 
            'base_output_directory': str(self.base_output_directory),
            'report_title': self.report_config.get("report_title", DEFAULT_REPORT_CONFIG["report_title"]), # Add title from config
            'config_used': self.report_config # Include the actual config used
        }
        
        # 5. Add information about failed samples (from _summarize_run_status logic)
        failed_samples = [
            sample_id 
            for sample_id, data in self.checkpoint_data.items() 
            if data.get('status') == 'failed'
        ]
        report_context['failed_samples'] = failed_samples
        
        logger.info("Finished preparing report data.")
        return report_context

    def _prepare_summary_stats(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates overall summary statistics from the aggregated metrics DataFrame.

        Args:
            metrics_df: DataFrame containing metrics for successfully processed samples.

        Returns:
            A dictionary containing summary statistics.
        """
        logger.info("Calculating summary statistics...")
        summary = {}
        
        if metrics_df.empty:
            summary['message'] = "No valid metrics found to summarize."
            logger.warning(summary['message'])
            return summary

        # Identify numeric columns dynamically, exclude 'sample_id' if present and numeric by chance
        numeric_cols = metrics_df.select_dtypes(include=np.number).columns
        if 'sample_id' in numeric_cols: # This shouldn't happen if sample_id is string, but safe check
             try:
                numeric_cols = numeric_cols.drop('sample_id')
             except KeyError:
                 pass # Ignore if sample_id wasn't actually numeric
            
        if not numeric_cols.empty:
            logger.info(f"Calculating stats for numeric metrics: {list(numeric_cols)}")
            summary['num_samples_with_metrics'] = len(metrics_df)
            # Calculate statistics and convert to dict for JSON compatibility
            try:
                stats_df = metrics_df[numeric_cols].agg(
                    ['mean', 'median', 'min', 'max', 'std', 'count']
                )
                summary['metric_summaries'] = stats_df.to_dict()
                # Ensure NaN/Infinity are handled (important for display, not just JSON)
                summary['metric_summaries'] = self._handle_non_serializable_floats(summary['metric_summaries'])
            except Exception as e:
                 logger.error(f"Error during aggregation calculation: {e}", exc_info=True)
                 summary['error'] = f"Error during aggregation: {e}"
                 summary['metric_summaries'] = {}

        else:
             summary['message'] = "No numeric metrics found in the aggregated data."
             logger.warning(summary['message'])
                 
        
        logger.info("Finished calculating summary statistics.")
        return summary

    def _prepare_visualization_data(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepares data structures suitable for various visualizations, considering config.

        Args:
            metrics_df: DataFrame containing metrics for successfully processed samples.

        Returns:
            A dictionary containing data formatted for plotting libraries.
        """
        logger.info("Preparing visualization data...")
        viz_data = {
            "bar_charts": {},
            "line_plots": {},
            "scatter_plots": {},
            "heatmaps": {},
            "box_plots": {}
        }
        included_plots_config = self.report_config.get("included_plots", "all")
        should_include_plot = lambda plot_key: included_plots_config == "all" or plot_key in included_plots_config

        if metrics_df.empty:
            logger.warning("Metrics DataFrame is empty, cannot generate visualization data.")
            return viz_data

        # Example: Prepare data for a bar chart comparing a metric across samples
        plot_key_accuracy = "accuracy_per_sample"
        if should_include_plot(plot_key_accuracy):
            try:
                if 'accuracy' in metrics_df.columns and 'sample_id' in metrics_df.columns:
                    metrics_df['sample_id_str'] = metrics_df['sample_id'].astype(str) 
                    plot_ready_data = metrics_df[['sample_id_str', 'accuracy']].to_dict(orient='list')
                    viz_data["bar_charts"][plot_key_accuracy] = plot_ready_data
                else:
                     logger.warning(f"Required columns ('sample_id', 'accuracy') not found for '{plot_key_accuracy}' bar chart.")
            except Exception as e:
                 logger.error(f"Error preparing data for '{plot_key_accuracy}': {e}", exc_info=True)

        # Example: Prepare data for Plotly scatter plot if 'mse' and 'r2' exist
        plot_key_scatter = "mse_vs_r2"
        if should_include_plot(plot_key_scatter):
            try:
                if 'mse' in metrics_df.columns and 'r2' in metrics_df.columns:
                     scatter_data = metrics_df[['sample_id', 'mse', 'r2']].copy()
                     scatter_data['mse'] = pd.to_numeric(scatter_data['mse'], errors='coerce')
                     scatter_data['r2'] = pd.to_numeric(scatter_data['r2'], errors='coerce')
                     scatter_data.dropna(subset=['mse', 'r2'], inplace=True)
                     viz_data["scatter_plots"][plot_key_scatter] = scatter_data.to_dict(orient='list')
                else:
                     logger.warning(f"Required columns ('mse', 'r2') not found for '{plot_key_scatter}' scatter plot.")
            except Exception as e:
                 logger.error(f"Error preparing data for '{plot_key_scatter}': {e}", exc_info=True)

        logger.info("Finished preparing visualization data.")
        return viz_data

    def _prepare_styled_tables(self, metrics_df: pd.DataFrame, summary_stats_dict: Dict[str, Any]) -> Dict[str, str]:
        """
        Prepares pandas DataFrames as styled HTML tables.

        Args:
            metrics_df: DataFrame containing raw metrics for successful samples.
            summary_stats_dict: Dictionary containing calculated summary statistics.

        Returns:
            A dictionary mapping table names to HTML string representations.
        """
        logger.info("Preparing styled HTML tables...")
        styled_tables_html = {}
        float_format = "{:.3f}" # Default float formatting

        # Table 1: Summary Statistics
        if 'metric_summaries' in summary_stats_dict:
            try:
                summary_df = pd.DataFrame(summary_stats_dict['metric_summaries']).round(3) 
                # Apply styling
                styler = summary_df.style.format(float_format)
                # Add more styling: e.g., highlight min/max
                numeric_cols_summary = summary_df.select_dtypes(include=np.number).columns
                if 'count' not in numeric_cols_summary: # Don't highlight count usually
                    styler = styler.highlight_max(axis=1, color='lightgreen', subset=numeric_cols_summary.drop('count', errors='ignore'))
                    styler = styler.highlight_min(axis=1, color='lightcoral', subset=numeric_cols_summary.drop('count', errors='ignore'))
                
                styled_tables_html['summary_statistics'] = styler.to_html(
                    classes=["table", "table-sm", "table-striped", "table-hover"], 
                    border=0 # Let Bootstrap handle borders
                )
            except Exception as e:
                 logger.error(f"Error styling summary statistics table: {e}", exc_info=True)
                 styled_tables_html['summary_statistics'] = "<p class='text-danger'>Error generating summary table.</p>"
        else:
             styled_tables_html['summary_statistics'] = "<p class='text-muted'>No summary statistics available.</p>"


        # Table 2: Detailed Metrics per Sample
        if not metrics_df.empty:
             try:
                 # Select relevant columns, maybe round numerics
                 detailed_df = metrics_df.copy()
                 numeric_cols_detail = detailed_df.select_dtypes(include=np.number).columns
                 if not numeric_cols_detail.empty:
                     detailed_df[numeric_cols_detail] = detailed_df[numeric_cols_detail].round(3)
                 
                 # Select columns for display (example: exclude helper columns or error columns)
                 cols_to_display = [
                     col for col in detailed_df.columns 
                     if col not in ['sample_id_str'] and 'error' not in col.lower() and 'path' not in col.lower()
                 ] 
                 # Ensure sample_id is first?
                 if 'sample_id' in cols_to_display:
                     cols_to_display.insert(0, cols_to_display.pop(cols_to_display.index('sample_id')))
                 
                 styler = detailed_df[cols_to_display].style.format(float_format, subset=numeric_cols_detail)
                 # Add other styling if needed (e.g., gradients based on values)
                 
                 styled_tables_html['detailed_metrics'] = styler.to_html(
                     classes=["table", "table-sm", "table-striped", "table-hover"], 
                     index=False, # Don't show DataFrame index
                     border=0
                 )
             except Exception as e:
                 logger.error(f"Error styling detailed metrics table: {e}", exc_info=True)
                 styled_tables_html['detailed_metrics'] = "<p class='text-danger'>Error generating detailed metrics table.</p>"
        else:
            styled_tables_html['detailed_metrics'] = "<p class='text-muted'>No detailed metrics available.</p>"

        logger.info("Finished preparing styled tables.")
        return styled_tables_html

    def _handle_non_serializable_floats(self, data: Any) -> Any:
        """Recursively converts non-serializable floats (NaN, Inf) to None in nested dicts/lists."""
        if isinstance(data, dict):
            return {k: self._handle_non_serializable_floats(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._handle_non_serializable_floats(item) for item in data]
        elif isinstance(data, float):
            if pd.isna(data) or np.isinf(data):
                return None # Convert NaN/Inf to None 
            return data
        # Handle numpy types that might remain if not converted earlier
        elif isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.bool_): return bool(obj)
        else:
            return data # Return other types as is

    def _generate_reports(self, report_context: Dict[str, Any], summary_dir: Path):
        """
        Generates report files based on the configuration.
        """
        logger.info(f"Generating reports in {summary_dir} using resolved config: {self.report_config}") # Log the config being used
        output_formats = self.report_config.get('output_formats', [])

        # --- Save Context Snapshot --- 
        context_json_path = summary_dir / "report_context_snapshot.json"
        try:
            serializable_context = {}
            for key, value in report_context.items():
                if key not in ['styled_tables_html']: # Exclude potentially complex/large items
                     try:
                         # Attempt to serialize, using NumpyJSONEncoder
                         json.dumps({key: value}, cls=NumpyJSONEncoder) 
                         serializable_context[key] = value
                     except TypeError:
                          logger.warning(f"Could not serialize key '{key}' for context snapshot. Skipping.")
            
            with open(context_json_path, 'w') as f:
                json.dump(serializable_context, f, indent=2, cls=NumpyJSONEncoder)
            logger.info(f"Saved report context snapshot to {context_json_path}")
        except Exception as e:
            logger.error(f"Failed to save report context JSON snapshot: {e}", exc_info=True)
            
        # --- Save Intermediate CSVs --- 
        summary_stats_data = report_context.get('summary_stats_dict', {}).get('metric_summaries')
        if summary_stats_data:
             try:
                 summary_df = pd.DataFrame(summary_stats_data)
                 summary_csv_path = summary_dir / "summary_statistics.csv"
                 summary_df.to_csv(summary_csv_path)
                 logger.info(f"Saved summary statistics table to {summary_csv_path}")
             except Exception as e:
                 logger.error(f"Failed to save summary statistics CSV: {e}", exc_info=True)
                 
        logger.info("Skipping saving detailed metrics CSV in _generate_reports (use consolidated_metrics.json).")

        # --- Generate HTML Report (Conditional) ---
        html_report_path: Optional[Path] = None
        if "html" in output_formats:
            try:
                html_report_path = self.generate_html_report(report_context, summary_dir)
                if html_report_path:
                     logger.info(f"Successfully generated HTML report: {html_report_path}")
                else:
                     logger.warning("HTML report generation failed or was skipped.")
            except Exception as e:
                logger.error(f"Error generating HTML report: {e}", exc_info=True)
        else:
             logger.info("Skipping HTML report generation based on configuration.")

        # --- Generate PDF Report (Conditional) ---
        if "pdf" in output_formats:
            # PDF generation depends on HTML existing
            if html_report_path and html_report_path.is_file():
                try:
                    pdf_path = self.generate_pdf_report(html_report_path, summary_dir)
                    if pdf_path:
                        logger.info(f"Successfully generated PDF report: {pdf_path}")
                    else:
                        logger.warning("PDF report generation failed or was skipped.")
                except Exception as e:
                     logger.error(f"Error generating PDF report: {e}", exc_info=True)
            elif "html" in output_formats:
                 logger.warning(f"HTML report file not found or failed ({html_report_path}), cannot generate PDF.")
            else:
                 # If only PDF was requested, we need to generate HTML first anyway
                 logger.info("PDF only requested; generating temporary HTML for conversion...")
                 temp_html_path = None
                 try:
                     temp_html_path = self.generate_html_report(report_context, summary_dir)
                     if temp_html_path and temp_html_path.is_file():
                         pdf_path = self.generate_pdf_report(temp_html_path, summary_dir)
                         if pdf_path:
                             logger.info(f"Successfully generated PDF report (from temp HTML): {pdf_path}")
                         else:
                             logger.warning("PDF report generation failed or was skipped.")
                     else:
                          logger.error("Failed to generate temporary HTML for PDF conversion.")
                 except Exception as e:
                      logger.error(f"Error generating PDF report via temporary HTML: {e}", exc_info=True)
                 finally:
                     # Clean up temporary HTML if it was created
                     if temp_html_path and temp_html_path.exists() and temp_html_path != html_report_path: 
                         try:
                             temp_html_path.unlink()
                             logger.info(f"Removed temporary HTML file: {temp_html_path}")
                         except OSError as unlink_e:
                              logger.warning(f"Failed to remove temporary HTML file {temp_html_path}: {unlink_e}")
        else:
            logger.info("Skipping PDF report generation based on configuration.")
            
        logger.info("Finished report generation process.")

    # --- Plotting Helper Methods ---
    def _plot_matplotlib_bar(self, x_data: List[Any], y_data: List[float], title: str, xlabel: str, ylabel: str) -> Optional[str]:
        """Generates a Matplotlib bar chart and returns it as a base64 PNG string."""
        if not x_data or not y_data or len(x_data) != len(y_data):
             logger.warning(f"Invalid data for Matplotlib bar chart '{title}'. Skipping.")
             return None
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x_data, y_data, color=sns.color_palette("viridis", len(x_data)))
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
             logger.error(f"Error generating Matplotlib bar chart '{title}': {e}", exc_info=True)
             return None

    def _plot_plotly_scatter(self, x_data: List[float], y_data: List[float], labels: List[str], title: str, xlabel: str, ylabel: str) -> Optional[str]:
        """Generates an interactive Plotly scatter plot and returns the HTML div string."""
        if not all([x_data, y_data, labels]) or not (len(x_data) == len(y_data) == len(labels)):
            logger.warning(f"Invalid data for Plotly scatter plot '{title}'. Skipping.")
            return None
        try:
            fig = px.scatter(
                x=x_data, 
                y=y_data, 
                text=labels, # Show sample_id on hover
                title=title,
                labels={ 'x': xlabel, 'y': ylabel }
            )
            fig.update_traces(textposition='top center')
            fig.update_layout(
                 height=500, 
                 margin=dict(l=20, r=20, t=50, b=20),
                 hovermode="closest"
            )
            # include_plotlyjs='cdn' makes the HTML self-contained for Plotly JS
            return fig.to_html(full_html=False, include_plotlyjs='cdn') 
        except Exception as e:
             logger.error(f"Error generating Plotly scatter plot '{title}': {e}", exc_info=True)
             return None

    # --- HTML Report Generation --- 
    def generate_html_report(self, report_context: Dict[str, Any], summary_dir: Path) -> Optional[Path]:
        """
        Generates the HTML report using Jinja2 and prepared context data, considering config.
        """
        logger.info("Generating HTML report...")
        if not self.jinja_env:
            logger.error("Jinja2 environment not initialized. Cannot generate HTML report.")
            return None
            
        try:
            template = self.jinja_env.get_template("report_template.html")
        except Exception as e:
             logger.error(f"Failed to load Jinja2 template 'report_template.html': {e}", exc_info=True)
             return None

        # Prepare final context specifically for the template
        template_context = {
            'run_metadata': report_context.get('run_metadata', {}),
            'summary_stats': report_context.get('summary_stats_dict', {}), # Pass summary stats dict
            'tables': report_context.get('styled_tables_html', {}),
            'failed_samples': report_context.get('failed_samples', []),
            # Config related items for template logic
            'included_sections': self.report_config.get('included_sections', list(DEFAULT_REPORT_CONFIG['included_sections'])),
            'theme': self.report_config.get('theme', DEFAULT_REPORT_CONFIG['theme'])
        }
        
        # Generate required plots based on config and add them to the context
        template_context['plots'] = {}
        viz_data = report_context.get('visualization_data', {})
        included_plots_config = self.report_config.get("included_plots", "all")
        should_include_plot = lambda plot_key: included_plots_config == "all" or plot_key in included_plots_config

        # Example: Generate accuracy bar chart (Matplotlib)
        plot_key_accuracy = "accuracy_per_sample"
        if should_include_plot(plot_key_accuracy):
            bar_data = viz_data.get('bar_charts', {}).get(plot_key_accuracy)
            if bar_data:
                template_context['plots']['accuracy_bar'] = self._plot_matplotlib_bar(
                    x_data=bar_data.get('sample_id_str', []),
                    y_data=bar_data.get('accuracy', []),
                    title="Accuracy per Sample",
                    xlabel="Sample ID",
                    ylabel="Accuracy"
                )
            
        # Example: Generate MSE vs R2 scatter plot (Plotly)
        plot_key_scatter = "mse_vs_r2"
        if should_include_plot(plot_key_scatter):
            scatter_data = viz_data.get('scatter_plots', {}).get(plot_key_scatter)
            if scatter_data:
                 template_context['plots']['mse_r2_scatter'] = self._plot_plotly_scatter(
                     x_data=scatter_data.get('mse', []),
                     y_data=scatter_data.get('r2', []),
                     labels=scatter_data.get('sample_id', []),
                     title="MSE vs R² per Sample",
                     xlabel="Mean Squared Error (MSE)",
                     ylabel="R² Score"
                 )

        # Render the template
        try:
            html_content = template.render(template_context)
        except Exception as e:
            logger.error(f"Failed to render Jinja2 template: {e}", exc_info=True)
            return None

        # Save to file
        timestamp = report_context.get('run_metadata', {}).get('timestamp', datetime.now()).strftime('%Y%m%d_%H%M%S')
        output_path = summary_dir / f"epibench_report_{timestamp}.html"
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"HTML report saved to: {output_path}")
            return output_path
        except IOError as e:
            logger.error(f"Failed to write HTML report to {output_path}: {e}", exc_info=True)
            return None
            
    # --- PDF Report Generation --- 
    def generate_pdf_report(self, html_report_path: Path, summary_dir: Path) -> Optional[Path]:
        """
        Generates a PDF report from an existing HTML report file using WeasyPrint.

        Args:
            html_report_path: Path to the generated HTML report file.
            summary_dir: Directory where the PDF report will be saved.

        Returns:
            Path to the generated PDF file, or None if generation failed.
        """
        logger.info(f"Generating PDF report from: {html_report_path}")
        
        try:
            # Check if WeasyPrint is available (optional, raises ImportError if not)
            import weasyprint 
        except ImportError:
            logger.error("WeasyPrint library not found. Cannot generate PDF report. Please install it (`pip install WeasyPrint`).")
            return None
            
        # Define paths
        pdf_output_path = html_report_path.with_suffix('.pdf')
        css_path = Path(__file__).parent / "templates" / "report_print.css"
        
        if not css_path.is_file():
            logger.warning(f"Print CSS file not found at {css_path}. PDF styling might be incorrect.")
            stylesheets = None
        else:
            try:
                 stylesheets = [weasyprint.CSS(filename=str(css_path))]
            except Exception as css_e:
                 logger.error(f"Failed to load print CSS from {css_path}: {css_e}", exc_info=True)
                 stylesheets = None # Proceed without print CSS

        # Read HTML content
        try:
            with open(html_report_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except IOError as e:
             logger.error(f"Failed to read HTML report file {html_report_path}: {e}", exc_info=True)
             return None

        # Generate PDF
        try:
            html = weasyprint.HTML(string=html_content, base_url=str(summary_dir))
            # Note: font_config and optimize_size can be added for finer control
            html.write_pdf(str(pdf_output_path), stylesheets=stylesheets)
            logger.info(f"PDF report saved to: {pdf_output_path}")
            return pdf_output_path
        except Exception as e:
            logger.error(f"WeasyPrint failed to generate PDF: {e}", exc_info=True)
            # Log specific WeasyPrint errors if helpful
            if hasattr(e, 'message'):
                 logger.error(f"WeasyPrint error detail: {e.message}")
            return None
            
    def _summarize_run_status(self, summary_dir: Path):
        """Generates a summary of sample statuses based on checkpoint data."""
        logger.info("Generating run status summary...")
        status_summary = {}
        total_samples = len(self.checkpoint_data)
        status_counts = {'completed': 0, 'failed': 0, 'pending': 0, 'other': 0}
        
        failed_samples = []
        
        for sample_id, data in self.checkpoint_data.items():
            status = data.get('status')
            if status == 'completed':
                status_counts['completed'] += 1
            elif status == 'failed':
                 status_counts['failed'] += 1
                 failed_samples.append(sample_id)
            elif status is None or status == 'pending':
                 status_counts['pending'] += 1
            else:
                 status_counts['other'] += 1
                 
        status_summary['total_samples_in_checkpoint'] = total_samples
        status_summary['status_counts'] = status_counts
        status_summary['failed_sample_ids'] = failed_samples
        
        summary_file = summary_dir / "run_status_summary.json"
        try:
            with open(summary_file, 'w') as f:
                json.dump(status_summary, f, indent=4)
            logger.info(f"Saved run status summary to {summary_file}")
        except IOError as e:
            logger.error(f"Failed to save run status summary: {e}")


# Example usage (called potentially from a main script or after pipeline execution)
# if __name__ == "__main__":
#     # This would require loading the main config and checkpoint data first
#     # base_output = Path("path/to/your/output_directory")
#     # checkpoint_path = base_output / "pipeline_checkpoint.json" 
#     
#     # try:
#     #     # validated_config = validate_process_config(str(main_cfg_path)) # No longer need main config validation here
#     #     with open(checkpoint_path, 'r') as f:
#     #         ckpt_data = json.load(f)
#     # except Exception as e:
#     #     print(f"Error loading checkpoint: {e}")
#     #     exit(1)
#         
#     # collector = ResultsCollector(base_output_directory=base_output, checkpoint_data=ckpt_data)
#     # collected_results = collector.collect_all() 
#     # print(f"Collection finished. Summary results: {collected_results}") 
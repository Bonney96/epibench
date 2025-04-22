import logging
from pathlib import Path
import pandas as pd
import numpy as np # Ensure numpy is imported
from typing import List, Dict, Any, Optional
import json
# import yaml # Removed unused import
import glob # Import glob for finding files

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
        # Let the base class default method raise the TypeError for other types
        return super(NumpyJSONEncoder, self).default(obj)


class ResultsCollector:
    """
    Collects, aggregates, and reports results from pipeline execution across multiple samples.
    """

    def __init__(self, base_output_directory: Path, checkpoint_data: Dict[str, Any]):
        """
        Initializes the ResultsCollector.

        Args:
            base_output_directory: The root directory where all pipeline outputs are stored.
            checkpoint_data: Dictionary containing the status of each processed sample.
        """
        # self.main_config = main_config # Removed main_config
        self.checkpoint_data = checkpoint_data
        # --- Use base_output_directory directly ---
        self.base_output_directory = Path(base_output_directory) # Ensure it's a Path object
        logger.info(f"ResultsCollector initialized for output directory: {self.base_output_directory}")

    def collect_all(self) -> Dict[str, Any]:
        """
        Orchestrates the collection, aggregation, and reporting process for all samples.
        Returns the aggregated data dictionary.
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
        
        # 4. Generate summary statistics
        summary_stats = self._calculate_summary_stats(aggregated_data)

        # 5. Generate reports
        self._generate_reports(aggregated_data, summary_stats, summary_dir)

        # 6. Summarize success/failure
        self._summarize_run_status(summary_dir)

        logger.info("Results collection finished.")
        # Return the aggregated data as planned for now
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
                        all_results["metrics"][sample_id] = json.load(f)
                        logger.debug(f"Loaded metrics for {sample_id} from {metrics_file}")
                except (IOError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to load or parse metrics file {metrics_file} for sample {sample_id}: {e}")
                    all_results["metrics"][sample_id] = {"error": "failed_to_load", "path": str(metrics_file)}
            else:
                logger.warning(f"Metrics file not found for completed sample {sample_id} at {metrics_file}")
                all_results["metrics"][sample_id] = {"error": "not_found", "path": str(metrics_file)}

            # 2. Find Predictions File Path
            predictions_file = predict_dir / "predictions.csv" # Assumed file name
            if predictions_file.is_file():
                 all_results["prediction_files"][sample_id] = str(predictions_file)
                 logger.debug(f"Found predictions file for {sample_id} at {predictions_file}")
                 # Optionally load the CSV here if needed for stats, but could be large
                 # try:
                 #     pred_df = pd.read_csv(predictions_file)
                 #     # Store df or summary stats
                 # except Exception as e:
                 #     logger.warning(f"Failed to load predictions csv {predictions_file}: {e}")
            else:
                 logger.warning(f"Predictions file not found for completed sample {sample_id} at {predictions_file}")
                 all_results["prediction_files"][sample_id] = {"error": "not_found", "path": str(predictions_file)}

            # 3. Training Log Info (Simplified)
            # The trainer primarily logs to TensorBoard. Parsing TensorBoard logs is out of scope.
            # Log a warning and point to the assumed directory structure.
            # --- Use assumed default training output directory name ---
            train_log_dir = sample_output_dir / 'training_output' # Assume default name
            logger.warning(f"Skipping detailed training log collection for {sample_id}. "
                           f"Check TensorBoard logs potentially within '{train_log_dir}' for training progress.")
            # Initialize the key to provide consistent structure
            all_results["training_logs"][sample_id] = {"status": "skipped", "reason": "TensorBoard format not parsed", "assumed_dir": str(train_log_dir)}

            # Add logic for other expected outputs if necessary...

        # Consolidate all metrics into a single DataFrame for easier reporting later
        # Derive summary_dir here for the helper call
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

    def _calculate_summary_stats(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates overall summary statistics from the aggregated metrics.

        Args:
            aggregated_data: The dictionary returned by _aggregate_sample_results.

        Returns:
            A dictionary containing summary statistics.
        """
        logger.info("Calculating summary statistics...")
        summary = {}
        
        all_metrics = aggregated_data.get("metrics", {})
        # Filter out samples where metrics loading failed
        valid_metrics_list = [
            m for m in all_metrics.values() 
            if isinstance(m, dict) and "error" not in m
        ]

        if not valid_metrics_list:
            summary['message'] = "No valid metrics found to summarize."
            logger.warning(summary['message'])
            return summary

        # Create a DataFrame for easier calculation
        try:
            df = pd.DataFrame(valid_metrics_list)
            # Identify numeric columns dynamically
            numeric_cols = df.select_dtypes(include=np.number).columns
            
            if not numeric_cols.empty:
                logger.info(f"Calculating stats for numeric metrics: {list(numeric_cols)}")
                summary['num_samples_with_metrics'] = len(df)
                # Calculate statistics and convert to dict for JSON compatibility
                summary['metric_summaries'] = df[numeric_cols].agg(
                    ['mean', 'median', 'min', 'max', 'std', 'count']
                ).to_dict()
                # Ensure NaN/Infinity are handled for JSON
                summary['metric_summaries'] = self._handle_non_serializable_floats(summary['metric_summaries'])

            else:
                 summary['message'] = "No numeric metrics found in the aggregated data."
                 logger.warning(summary['message'])
                 
        except pd.errors.EmptyDataError:
            # pass # Removed pass
            summary['message'] = "Aggregated metrics resulted in an empty DataFrame."
            logger.warning(summary['message'])
        except Exception as e:
            logger.error(f"Error calculating summary statistics using pandas: {e}", exc_info=True)
            summary['error'] = f"Failed to calculate summary stats: {e}"
        
        logger.info("Finished calculating summary statistics.")
        return summary

    def _handle_non_serializable_floats(self, data: Any) -> Any:
        """Recursively converts non-serializable floats (NaN, Inf) to None in nested dicts/lists."""
        if isinstance(data, dict):
            return {k: self._handle_non_serializable_floats(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._handle_non_serializable_floats(item) for item in data]
        elif isinstance(data, float):
            if pd.isna(data) or np.isinf(data):
                return None # Convert NaN/Inf to None for JSON
            return data
        else:
            return data # Return other types as is

    def _generate_reports(self, aggregated_data: Dict[str, Any], summary_stats: Dict[str, Any], summary_dir: Path):
        """
        Generates various report files (CSV, JSON, potentially HTML/PDF) in the summary directory.

        Placeholder: Needs specific report generation logic.

        Args:
            aggregated_data: Aggregated data from samples.
            summary_stats: Calculated summary statistics.
            summary_dir: The directory to save reports into.
        """
        logger.info(f"Generating reports in {summary_dir}...")

        # 1. Save raw aggregated metrics (JSON)
        metrics_data = aggregated_data.get("metrics", {})
        if metrics_data:
            metrics_json_path = summary_dir / "aggregated_metrics_raw.json"
            try:
                with open(metrics_json_path, 'w') as f:
                    # Use NumpyJSONEncoder
                    json.dump(metrics_data, f, indent=4, cls=NumpyJSONEncoder) 
                logger.info(f"Saved raw aggregated metrics to {metrics_json_path}")
            except Exception as e: # Broaden exception catch
                logger.error(f"Failed to save raw aggregated metrics JSON: {e}", exc_info=True)
            # Consolidated CSV is saved in _aggregate_sample_results via helper

        # 2. Save summary statistics
        summary_json_path = summary_dir / "summary_statistics.json"
        try:
            with open(summary_json_path, 'w') as f:
                 # Use NumpyJSONEncoder here too
                json.dump(summary_stats, f, indent=4, cls=NumpyJSONEncoder) 
            logger.info(f"Saved summary statistics to {summary_json_path}")
        except Exception as e: # Broaden exception catch
            logger.error(f"Failed to save summary statistics JSON: {e}", exc_info=True)
            
        # 3. Generate HTML/PDF report (Placeholder)
        logger.warning("HTML/PDF report generation is not yet implemented.")

        logger.info("Finished generating reports.")
        
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
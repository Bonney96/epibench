# -*- coding: utf-8 -*-
"""Framework for comparing models trained on different sample groups."""

import logging
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
# Add imports for models, data loaders, evaluation metrics as needed
from ..config.config_manager import ConfigManager
from ..models import models # Import model registry
from ..training.trainer import Trainer # For loading models
from ..data.data_loader import create_dataloaders 
from ..evaluation import calculate_regression_metrics # Assuming this function exists and returns a dict
from ..utils.io_utils import ensure_dir, load_predictions, save_results # Corrected: utils instead of util
import torch
from tqdm import tqdm
import os
import json
from scipy import stats # Added import

logger = logging.getLogger(__name__)

class ComparativeAnalyzer:
    """
    Analyzes and compares the performance of models trained or evaluated 
    on different sample groups (e.g., different cell types, conditions).
    """

    def __init__(self, 
                 model_configs: List[Dict[str, Any]], 
                 sample_group_data: Dict[str, Dict[str, str]],
                 base_output_dir: str):
        """
        Initializes the ComparativeAnalyzer.

        Args:
            model_configs: A list of configuration dictionaries, one for each 
                           model/training setup to compare. Each config should 
                           contain details needed to load the model and data.
            sample_group_data: A dictionary where keys are group names (e.g., 'AML', 'CD34') 
                               and values are dictionaries specifying paths to data 
                               relevant to that group (e.g., {'train': 'path/to/aml_train.h5', 
                               'test': 'path/to/aml_test.h5'}).
            base_output_dir: The base directory where analysis results (reports, plots) 
                             will be saved. Subdirectories for each analysis might be created.
        """
        logger.info("Initializing ComparativeAnalyzer...")
        if not model_configs:
            raise ValueError("At least one model configuration must be provided.")
        if not sample_group_data:
             raise ValueError("Sample group data must be provided.")
             
        self.model_configs = model_configs
        self.sample_group_data = sample_group_data
        self.base_output_dir = base_output_dir
        
        # Placeholder for storing results
        self.results: Dict[str, Any] = {} 

        logger.info(f"Analyzer initialized with {len(model_configs)} model configs and {len(sample_group_data)} sample groups.")
        logger.debug(f"Model Configs: {model_configs}")
        logger.debug(f"Sample Group Data: {sample_group_data}")
        logger.debug(f"Base Output Directory: {base_output_dir}")

    def run_cross_sample_evaluation(self, metrics_to_compute: List[str]) -> Dict[str, Any]:
        """
        Performs cross-sample evaluation: loads models based on configs 
        and evaluates them on the test sets of specified sample groups.

        Args:
            metrics_to_compute: A list of metric names (strings) to calculate 
                                (e.g., ['mse', 'r2', 'pearson']). 
                                Assumes `calculate_regression_metrics` handles these.

        Returns:
            A dictionary containing evaluation results, structured by model config index 
            and sample group name. Example: 
            { 
                'model_0': { 
                    'AML': {'mse': 0.1, 'r2': 0.8}, 
                    'CD34': {'mse': 0.15, 'r2': 0.75} 
                }, 
                # ... other models
            }
        """
        logger.info("Running cross-sample evaluation...")
        cross_eval_results = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device} for evaluation")

        for idx, model_config_entry in enumerate(self.model_configs):
            model_key = f"model_{idx}"
            logger.info(f"--- Evaluating {model_key} --- ")
            cross_eval_results[model_key] = {}
            
            try:
                # --- Load Model and its Training Config --- 
                config_path = model_config_entry.get('config')
                checkpoint_path = model_config_entry.get('checkpoint')
                if not config_path or not checkpoint_path:
                     logger.warning(f"Skipping {model_key}: Missing 'config' or 'checkpoint' path.")
                     continue
                     
                logger.info(f"Loading training config for {model_key} from: {config_path}")
                cfg_manager = ConfigManager(config_path)
                train_config = cfg_manager.get_config()
                
                model_name = train_config.get('model', {}).get('name')
                model_params = train_config.get('model', {}).get('params', {})
                if not model_name:
                    logger.warning(f"Skipping {model_key}: Model name not found in config {config_path}.")
                    continue
                    
                ModelClass = models.get_model(model_name)
                model_instance = ModelClass(**model_params)
                
                logger.info(f"Loading model weights for {model_key} from: {checkpoint_path}")
                model_instance, _, _, _ = Trainer.load_model(
                    checkpoint_path=checkpoint_path, 
                    model=model_instance, 
                    device=device,
                    load_optimizer_state=False,
                    load_scheduler_state=False
                )
                model_instance.eval() # Set to evaluation mode

                # --- Evaluate on Each Sample Group's Test Data --- 
                for group_name, group_data_paths in self.sample_group_data.items():
                     test_data_path = group_data_paths.get('test')
                     if not test_data_path:
                         logger.warning(f"Skipping evaluation for group '{group_name}' on {model_key}: No 'test' data path specified.")
                         continue
                         
                     logger.info(f"Evaluating {model_key} on test data for group: '{group_name}' ({test_data_path})")
                     
                     try:
                         # --- Load Test Data --- 
                         data_loader_config = train_config.get('data', {}).copy() # Use training config for data params
                         data_loader_config['test_path'] = test_data_path # Override with specific test path
                         data_loader_config['shuffle_test'] = False # Ensure no shuffling for eval
                         # Adjust batch size if needed, e.g., allow override
                         # data_loader_config['batch_size'] = eval_batch_size or data_loader_config.get('batch_size', 32)

                         dataloaders = create_dataloaders(data_loader_config, splits=['test'])
                         test_loader = dataloaders.get('test')

                         if not test_loader:
                             logger.warning(f"Could not create test loader for group '{group_name}'. Skipping.")
                             continue

                         # --- Run Predictions --- 
                         all_preds = []
                         all_targets = []
                         with torch.no_grad():
                            for batch in tqdm(test_loader, desc=f"Predicting {group_name}", leave=False):
                                # Assuming loader yields (inputs, targets) or similar
                                # Adapt based on actual data loader structure
                                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                                     inputs = batch[0].to(device)
                                     targets = batch[1].to(device)
                                else:
                                     logger.warning(f"Unexpected batch format for group '{group_name}'. Skipping batch.")
                                     continue # Or handle differently
                                
                                preds = model_instance(inputs)
                                all_preds.append(preds.cpu().numpy())
                                all_targets.append(targets.cpu().numpy())
                         
                         if not all_preds:
                             logger.warning(f"No predictions generated for group '{group_name}'. Skipping metric calculation.")
                             continue

                         all_preds_np = np.concatenate(all_preds, axis=0)
                         all_targets_np = np.concatenate(all_targets, axis=0)

                         # --- Calculate Metrics --- 
                         logger.debug(f"Calculating metrics for {model_key} on '{group_name}': {metrics_to_compute}")
                         # Assuming calculate_regression_metrics takes numpy arrays and returns a dict
                         metrics_results = calculate_regression_metrics(
                             y_true=all_targets_np,
                             y_pred=all_preds_np,
                             metrics=metrics_to_compute
                         )
                         
                         cross_eval_results[model_key][group_name] = metrics_results
                         logger.info(f"Metrics for {model_key} on '{group_name}': {metrics_results}")

                     except FileNotFoundError:
                        logger.error(f"Test data file not found for group '{group_name}': {test_data_path}")
                     except Exception as eval_err:
                         logger.error(f"Error evaluating {model_key} on group '{group_name}': {eval_err}", exc_info=True)
                         # Optionally store error info in results
                         cross_eval_results[model_key][group_name] = {"error": str(eval_err)}
                         
            except FileNotFoundError as fnf_err:
                 logger.error(f"File not found during setup for {model_key}: {fnf_err}")
                 cross_eval_results[model_key]["error"] = f"File not found: {fnf_err}"
            except Exception as model_err:
                 logger.error(f"Error processing model config {idx} ({model_key}): {model_err}", exc_info=True)
                 cross_eval_results[model_key]["error"] = str(model_err)

        self.results['cross_sample_eval'] = cross_eval_results
        logger.info("Cross-sample evaluation finished.")
        # logger.debug(f"Cross-evaluation results: {cross_eval_results}") # Can be very verbose
        return cross_eval_results

    def perform_statistical_tests(self, metric='mse'):
        """
        Performs statistical comparisons of model performance across sample groups based on aggregated metrics.

        Note: This method currently compares single aggregate metric values (e.g., mean MSE).
        Robust statistical testing (e.g., t-tests, ANOVA, Mann-Whitney U) typically requires
        multiple data points (samples) per group/model being compared (e.g., scores per instance,
        metrics per cross-validation fold or bootstrap sample). The current implementation
        provides basic difference reporting as a placeholder and highlights where proper
        statistical tests should be implemented if sample data becomes available.

        Args:
            metric (str): The evaluation metric stored in `self.evaluation_results` to use
                          for comparison (e.g., 'mse', 'mae'). Defaults to 'mse'.

        Returns:
            dict: A dictionary containing the results of the comparisons.
                  Keys describe the comparison (e.g., 'modelA_group1_vs_group2').
                  Values contain the metrics, their difference, and placeholders for p-value/statistic.
                  Returns None if evaluation results are missing.
        """
        logger.info(f"Performing statistical comparisons based on aggregated metric: {metric}")
        test_results = {}

        eval_results = self.results.get('cross_sample_eval')
        if not eval_results or not isinstance(eval_results, dict):
            logger.error("Cross-sample evaluation results not found or invalid. Cannot perform statistical comparisons.")
            self.statistical_test_results = None
            return None

        model_keys = list(eval_results.keys())
        if not model_keys or any("error" in eval_results[mk] for mk in model_keys if isinstance(eval_results[mk], dict)):
             logger.warning("Evaluation results contain errors or no models were evaluated. Statistical comparisons may be incomplete.")
             # Proceed cautiously, comparisons might fail

        sample_groups = list(self.sample_group_data.keys())

        try:
            # 1. Compare each model across pairs of sample groups using the aggregate metric
            if len(sample_groups) >= 2:
                logger.info("Comparing models across sample group pairs...")
                for model_name in model_keys:
                    if not isinstance(eval_results.get(model_name), dict):
                        logger.warning(f"Evaluation results for model {model_name} are not in the expected format. Skipping cross-group comparison.")
                        continue

                    for i in range(len(sample_groups)):
                        for j in range(i + 1, len(sample_groups)):
                            group1 = sample_groups[i]
                            group2 = sample_groups[j]

                            # Check if metrics exist for both groups for the current model
                            metric1_data = eval_results.get(model_name, {}).get(group1, {})
                            metric2_data = eval_results.get(model_name, {}).get(group2, {})

                            if not isinstance(metric1_data, dict) or not isinstance(metric2_data, dict):
                                 logger.warning(f"Metric data for model {model_name} in group {group1} or {group2} is not a dictionary. Skipping.")
                                 continue

                            metric1 = metric1_data.get(metric)
                            metric2 = metric2_data.get(metric)


                            if metric1 is not None and metric2 is not None:
                                test_key = f"{model_name}_{group1}_vs_{group2}"
                                difference = metric1 - metric2
                                # Placeholder for actual statistical test
                                # Example: stats.ttest_ind(sample1, sample2) or stats.mannwhitneyu(sample1, sample2)
                                # Requires `sample1` and `sample2` to be arrays/lists of scores, not single aggregate values.
                                test_results[test_key] = {
                                    f'{metric}_{group1}': metric1,
                                    f'{metric}_{group2}': metric2,
                                    'difference': difference,
                                    'test_type': 'Comparison (Placeholder - Requires Sample Data for Stat Test)',
                                    'p_value': None, # Placeholder
                                    'statistic': None # Placeholder
                                }
                                logger.debug(f"Comparison {test_key}: Diff={difference:.4f} ({metric})")
                                logger.warning(f"Comparison for {test_key} based on aggregate '{metric}'. Statistical test requires sample data.")
                            else:
                                logger.warning(f"Missing metric '{metric}' for model {model_name} in group {group1} or {group2}. Skipping comparison.")

            # 2. Compare pairs of models within each sample group using the aggregate metric
            if len(model_keys) >= 2:
                logger.info("Comparing model pairs within each sample group...")
                for group_name in sample_groups:
                     for i in range(len(model_keys)):
                         for j in range(i + 1, len(model_keys)):
                             model1_name = model_keys[i]
                             model2_name = model_keys[j]

                             # Check if metrics exist for both models for the current group
                             metric1_data = eval_results.get(model1_name, {}).get(group_name, {})
                             metric2_data = eval_results.get(model2_name, {}).get(group_name, {})

                             if not isinstance(metric1_data, dict) or not isinstance(metric2_data, dict):
                                 logger.warning(f"Metric data for model {model1_name} or {model2_name} in group {group_name} is not a dictionary. Skipping.")
                                 continue

                             metric1 = metric1_data.get(metric)
                             metric2 = metric2_data.get(metric)


                             if metric1 is not None and metric2 is not None:
                                 test_key = f"{model1_name}_vs_{model2_name}_on_{group_name}"
                                 difference = metric1 - metric2
                                 # Placeholder for actual statistical test
                                 # Example: stats.ttest_rel(sample1, sample2) or stats.wilcoxon(sample1, sample2)
                                 # Requires paired samples (e.g., scores from both models on the same instances/folds).
                                 test_results[test_key] = {
                                     f'{metric}_{model1_name}': metric1,
                                     f'{metric}_{model2_name}': metric2,
                                     'difference': difference,
                                     'test_type': 'Comparison (Placeholder - Requires Sample Data for Stat Test)',
                                     'p_value': None, # Placeholder
                                     'statistic': None # Placeholder
                                 }
                                 logger.debug(f"Comparison {test_key}: Diff={difference:.4f} ({metric})")
                                 logger.warning(f"Comparison for {test_key} based on aggregate '{metric}'. Statistical test requires sample data.")
                             else:
                                 logger.warning(f"Missing metric '{metric}' for model {model1_name} or {model2_name} in group {group_name}. Skipping comparison.")

            if not test_results:
                logger.warning(f"No statistical comparisons could be generated for metric '{metric}'. Check evaluation results and group/model availability.")

            logger.info(f"Statistical comparisons generated ({len(test_results)} comparisons, placeholders).")
            self.statistical_test_results = test_results
            # Save results if an output directory is defined
            if hasattr(self, 'base_output_dir') and self.base_output_dir:
                try:
                    stats_path = os.path.join(self.base_output_dir, 'statistical_test_results.json')
                    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
                    with open(stats_path, 'w') as f:
                        json.dump(test_results, f, indent=4)
                    logger.info(f"Statistical comparison results saved to {stats_path}")
                except Exception as e:
                    logger.error(f"Failed to save statistical comparison results: {e}", exc_info=True)

        except ImportError:
             logger.error("scipy is not installed. Cannot perform statistical comparisons. Run `pip install scipy`")
             self.statistical_test_results = {"error": "scipy not installed."}
        except Exception as e:
             logger.error(f"Error during statistical comparisons: {e}", exc_info=True)
             self.statistical_test_results = {"error": f"Error during comparisons: {e}"}

        return test_results

    def generate_comparative_report(self, output_format: str = 'json') -> str:
        """
        Generates a comparative report summarizing the evaluation and statistical test results.

        Args:
            output_format: The desired format for the report ('json', 'csv', 'md').

        Returns:
            The file path where the report was saved.
        """
        logger.info(f"Generating comparative report in format: {output_format}...")
        report_path = os.path.join(self.base_output_dir, f"comparative_report.{output_format}")
        ensure_dir(os.path.dirname(report_path)) # Ensure base dir exists

        eval_results = self.results.get('cross_sample_eval', {})
        stat_results = self.results.get('statistical_tests', {})

        try:
            if output_format == 'json':
                with open(report_path, 'w') as f:
                    json.dump(self.results, f, indent=2)
                logger.info(f"JSON report saved to: {report_path}")
                
            elif output_format == 'csv':
                # --- Generate CSV Report --- 
                report_data = []
                # Flatten the results for CSV
                for model_key, group_results in eval_results.items():
                    if isinstance(group_results, dict):
                        for group_name, metrics in group_results.items():
                             if isinstance(metrics, dict) and "error" not in metrics:
                                 row = {'model': model_key, 'sample_group': group_name}
                                 row.update(metrics) # Add all metrics as columns
                                 report_data.append(row)
                             elif isinstance(metrics, dict) and "error" in metrics:
                                  report_data.append({'model': model_key, 'sample_group': group_name, 'error': metrics["error"]})
                    elif model_key == 'error': # Handle top-level model errors
                         report_data.append({'model': model_key, 'sample_group': 'N/A', 'error': group_results})
                         
                if not report_data:
                     logger.warning("No evaluation data available to generate CSV report.")
                     # Create empty file or file with headers?
                     with open(report_path, 'w') as f:
                          f.write("No evaluation data available.") # Placeholder content
                else:
                     df = pd.DataFrame(report_data)
                     # Reorder columns potentially
                     cols = ['model', 'sample_group'] + sorted([c for c in df.columns if c not in ['model', 'sample_group', 'error']])
                     if 'error' in df.columns:
                         cols.append('error')
                     df = df[cols]
                     df.to_csv(report_path, index=False)
                     logger.info(f"CSV report saved to: {report_path}")
                     
                     # Optionally, append statistical results summary to CSV or save separately
                     # For simplicity, stats are not included in this CSV structure yet.
                     
            elif output_format == 'md':
                 # --- Generate Markdown Report --- 
                 md_content = "# Comparative Analysis Report\n\n"
                 
                 md_content += "## Evaluation Results\n\n"
                 if eval_results:
                     # Convert eval results to markdown table (using pandas for simplicity)
                     report_data_md = []
                     for model_key, group_results in eval_results.items():
                          if isinstance(group_results, dict):
                              for group_name, metrics in group_results.items():
                                  if isinstance(metrics, dict) and "error" not in metrics:
                                       row = {'Model': model_key, 'Sample Group': group_name}
                                       row.update(metrics)
                                       report_data_md.append(row)
                                  elif isinstance(metrics, dict) and "error" in metrics:
                                       report_data_md.append({'Model': model_key, 'Sample Group': group_name, 'Status': f'Error: {metrics["error"]}'})
                          elif model_key == 'error':
                               report_data_md.append({'Model': model_key, 'Sample Group': 'N/A', 'Status': f'Error: {group_results}'})
                               
                     if report_data_md:
                         df_md = pd.DataFrame(report_data_md)
                         # Basic formatting for markdown table
                         md_content += df_md.to_markdown(index=False)
                         md_content += "\n\n"
                     else:
                          md_content += "No evaluation results available.\n\n"
                 else:
                     md_content += "No evaluation results available.\n\n"

                 md_content += "## Statistical Test Results (Placeholders)\n\n"
                 if stat_results:
                     for test_key, test_data in stat_results.items():
                         md_content += f"- **{test_key}**:\n"
                         if isinstance(test_data, dict):
                             for res_key, res_val in test_data.items():
                                  md_content += f"  - {res_key}: {res_val}\n"
                         else:
                             md_content += f"  - Result: {test_data}\n"
                     md_content += "\n*Note: Statistical tests shown are placeholders and require results from multiple runs/folds for validity.*\n"
                 else:
                     md_content += "No statistical test results available.\n"
                     
                 with open(report_path, 'w') as f:
                     f.write(md_content)
                 logger.info(f"Markdown report saved to: {report_path}")
                 
            else:
                logger.error(f"Unsupported report format: {output_format}. Supported formats: json, csv, md.")
                # Fallback or raise error?
                return None # Indicate failure
                
        except ImportError as imp_err:
             logger.error(f"Missing library required for format '{output_format}': {imp_err}. Try `pip install pandas` for csv/md.")
             return None
        except Exception as e:
            logger.error(f"Error generating report in format {output_format}: {e}", exc_info=True)
            return None
            
        return report_path

    def generate_visualizations(self, plot_metric: Optional[str] = None) -> List[str]:
        """
        Generates plots (e.g., box plots, heatmaps) to visualize comparative performance.

        Args:
            plot_metric: Optional specific metric name from evaluation results to plot.
                         If None, attempts to plot the first available metric.

        Returns:
            A list of file paths where the plots were saved.
        """
        logger.info("Generating visualizations...")
        plot_paths = []
        eval_results = self.results.get('cross_sample_eval', {})

        if not eval_results:
            logger.warning("No evaluation results found. Skipping visualization generation.")
            return plot_paths

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme(style="whitegrid") # Set a nice theme
            
            # --- Prepare data for plotting --- 
            plot_data_list = []
            available_metrics = set()
            for model_key, group_results in eval_results.items():
                if isinstance(group_results, dict):
                     for group_name, metrics in group_results.items():
                         if isinstance(metrics, dict) and "error" not in metrics:
                             for metric_name, metric_value in metrics.items():
                                 # Check if the metric value is suitable for plotting (numeric)
                                 if isinstance(metric_value, (int, float, np.number)):
                                      plot_data_list.append({
                                          'model': model_key,
                                          'sample_group': group_name,
                                          'metric_name': metric_name,
                                          'metric_value': metric_value
                                      })
                                      available_metrics.add(metric_name)
                                 # Add handling for list/array metrics if needed for boxplots

            if not plot_data_list:
                 logger.warning("No plottable metric data found in evaluation results. Skipping visualization.")
                 return plot_paths
            
            plot_df = pd.DataFrame(plot_data_list)
            
            # Determine which metric to plot
            metric_to_plot = plot_metric
            if not metric_to_plot:
                 if available_metrics:
                     metric_to_plot = sorted(list(available_metrics))[0] # Default to first metric
                     logger.info(f"No specific plot_metric provided, defaulting to plot: '{metric_to_plot}'")
                 else:
                      logger.warning("Could not determine a metric to plot.")
                      return plot_paths # No metric identified
            elif metric_to_plot not in available_metrics:
                 logger.error(f"Specified plot_metric '{metric_to_plot}' not found in evaluation results. Available: {available_metrics}")
                 return plot_paths
                 
            # Filter DataFrame for the chosen metric
            metric_df = plot_df[plot_df['metric_name'] == metric_to_plot]
            
            if metric_df.empty:
                logger.warning(f"No data available for metric '{metric_to_plot}' to generate plot.")
                return plot_paths

            # --- Generate Box Plot (Example) --- 
            # Compare metric across sample groups for each model
            # Or compare models within each sample group
            logger.info(f"Generating box plot for metric: '{metric_to_plot}'...")
            plt.figure(figsize=(10, 6)) # Adjust figure size as needed
            ax = sns.boxplot(data=metric_df, x='sample_group', y='metric_value', hue='model')
            ax.set_title(f"Comparison of '{metric_to_plot}' Across Sample Groups and Models")
            ax.set_xlabel("Sample Group")
            ax.set_ylabel(f"Metric Value ({metric_to_plot})")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plot_filename = f"comparative_boxplot_{metric_to_plot}.png"
            plot_path = os.path.join(self.base_output_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close() # Close the figure
            plot_paths.append(plot_path)
            logger.info(f"Box plot saved to: {plot_path}")

            # --- Add other plot types here (e.g., Heatmap, Scatter) ---
            # Placeholder for additional visualizations

        except ImportError:
            logger.warning("matplotlib or seaborn not installed. Cannot generate visualizations. Run `pip install matplotlib seaborn pandas`")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}", exc_info=True)
            # Ensure plots are closed if error occurs mid-generation
            try: plt.close() 
            except NameError: pass
            except Exception: pass

        logger.info(f"Visualization generation finished. {len(plot_paths)} plots created.")
        return plot_paths

    def run_analysis(self, metrics_to_compute: List[str], comparison_metric: str, report_format: str = 'json', generate_plots: bool = True):
        """
        Runs the full comparative analysis pipeline.
        
        Args:
            metrics_to_compute: Metrics to calculate during evaluation.
            comparison_metric: Metric to use for statistical comparisons.
            report_format: Format for the output report.
            generate_plots: Whether to generate visualizations.
        """
        logger.info("Starting full comparative analysis pipeline...")
        self.run_cross_sample_evaluation(metrics_to_compute)
        self.perform_statistical_tests(comparison_metric)
        self.generate_comparative_report(output_format=report_format)
        if generate_plots:
            self.generate_visualizations()
        logger.info("Comparative analysis pipeline finished.")


# Example Usage (Placeholder)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Running ComparativeAnalyzer example...")

    # Dummy data structures - Replace with actual config/data loading
    dummy_model_configs = [
        {'model_name': 'SeqCNNRegressor', 'checkpoint': 'path/to/model1.pt', 'config': 'path/to/config1.yaml'},
        # {'model_name': 'AnotherModel', 'checkpoint': 'path/to/model2.pt', 'config': 'path/to/config2.yaml'},
    ]
    dummy_sample_groups = {
        'AML': {'test': 'path/to/aml_test.parquet'},
        'CD34': {'test': 'path/to/cd34_test.parquet'}
    }
    dummy_output_dir = './comparative_analysis_output'
    
    try:
        # Create dummy output dir if it doesn't exist
        os.makedirs(dummy_output_dir, exist_ok=True)
        
        analyzer = ComparativeAnalyzer(
            model_configs=dummy_model_configs,
            sample_group_data=dummy_sample_groups,
            base_output_dir=dummy_output_dir
        )
        
        # Run the placeholder analysis
        analyzer.run_analysis(
            metrics_to_compute=['mse', 'r2'], 
            comparison_metric='mse',
            report_format='json',
            generate_plots=True
        )
        
        logger.info("ComparativeAnalyzer example completed.")
        
    except Exception as e:
        logger.error(f"Error in ComparativeAnalyzer example: {e}", exc_info=True) 
import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import the class to test
from epibench.pipeline.results_collector import ResultsCollector
# --- Remove unused config imports ---
# from epibench.validation.config_validator import ProcessConfig, LoggingConfig 

# --- Fixtures ---

# --- Rename fixture to provide base_output_dir Path ---
@pytest.fixture
def base_output_dir(tmp_path):
    """Provides a Path object for the base output directory in temp space."""
    output_path = tmp_path / "test_collector_output"
    output_path.mkdir(parents=True, exist_ok=True) # Ensure it exists
    return output_path

@pytest.fixture
def sample_checkpoint_data():
    """Provides sample checkpoint data with different statuses."""
    return {
        "sample_ok_1": {"status": "completed"},
        "sample_ok_2": {"status": "completed"},
        "sample_failed": {"status": "failed"},
        "sample_pending": {},
        "sample_missing_metrics": {"status": "completed"},
        "sample_missing_preds": {"status": "completed"},
        "sample_bad_metrics": {"status": "completed"},
    }

# --- Update fixture signature and usage ---
@pytest.fixture
def setup_output_dirs(base_output_dir, sample_checkpoint_data):
    """Creates dummy output directories and files based on checkpoint data."""
    # base_output_dir is now directly the Path object from the fixture
    # base_output_dir.mkdir(parents=True, exist_ok=True) # Already created in base_output_dir fixture

    created_files = {
        "base": base_output_dir,
        "metrics": {},
        "predictions": {}
    }

    # Create structure only for samples marked as potentially having results
    for sample_id, data in sample_checkpoint_data.items():
        if data.get("status") == "completed":
            sample_dir = base_output_dir / sample_id
            eval_dir = sample_dir / "evaluation_output"
            predict_dir = sample_dir / "prediction_output"
            train_dir = sample_dir / "training_output"
            
            eval_dir.mkdir(parents=True, exist_ok=True)
            predict_dir.mkdir(parents=True, exist_ok=True)
            train_dir.mkdir(parents=True, exist_ok=True)

            # --- Create dummy files based on sample ID for different test cases ---
            
            # Standard good samples
            if sample_id in ["sample_ok_1", "sample_ok_2"]:
                metrics_path = eval_dir / "evaluation_metrics.json"
                metrics_content = {"mse": 0.1 if sample_id == "sample_ok_1" else 0.2, "r2": 0.9 if sample_id == "sample_ok_1" else 0.8}
                with open(metrics_path, 'w') as f: json.dump(metrics_content, f)
                created_files["metrics"][sample_id] = metrics_path
                
                preds_path = predict_dir / "predictions.csv"
                pd.DataFrame({'predictions': [1,2,3]}).to_csv(preds_path, index=False)
                created_files["predictions"][sample_id] = preds_path

            # Sample completed but missing metrics file
            elif sample_id == "sample_missing_metrics":
                preds_path = predict_dir / "predictions.csv"
                pd.DataFrame({'predictions': [4,5,6]}).to_csv(preds_path, index=False)
                created_files["predictions"][sample_id] = preds_path
                created_files["metrics"][sample_id] = None 

            # Sample completed but missing predictions file
            elif sample_id == "sample_missing_preds":
                metrics_path = eval_dir / "evaluation_metrics.json"
                metrics_content = {"mse": 0.3, "r2": 0.7}
                with open(metrics_path, 'w') as f: json.dump(metrics_content, f)
                created_files["metrics"][sample_id] = metrics_path
                created_files["predictions"][sample_id] = None
                
            # Sample with bad/corrupted metrics file
            elif sample_id == "sample_bad_metrics":
                metrics_path = eval_dir / "evaluation_metrics.json"
                with open(metrics_path, 'w') as f: f.write("{invalid json")
                created_files["metrics"][sample_id] = metrics_path
                
                preds_path = predict_dir / "predictions.csv"
                pd.DataFrame({'predictions': [7,8,9]}).to_csv(preds_path, index=False)
                created_files["predictions"][sample_id] = preds_path

    return created_files

# --- Test Class --- 

class TestResultsCollector:

    # --- Update test signature and assertions ---
    def test_init(self, base_output_dir, sample_checkpoint_data):
        """Test initialization of the collector."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data)
        # assert collector.main_config == mock_main_config # Removed
        assert collector.checkpoint_data == sample_checkpoint_data
        assert collector.base_output_directory == base_output_dir

    # --- Update test signature ---
    def test_find_completed_samples(self, base_output_dir, sample_checkpoint_data):
        """Test identification of completed samples."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data)
        completed = collector._find_completed_samples()
        expected_completed = [
            "sample_ok_1", "sample_ok_2", "sample_missing_metrics", 
            "sample_missing_preds", "sample_bad_metrics"
        ]
        assert sorted(completed) == sorted(expected_completed)
        assert len(completed) == 5

    # --- Update test signature ---
    def test_aggregate_sample_results_success(self, base_output_dir, sample_checkpoint_data, setup_output_dirs):
        """Test aggregation works for samples with expected files."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data)
        completed_to_test = ["sample_ok_1", "sample_ok_2"]
        summary_dir = setup_output_dirs['base'] / "summary_reports"
        summary_dir.mkdir(exist_ok=True)
        
        aggregated = collector._aggregate_sample_results(completed_to_test)

        assert "sample_ok_1" in aggregated["metrics"]
        assert aggregated["metrics"]["sample_ok_1"]["mse"] == 0.1
        assert "sample_ok_2" in aggregated["metrics"]
        assert aggregated["metrics"]["sample_ok_2"]["r2"] == 0.8
        
        assert "sample_ok_1" in aggregated["prediction_files"]
        assert aggregated["prediction_files"]["sample_ok_1"] == str(setup_output_dirs["predictions"]["sample_ok_1"])
        assert "sample_ok_2" in aggregated["prediction_files"]
        assert aggregated["prediction_files"]["sample_ok_2"] == str(setup_output_dirs["predictions"]["sample_ok_2"])

    # --- Update test signature ---
    def test_aggregate_sample_results_missing_metrics(self, base_output_dir, sample_checkpoint_data, setup_output_dirs):
        """Test aggregation when metrics file is missing."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data)
        completed_to_test = ["sample_missing_metrics"]
        aggregated = collector._aggregate_sample_results(completed_to_test)
        
        assert "sample_missing_metrics" in aggregated["metrics"]
        assert aggregated["metrics"]["sample_missing_metrics"]["error"] == "not_found"
        assert "sample_missing_metrics" in aggregated["prediction_files"]
        assert isinstance(aggregated["prediction_files"]["sample_missing_metrics"], str)

    # --- Update test signature ---
    def test_aggregate_sample_results_missing_predictions(self, base_output_dir, sample_checkpoint_data, setup_output_dirs):
        """Test aggregation when predictions file is missing."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data)
        completed_to_test = ["sample_missing_preds"]
        aggregated = collector._aggregate_sample_results(completed_to_test)
        
        assert "sample_missing_preds" in aggregated["metrics"]
        assert aggregated["metrics"]["sample_missing_preds"]["mse"] == 0.3
        assert "sample_missing_preds" in aggregated["prediction_files"]
        assert aggregated["prediction_files"]["sample_missing_preds"]["error"] == "not_found"

    # --- Update test signature ---
    def test_aggregate_sample_results_bad_metrics(self, base_output_dir, sample_checkpoint_data, setup_output_dirs):
        """Test aggregation when metrics file is corrupted."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data)
        completed_to_test = ["sample_bad_metrics"]
        aggregated = collector._aggregate_sample_results(completed_to_test)
        
        assert "sample_bad_metrics" in aggregated["metrics"]
        assert aggregated["metrics"]["sample_bad_metrics"]["error"] == "failed_to_load"
        assert "sample_bad_metrics" in aggregated["prediction_files"]
        assert isinstance(aggregated["prediction_files"]["sample_bad_metrics"], str)
        
    # --- Update test signature ---
    def test_calculate_summary_stats_success(self, base_output_dir, sample_checkpoint_data):
        """Test statistics calculation with valid aggregated data."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data)
        aggregated_data = {
            "metrics": {
                "sample_ok_1": {"mse": 0.1, "r2": 0.9, "non_numeric": "a"},
                "sample_ok_2": {"mse": 0.2, "r2": 0.8, "extra_metric": 5},
                "sample_err": {"error": "not_found"}
            },
             "prediction_files": {}
        }
        summary = collector._calculate_summary_stats(aggregated_data)
        
        assert "metric_summaries" in summary
        assert summary["num_samples_with_metrics"] == 2
        assert "mse" in summary["metric_summaries"]
        assert "r2" in summary["metric_summaries"]
        assert "extra_metric" in summary["metric_summaries"]
        assert "non_numeric" not in summary["metric_summaries"]
        assert summary["metric_summaries"]["mse"]["mean"] == pytest.approx(0.15)
        assert summary["metric_summaries"]["r2"]["count"] == 2
        assert summary["metric_summaries"]["extra_metric"]["count"] == 1

    # --- Update test signature ---
    def test_calculate_summary_stats_no_numeric(self, base_output_dir, sample_checkpoint_data):
        """Test stats calc when metrics exist but none are numeric."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data)
        aggregated_data = {"metrics": {"s1": {"name": "a"}, "s2": {"name": "b"}}}
        summary = collector._calculate_summary_stats(aggregated_data)
        assert summary["message"] == "No numeric metrics found in the aggregated data."
        assert "metric_summaries" not in summary

    # --- Update test signature ---
    def test_calculate_summary_stats_no_valid(self, base_output_dir, sample_checkpoint_data):
        """Test stats calc when no valid metrics were aggregated."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data)
        aggregated_data = {"metrics": {"s1": {"error": "not_found"}}}
        summary = collector._calculate_summary_stats(aggregated_data)
        assert summary["message"] == "No valid metrics found to summarize."
        assert "metric_summaries" not in summary
        
    # --- Update test signature ---
    def test_generate_reports(self, base_output_dir, sample_checkpoint_data, setup_output_dirs):
        """Test that report files are generated."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data)
        summary_dir = setup_output_dirs['base'] / "summary_reports"
        summary_dir.mkdir(exist_ok=True)
        
        aggregated_data = {
            "metrics": {"s1": {"mse": 0.1}}, 
            "prediction_files": {},
            "training_logs": {}
        }
        summary_stats = {"num_samples_with_metrics": 1, "metric_summaries": {"mse": {"mean": 0.1}}}
        
        collector._generate_reports(aggregated_data, summary_stats, summary_dir)
            
        assert (summary_dir / "aggregated_metrics_raw.json").is_file()
        assert (summary_dir / "summary_statistics.json").is_file()

    # --- Update test signature ---
    def test_summarize_run_status(self, base_output_dir, sample_checkpoint_data, setup_output_dirs):
        """Test the generation of the run status summary JSON."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data)
        summary_dir = setup_output_dirs['base'] / "summary_reports"
        summary_dir.mkdir(exist_ok=True)
        
        collector._summarize_run_status(summary_dir)
        
        summary_file = summary_dir / "run_status_summary.json"
        assert summary_file.is_file()
        with open(summary_file, 'r') as f:
            status_data = json.load(f)
            
        assert status_data["total_samples_in_checkpoint"] == len(sample_checkpoint_data)
        assert status_data["status_counts"]["completed"] == 5
        assert status_data["status_counts"]["failed"] == 1
        assert status_data["status_counts"]["pending"] == 1
        assert status_data["failed_sample_ids"] == ["sample_failed"]
        
    # --- Placeholder Test for collect_all orchestration --- 
    
    @patch.object(ResultsCollector, '_find_completed_samples')
    @patch.object(ResultsCollector, '_aggregate_sample_results')
    @patch.object(ResultsCollector, '_calculate_summary_stats')
    @patch.object(ResultsCollector, '_generate_reports')
    @patch.object(ResultsCollector, '_summarize_run_status')
    # --- Update test signature ---
    def test_collect_all_orchestration(self, mock_summarize, mock_generate, mock_calculate, mock_aggregate, mock_find, 
                                       base_output_dir, sample_checkpoint_data, setup_output_dirs):
        """Test that collect_all calls the helper methods in order."""
        # --- Update instantiation ---
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data)
        # --- Update expected path ---
        expected_summary_dir = base_output_dir / "summary_reports"

        mock_find.return_value = ["sample_ok_1"]
        mock_aggregate.return_value = {
            "metrics": {"sample_ok_1": {"mse": 0.1}},
            "prediction_files": {},
            "training_logs": {}
        }
        mock_calculate.return_value = {"metric_summaries": {"mse": {"mean": 0.1}}}

        with patch('pathlib.Path.mkdir') as mock_mkdir:
            collector.collect_all()
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)

        mock_find.assert_called_once()
        mock_aggregate.assert_called_once_with(["sample_ok_1"])
        mock_calculate.assert_called_once_with(mock_aggregate.return_value)
        mock_generate.assert_called_once_with(mock_aggregate.return_value, mock_calculate.return_value, expected_summary_dir)
        mock_summarize.assert_called_once_with(expected_summary_dir)

    @patch.object(ResultsCollector, '_find_completed_samples')
    @patch.object(ResultsCollector, '_aggregate_sample_results')
    # --- Update test signature ---
    def test_collect_all_no_completed(self, mock_aggregate, mock_find,
                                    base_output_dir, sample_checkpoint_data, setup_output_dirs):
        """Test collect_all when no samples are completed."""
        # --- Update instantiation ---
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data)
        mock_find.return_value = []

        collector.collect_all()

        mock_find.assert_called_once()
        mock_aggregate.assert_not_called() 
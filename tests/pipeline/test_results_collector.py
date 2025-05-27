import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY
import tempfile # For temp directories in tests
import shutil # For cleaning up temp directories
from datetime import datetime

# Import the class to test
from epibench.pipeline.results_collector import ResultsCollector, NumpyJSONEncoder
# --- Remove unused config imports ---
# from epibench.validation.config_validator import ProcessConfig, LoggingConfig 

# --- Fixtures ---

# --- Rename fixture to provide base_output_dir Path ---
@pytest.fixture
def base_output_dir(tmp_path):
    """Provides a Path object for the base output directory in temp space."""
    output_path = tmp_path / "test_collector_output"
    output_path.mkdir(parents=True, exist_ok=True) # Ensure it exists
    # Create templates dir for Jinja setup
    (output_path.parent / "epibench" / "pipeline" / "templates").mkdir(parents=True, exist_ok=True)
    return output_path

@pytest.fixture
def sample_checkpoint_data():
    """Provides sample checkpoint data with different statuses."""
    return {
        "sample_ok_1": {"status": "completed", "config": {"model": "model_a"}}, # Added config example
        "sample_ok_2": {"status": "completed", "config": {"model": "model_b"}},
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
                # Add sample_id implicitly during aggregation now
                metrics_content = {"mse": 0.1 if sample_id == "sample_ok_1" else 0.2, "r2": 0.9 if sample_id == "sample_ok_1" else 0.8, "accuracy": 0.95 if sample_id == "sample_ok_1" else 0.9}
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
                # Add sample_id implicitly during aggregation now
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

    # Create a dummy template file for Jinja setup
    template_dir = Path(__file__).parent.parent / "epibench" / "pipeline" / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "report_template.html").write_text("<html><body>{{ run_metadata.timestamp }}</body></html>")

    return created_files

# --- Helper Function for Sample DataFrames ---

def create_sample_metrics_df(num_samples=2):
    data = []
    for i in range(1, num_samples + 1):
        data.append({
            "sample_id": f"s{i}",
            "mse": 0.1 * i,
            "r2": 1.0 - (0.1 * i),
            "accuracy": 0.9 - (0.05 * i),
            "non_numeric": f"cat_{i}"
        })
    return pd.DataFrame(data)


# --- Test Class --- 

class TestResultsCollector:

    @pytest.fixture(autouse=True)
    def _setup_templates(self, tmp_path):
        """Ensure template directory exists for tests needing Jinja env."""
        self.template_dir = tmp_path / "templates"
        self.template_dir.mkdir(exist_ok=True)
        (self.template_dir / "report_template.html").write_text("<html><body>Generated: {{ run_metadata.timestamp.strftime('%Y') }}<hr>{{ tables.summary_statistics | safe }}<hr>{{ tables.detailed_metrics | safe }}<hr><img src=\"{{ plots.accuracy_bar }}\"><hr>{{ plots.mse_r2_scatter | safe }}</body></html>")
        # Patch the location where ResultsCollector looks for templates
        self.template_patch = patch('epibench.pipeline.results_collector.Path')
        mock_path = self.template_patch.start()
        # Make Path(__file__).parent return the tmp_path structure
        mock_path.return_value.parent.__truediv__.return_value = self.template_dir
        # Mock Path() constructor calls within ResultsCollector to return controlled paths
        # This prevents accidental access to the real filesystem during tests
        # Configure the mock for Path(base_output_directory) in __init__
        mock_path.side_effect = lambda x: tmp_path / x if isinstance(x, str) else mock_path.return_value # Simple side effect
        yield
        self.template_patch.stop()

    # --- Update test signature and assertions ---
    def test_init(self, base_output_dir, sample_checkpoint_data):
        """Test initialization of the collector, including Jinja env and default config."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data)
        assert collector.checkpoint_data == sample_checkpoint_data
        assert collector.base_output_directory == base_output_dir
        assert collector.jinja_env is not None # Check Jinja env is initialized
        # Check default config is loaded
        from epibench.pipeline.results_collector import DEFAULT_REPORT_CONFIG
        assert collector.report_config == DEFAULT_REPORT_CONFIG

    def test_init_with_custom_config(self, base_output_dir, sample_checkpoint_data):
        """Test initialization with a user-provided config."""
        custom_config = {"output_formats": ["html"], "report_preset": "custom"}
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=custom_config)
        assert collector.report_config["output_formats"] == ["html"]
        assert collector.report_config["report_preset"] == "custom"
        # Check other defaults are still present
        from epibench.pipeline.results_collector import DEFAULT_REPORT_CONFIG
        assert collector.report_config["included_sections"] == DEFAULT_REPORT_CONFIG["included_sections"]

    # --- Tests for _process_report_config --- 
    def test_process_report_config_defaults(self, base_output_dir, sample_checkpoint_data):
        """Test config processing uses defaults when no user config provided."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        from epibench.pipeline.results_collector import DEFAULT_REPORT_CONFIG
        assert collector.report_config == DEFAULT_REPORT_CONFIG

    def test_process_report_config_preset_summary(self, base_output_dir, sample_checkpoint_data):
        """Test applying the 'summary' preset."""
        user_config = {"report_preset": "summary"}
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=user_config)
        from epibench.pipeline.results_collector import REPORT_PRESETS
        summary_preset = REPORT_PRESETS["summary"]
        assert collector.report_config["included_sections"] == summary_preset["included_sections"]
        assert collector.report_config["included_metrics"] == summary_preset["included_metrics"]
        assert collector.report_config["included_plots"] == summary_preset["included_plots"]
        # Check theme merge
        assert collector.report_config["theme"]["primary_color"] == summary_preset["theme"]["primary_color"]

    def test_process_report_config_preset_with_overrides(self, base_output_dir, sample_checkpoint_data):
        """Test applying a preset and then specific user overrides."""
        user_config = {
            "report_preset": "summary", 
            "output_formats": ["pdf"], # Override preset's format
            "theme": {"font_family": "monospace"} # Override theme aspect
        }
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=user_config)
        from epibench.pipeline.results_collector import REPORT_PRESETS, DEFAULT_REPORT_CONFIG
        summary_preset = REPORT_PRESETS["summary"]
        # Check overrides
        assert collector.report_config["output_formats"] == ["pdf"]
        assert collector.report_config["theme"]["font_family"] == "monospace"
        # Check preset values that weren't overridden
        assert collector.report_config["included_sections"] == summary_preset["included_sections"]
        # Check default values that weren't touched by preset or override
        assert collector.report_config["report_title"] == DEFAULT_REPORT_CONFIG["report_title"]

    def test_process_report_config_custom_with_overrides(self, base_output_dir, sample_checkpoint_data):
        """Test custom config (no preset applied) with specific settings."""
        user_config = {
            "report_preset": "custom", # Explicitly custom
            "included_sections": ["summary_stats"], 
            "included_metrics": ["mse"], 
            "theme": {"primary_color": "#FF0000"} 
        }
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=user_config)
        from epibench.pipeline.results_collector import DEFAULT_REPORT_CONFIG
        # Check overrides
        assert collector.report_config["included_sections"] == ["summary_stats"]
        assert collector.report_config["included_metrics"] == ["mse"]
        assert collector.report_config["theme"]["primary_color"] == "#FF0000"
        # Check defaults were kept where not overridden
        assert collector.report_config["output_formats"] == DEFAULT_REPORT_CONFIG["output_formats"]
        assert collector.report_config["included_plots"] == DEFAULT_REPORT_CONFIG["included_plots"]

    def test_process_report_config_invalid_values(self, base_output_dir, sample_checkpoint_data):
        """Test validation logic reverts invalid config types to defaults."""
        user_config = {
            "output_formats": "html", # Invalid type
            "included_sections": { "a": 1 }, # Invalid type
            "included_metrics": 123, # Invalid type
            "included_plots": False, # Invalid type
            "theme": "bad_theme", # Invalid type
            "unknown_key": "should_be_ignored"
        }
        # Capture warnings during init
        with pytest.warns(UserWarning): # Use UserWarning if logger uses warnings.warn, otherwise check log output
             collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=user_config)
         
        from epibench.pipeline.results_collector import DEFAULT_REPORT_CONFIG
        # Check that invalid types were reverted to defaults
        assert collector.report_config["output_formats"] == DEFAULT_REPORT_CONFIG["output_formats"]
        assert collector.report_config["included_sections"] == DEFAULT_REPORT_CONFIG["included_sections"]
        assert collector.report_config["included_metrics"] == DEFAULT_REPORT_CONFIG["included_metrics"]
        assert collector.report_config["included_plots"] == DEFAULT_REPORT_CONFIG["included_plots"]
        assert collector.report_config["theme"] == DEFAULT_REPORT_CONFIG["theme"]
        # Check unknown key was ignored (or logged)
        assert "unknown_key" not in collector.report_config 

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
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        completed_to_test = ["sample_ok_1", "sample_ok_2"]
        summary_dir = setup_output_dirs['base'] / "summary_reports"
        summary_dir.mkdir(exist_ok=True)
        
        aggregated = collector._aggregate_sample_results(completed_to_test)

        assert "sample_ok_1" in aggregated["metrics"]
        assert aggregated["metrics"]["sample_ok_1"]["mse"] == 0.1
        assert aggregated["metrics"]["sample_ok_1"]["sample_id"] == "sample_ok_1" # Check added field
        assert "sample_ok_2" in aggregated["metrics"]
        assert aggregated["metrics"]["sample_ok_2"]["r2"] == 0.8
        assert aggregated["metrics"]["sample_ok_2"]["sample_id"] == "sample_ok_2" # Check added field
        
        assert "sample_ok_1" in aggregated["prediction_files"]
        assert aggregated["prediction_files"]["sample_ok_1"] == str(setup_output_dirs["predictions"]["sample_ok_1"])
        assert "sample_ok_2" in aggregated["prediction_files"]
        assert aggregated["prediction_files"]["sample_ok_2"] == str(setup_output_dirs["predictions"]["sample_ok_2"])
        
        # Check that consolidated metrics file was saved
        consolidated_file = summary_dir / "consolidated_metrics.json"
        assert consolidated_file.is_file()
        with open(consolidated_file, 'r') as f:
            saved_metrics = json.load(f)
            assert "sample_ok_1" in saved_metrics
            assert "sample_ok_2" in saved_metrics

    # --- Update test signature ---
    def test_aggregate_sample_results_missing_metrics(self, base_output_dir, sample_checkpoint_data, setup_output_dirs):
        """Test aggregation when metrics file is missing."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        completed_to_test = ["sample_missing_metrics"]
        aggregated = collector._aggregate_sample_results(completed_to_test)
        
        assert "sample_missing_metrics" in aggregated["metrics"]
        assert aggregated["metrics"]["sample_missing_metrics"]["error"] == "not_found"
        assert aggregated["metrics"]["sample_missing_metrics"]["sample_id"] == "sample_missing_metrics"
        assert "sample_missing_metrics" in aggregated["prediction_files"]
        assert isinstance(aggregated["prediction_files"]["sample_missing_metrics"], str)

    # --- Update test signature ---
    def test_aggregate_sample_results_missing_predictions(self, base_output_dir, sample_checkpoint_data, setup_output_dirs):
        """Test aggregation when predictions file is missing."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        completed_to_test = ["sample_missing_preds"]
        aggregated = collector._aggregate_sample_results(completed_to_test)
        
        assert "sample_missing_preds" in aggregated["metrics"]
        assert aggregated["metrics"]["sample_missing_preds"]["mse"] == 0.3
        assert aggregated["metrics"]["sample_missing_preds"]["sample_id"] == "sample_missing_preds"
        assert "sample_missing_preds" in aggregated["prediction_files"]
        assert aggregated["prediction_files"]["sample_missing_preds"]["error"] == "not_found"

    # --- Update test signature ---
    def test_aggregate_sample_results_bad_metrics(self, base_output_dir, sample_checkpoint_data, setup_output_dirs):
        """Test aggregation when metrics file is corrupted."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        completed_to_test = ["sample_bad_metrics"]
        aggregated = collector._aggregate_sample_results(completed_to_test)
        
        assert "sample_bad_metrics" in aggregated["metrics"]
        assert aggregated["metrics"]["sample_bad_metrics"]["error"] == "failed_to_load"
        assert aggregated["metrics"]["sample_bad_metrics"]["sample_id"] == "sample_bad_metrics"
        assert "sample_bad_metrics" in aggregated["prediction_files"]
        assert isinstance(aggregated["prediction_files"]["sample_bad_metrics"], str)
        
    # --- RENAME TEST and update to use metrics_df ---
    def test_prepare_summary_stats_success(self, base_output_dir, sample_checkpoint_data):
        """Test statistics calculation with a valid metrics DataFrame."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        metrics_df = pd.DataFrame([
            {"sample_id": "s1", "mse": 0.1, "r2": 0.9, "non_numeric": "a"},
            {"sample_id": "s2", "mse": 0.2, "r2": 0.8, "extra_metric": 5},
        ])
        summary = collector._prepare_summary_stats(metrics_df)
        
        assert "metric_summaries" in summary
        assert summary["num_samples_with_metrics"] == 2
        assert "mse" in summary["metric_summaries"]
        assert "r2" in summary["metric_summaries"]
        assert "extra_metric" in summary["metric_summaries"]
        assert "non_numeric" not in summary["metric_summaries"]
        assert summary["metric_summaries"]["mse"]["mean"] == pytest.approx(0.15)
        assert summary["metric_summaries"]["r2"]["count"] == 2
        assert summary["metric_summaries"]["extra_metric"]["count"] == 1

    # --- RENAME TEST and update to use metrics_df ---
    def test_prepare_summary_stats_no_numeric(self, base_output_dir, sample_checkpoint_data):
        """Test stats calc when DataFrame has no numeric metrics."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        metrics_df = pd.DataFrame([{"sample_id": "s1", "name": "a"}, {"sample_id": "s2","name": "b"}])
        summary = collector._prepare_summary_stats(metrics_df)
        assert summary["message"] == "No numeric metrics found in the aggregated data."
        assert "metric_summaries" not in summary

    # --- RENAME TEST and update to use metrics_df ---
    def test_prepare_summary_stats_empty_df(self, base_output_dir, sample_checkpoint_data):
        """Test stats calc when the input DataFrame is empty."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        metrics_df = pd.DataFrame()
        summary = collector._prepare_summary_stats(metrics_df)
        assert summary["message"] == "No valid metrics found to summarize."
        assert "metric_summaries" not in summary

    # --- ADD Placeholder Tests for New Methods ---
    def test_prepare_visualization_data(self, base_output_dir, sample_checkpoint_data):
        """Test placeholder for visualization data preparation."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        metrics_df = create_sample_metrics_df(num_samples=2)
        viz_data = collector._prepare_visualization_data(metrics_df)
        assert isinstance(viz_data, dict)
        # Add more specific checks based on placeholder implementation
        assert "bar_charts" in viz_data
        assert "accuracy_per_sample" in viz_data["bar_charts"]
        assert len(viz_data["bar_charts"]["accuracy_per_sample"]["sample_id_str"]) == 2
        assert "scatter_plots" in viz_data
        assert "mse_vs_r2" in viz_data["scatter_plots"]
        assert len(viz_data["scatter_plots"]["mse_vs_r2"]["sample_id"]) == 2

    # --- Update test to check for HTML output --- 
    def test_prepare_styled_tables(self, base_output_dir, sample_checkpoint_data):
        """Test that styled tables are prepared as HTML strings."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        metrics_df = create_sample_metrics_df()
        summary_stats = collector._prepare_summary_stats(metrics_df)
        styled_tables = collector._prepare_styled_tables(metrics_df, summary_stats)
        assert isinstance(styled_tables, dict)
        assert "summary_statistics" in styled_tables
        assert "detailed_metrics" in styled_tables
        # Check that the output is a string (HTML)
        assert isinstance(styled_tables["summary_statistics"], str)
        assert isinstance(styled_tables["detailed_metrics"], str)
        # Check for basic HTML table tags
        assert "<table" in styled_tables["summary_statistics"]
        assert "</table>" in styled_tables["summary_statistics"]
        assert "class=\"table" in styled_tables["summary_statistics"] # Check for Bootstrap class
        assert "<table" in styled_tables["detailed_metrics"]
        assert "</table>" in styled_tables["detailed_metrics"]
        assert "index=False" not in styled_tables["detailed_metrics"] # Check index is rendered for detailed?

    def test_prepare_report_data(self, base_output_dir, sample_checkpoint_data):
        """Test the main report data preparation method."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        # Simulate aggregated data with some valid and some invalid metrics
        aggregated_data = {
            "metrics": {
                "s1": {"sample_id": "s1", "mse": 0.1, "r2": 0.9, "accuracy": 0.95},
                "s2": {"sample_id": "s2", "mse": 0.2, "r2": 0.8, "accuracy": 0.90},
                "s_err": {"sample_id": "s_err", "error": "failed_to_load"}
            },
            "prediction_files": {}
        }
        report_context = collector._prepare_report_data(aggregated_data)

        assert "summary_stats_dict" in report_context # Check renamed key
        assert "visualization_data" in report_context
        assert "styled_tables_html" in report_context # Check renamed key
        assert "run_metadata" in report_context
        assert "failed_samples" in report_context
        assert report_context["summary_stats_dict"].get("num_samples_with_metrics") == 2
        assert report_context["run_metadata"]['total_samples_in_checkpoint'] == len(sample_checkpoint_data)
        assert "s_err" not in report_context["summary_stats_dict"].get("metric_summaries", {}).get("mse", {}).keys() # Check error sample excluded from stats
        assert report_context["failed_samples"] == ["sample_failed"] # Check failed samples list
        assert isinstance(report_context["run_metadata"]['timestamp'], datetime) # Check timestamp type
        
    # --- Test Plotting Helpers --- 
    def test_plot_matplotlib_bar(self, base_output_dir, sample_checkpoint_data):
        """Test the matplotlib bar plot helper returns base64 string."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        x = ['a', 'b']
        y = [10, 20]
        b64_string = collector._plot_matplotlib_bar(x, y, "Test Bar", "X", "Y")
        assert isinstance(b64_string, str)
        assert b64_string.startswith("data:image/png;base64,")

    def test_plot_plotly_scatter(self, base_output_dir, sample_checkpoint_data):
        """Test the plotly scatter plot helper returns HTML div string."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        x = [1, 2]
        y = [10, 20]
        labels = ['s1', 's2']
        html_div = collector._plot_plotly_scatter(x, y, labels, "Test Scatter", "X", "Y")
        assert isinstance(html_div, str)
        assert html_div.startswith("<div>")
        assert html_div.endswith("</div>")
        assert "plotly-graph-div" in html_div # Check for plotly class
        assert "cdn.plot.ly" in html_div # Check if CDN is included
        
    # --- Test HTML Report Generation --- 
    @patch.object(ResultsCollector, '_plot_matplotlib_bar')
    @patch.object(ResultsCollector, '_plot_plotly_scatter')
    def test_generate_html_report(self, mock_plotly_scatter, mock_mpl_bar, tmp_path, sample_checkpoint_data):
        """Test the main HTML report generation method."""
        # Setup mocks
        mock_mpl_bar.return_value = "data:image/png;base64,TEST_MPL_B64"
        mock_plotly_scatter.return_value = "<div id='plotly_div'>TEST_PLOTLY_DIV</div>"

        # Use tmp_path for base directory in this specific test
        test_base_dir = tmp_path / "html_test_base"
        collector = ResultsCollector(test_base_dir, sample_checkpoint_data, report_config=None)
        summary_dir = test_base_dir / "summary_reports" 
        # summary_dir.mkdir(exist_ok=True) # Should be created by method

        # Create comprehensive report context
        metrics_df = create_sample_metrics_df(2)
        aggregated_data = {"metrics": dict(zip(metrics_df['sample_id'], metrics_df.to_dict('records'))) }
        report_context = collector._prepare_report_data(aggregated_data)

        # Call the method under test
        output_path = collector.generate_html_report(report_context, summary_dir)

        # Assertions
        assert output_path is not None
        assert output_path.exists()
        assert output_path.name.startswith("epibench_report_")
        assert output_path.name.endswith(".html")
        
        # Check mocks called correctly
        mock_mpl_bar.assert_called_once() # Should be called based on viz_data prep
        mock_plotly_scatter.assert_called_once() # Should be called based on viz_data prep

        # Read generated HTML and check content
        html_content = output_path.read_text()
        assert "<title>EpiBench Evaluation Report</title>" in html_content
        assert "Summary Statistics" in html_content
        assert "Detailed Metrics per Sample" in html_content
        assert "Visualizations" in html_content
        assert "class=\"table table-sm table-striped table-hover\"" in html_content # Check styled table class
        assert "data:image/png;base64,TEST_MPL_B64" in html_content # Check MPL plot embedded
        assert "<div id='plotly_div'>TEST_PLOTLY_DIV</div>" in html_content # Check Plotly div embedded
        assert "Failed Samples" not in html_content # No failed samples in this context

        # Clean up
        # shutil.rmtree(summary_dir) # tmp_path fixture handles cleanup

    # --- Test PDF Report Generation (Mocking WeasyPrint) --- 
    @patch('epibench.pipeline.results_collector.weasyprint.HTML')
    @patch('epibench.pipeline.results_collector.weasyprint.CSS')
    def test_generate_pdf_report(self, mock_css, mock_html_weasy, tmp_path, sample_checkpoint_data):
        """Test PDF generation process, mocking WeasyPrint."""
        # Setup
        test_base_dir = tmp_path / "pdf_test_base"
        collector = ResultsCollector(test_base_dir, sample_checkpoint_data, report_config=None)
        summary_dir = test_base_dir / "summary_reports"
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy HTML file that PDF generation reads
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dummy_html_path = summary_dir / f"epibench_report_{timestamp}.html"
        dummy_html_path.write_text("<html><body>Test HTML</body></html>", encoding='utf-8')
        
        # Create dummy CSS file
        css_path = self.template_dir / "report_print.css" # Use template dir from autouse fixture
        css_path.write_text("@page { size: A4; }", encoding='utf-8')

        # Mock WeasyPrint HTML object and its write_pdf method
        mock_html_instance = MagicMock()
        mock_html_weasy.return_value = mock_html_instance
        mock_css_instance = MagicMock()
        mock_css.return_value = mock_css_instance

        # Call the method under test
        output_pdf_path = collector.generate_pdf_report(dummy_html_path, summary_dir)

        # Assertions
        assert output_pdf_path is not None
        assert output_pdf_path.name == dummy_html_path.with_suffix('.pdf').name
        
        # Check that WeasyPrint HTML was instantiated with correct string and base_url
        mock_html_weasy.assert_called_once_with(string="<html><body>Test HTML</body></html>", base_url=str(summary_dir))
        # Check that WeasyPrint CSS was instantiated with the correct path
        mock_css.assert_called_once_with(filename=str(css_path))
        # Check that write_pdf was called on the HTML instance
        mock_html_instance.write_pdf.assert_called_once_with(
            str(output_pdf_path), 
            stylesheets=[mock_css_instance]
        )

    # --- Update test signature and mocks for _generate_reports ---
    # This test now primarily checks that generate_html_report is called within _generate_reports
    @patch.object(ResultsCollector, 'generate_html_report')
    @patch.object(ResultsCollector, 'generate_pdf_report') # Also mock PDF generation
    def test_generate_reports_calls_html_and_pdf(self, mock_generate_pdf, mock_generate_html, base_output_dir, sample_checkpoint_data, setup_output_dirs):
        """Test that _generate_reports calls HTML and PDF generation helpers."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        summary_dir = setup_output_dirs['base'] / "summary_reports"
        summary_dir.mkdir(exist_ok=True)
        
        # Mock HTML generation to return a path
        mock_html_path = summary_dir / "mock_report.html"
        mock_generate_html.return_value = mock_html_path
        mock_html_path.touch() # Create the dummy file
        
        # Create sample report context (simpler version needed)
        report_context = {
            "run_metadata": {"timestamp": datetime.now()},
            "summary_stats_dict": {"num_samples_with_metrics": 1, "metric_summaries": {"mse": {"mean": 0.1}}},
            "visualization_data": {},
            "styled_tables_html": {
                "summary_statistics": "<table>...</table>",
                "detailed_metrics": "<table>...</table>"
            },
            "failed_samples": []
        }
        
        collector._generate_reports(report_context, summary_dir)
        
        # Check main HTML generation method was called
        mock_generate_html.assert_called_once_with(report_context, summary_dir)
        # Check PDF generation method was called with the HTML path
        mock_generate_pdf.assert_called_once_with(mock_html_path, summary_dir)
            
        # Check that intermediate files are still saved
        assert (summary_dir / "report_context_snapshot.json").is_file()
        assert (summary_dir / "summary_statistics.csv").is_file()
        # Detailed CSV saving is now skipped in _generate_reports
        # assert (summary_dir / "detailed_metrics.csv").is_file()

    # --- Update test signature ---
    def test_summarize_run_status(self, base_output_dir, sample_checkpoint_data, setup_output_dirs):
        """Test the generation of the run status summary JSON."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
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
    @patch.object(ResultsCollector, '_prepare_report_data') 
    @patch.object(ResultsCollector, '_generate_reports')
    @patch.object(ResultsCollector, '_summarize_run_status')
    # --- Update test signature and mocks ---
    def test_collect_all_orchestration(self, mock_summarize, mock_generate_reports, mock_prepare_data, mock_aggregate, mock_find, 
                                       base_output_dir, sample_checkpoint_data, setup_output_dirs):
        """Test that collect_all calls the helper methods in order."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        expected_summary_dir = base_output_dir / "summary_reports"

        # Mock return values
        mock_find.return_value = ["sample_ok_1"]
        mock_aggregate_return = {
            "metrics": {"sample_ok_1": {"sample_id": "sample_ok_1", "mse": 0.1}},
            "prediction_files": {},
            "training_logs": {}
        }
        mock_aggregate.return_value = mock_aggregate_return
        mock_prepare_data_return = {"summary_stats_dict": {"metric_summaries": {"mse": {"mean": 0.1}}}} # Example context
        mock_prepare_data.return_value = mock_prepare_data_return

        with patch('pathlib.Path.mkdir') as mock_mkdir:
            result = collector.collect_all()
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)

        # Assert calls
        mock_find.assert_called_once()
        mock_aggregate.assert_called_once_with(["sample_ok_1"])
        mock_prepare_data.assert_called_once_with(mock_aggregate_return)
        # Check _generate_reports call signature updated
        mock_generate_reports.assert_called_once_with(mock_prepare_data_return, expected_summary_dir)
        mock_summarize.assert_called_once_with(expected_summary_dir)
        # Check return value (currently still aggregated_data)
        assert result == mock_aggregate_return

    @patch.object(ResultsCollector, '_find_completed_samples')
    @patch.object(ResultsCollector, '_aggregate_sample_results')
    # --- Update test signature ---
    def test_collect_all_no_completed(self, mock_aggregate, mock_find,
                                    base_output_dir, sample_checkpoint_data, setup_output_dirs):
        """Test collect_all when no samples are completed."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        mock_find.return_value = []

        result = collector.collect_all()

        mock_find.assert_called_once()
        mock_aggregate.assert_not_called()
        # Ensure it returns the expected empty structure
        assert result == {"metrics": {}, "prediction_files": {}, "training_logs": {}} 

    def test_generate_html_report_no_jinja_env(self, tmp_path, sample_checkpoint_data):
        """Test HTML generation returns None and logs error if Jinja env is not set."""
        collector = ResultsCollector(tmp_path, sample_checkpoint_data)
        collector.jinja_env = None # Manually unset the environment
        summary_dir = tmp_path / "summary_reports"
        summary_dir.mkdir(exist_ok=True)
        report_context = {"run_metadata": {}} # Minimal context
        
        with patch.object(ResultsCollector.logger, 'error') as mock_log_error:
             output_path = collector.generate_html_report(report_context, summary_dir)
             assert output_path is None
             mock_log_error.assert_called_once_with("Jinja2 environment not initialized. Cannot generate HTML report.")

    def test_generate_html_report_template_not_found(self, tmp_path, sample_checkpoint_data):
        """Test HTML generation returns None and logs error if template is missing."""
        collector = ResultsCollector(tmp_path, sample_checkpoint_data)
        summary_dir = tmp_path / "summary_reports"
        summary_dir.mkdir(exist_ok=True)
        report_context = {"run_metadata": {}} # Minimal context
        
        # Make get_template raise an error
        collector.jinja_env.get_template = MagicMock(side_effect=Exception("Template not found error simulation"))
        
        with patch.object(ResultsCollector.logger, 'error') as mock_log_error:
            output_path = collector.generate_html_report(report_context, summary_dir)
            assert output_path is None
            mock_log_error.assert_called_once_with(
                "Failed to load Jinja2 template 'report_template.html': Template not found error simulation", 
                exc_info=True
            )
            
    @patch.object(ResultsCollector, '_plot_matplotlib_bar', side_effect=Exception("Plotting Failed"))
    def test_generate_html_report_plot_error(self, mock_plot_error, tmp_path, sample_checkpoint_data):
        """Test HTML generation completes but logs error if a plot helper fails."""
        config = {"included_plots": ["accuracy_per_sample"]} # Ensure the failing plot is requested
        collector = ResultsCollector(tmp_path, sample_checkpoint_data, report_config=config)
        summary_dir = tmp_path / "summary_reports"
        summary_dir.mkdir(exist_ok=True)
        
        # Provide necessary data for the plot call to happen
        report_context = {
            'visualization_data': {'bar_charts': {'accuracy_per_sample': {'sample_id_str': ['s1'], 'accuracy': [0.9]}}},
            'run_metadata': {'timestamp': datetime.now()},
            'tables': {},
            'failed_samples': []
            # Other context keys added by generate_html_report itself
        }

        with patch.object(ResultsCollector.logger, 'error') as mock_log_error:
            output_path = collector.generate_html_report(report_context, summary_dir)

        # Report should still be generated, but the plot will be missing
        assert output_path is not None 
        assert output_path.is_file()
        # Check that the plotting error was logged by the helper method
        mock_plot_error.assert_called_once_with(
            "Error generating Matplotlib bar chart 'Accuracy per Sample': Plotting Failed", 
            exc_info=True
        )
        # Check the generated HTML (optional, ensure plot img tag is absent or empty src)
        html_content = output_path.read_text()
        assert "Accuracy per Sample" in html_content # Section title might still be there
        assert "data:image/png;base64" not in html_content # No plot image

        assert "--epibench-primary-color: #123456" in html_content # Check theme color applied
        
    @patch.dict('sys.modules', {'weasyprint': None})
    def test_generate_pdf_report_no_weasyprint(self, tmp_path, sample_checkpoint_data):
        """Test PDF generation fails gracefully if WeasyPrint is not installed."""
        collector = ResultsCollector(tmp_path, sample_checkpoint_data)
        summary_dir = tmp_path / "summary_reports"
        summary_dir.mkdir(exist_ok=True)
        dummy_html_path = summary_dir / "report.html"
        dummy_html_path.touch()

        with patch.object(ResultsCollector.logger, 'error') as mock_log_error:
            pdf_path = collector.generate_pdf_report(dummy_html_path, summary_dir)
            assert pdf_path is None
            mock_log_error.assert_called_once_with(
                "WeasyPrint library not found. Cannot generate PDF report. Please install it (`pip install WeasyPrint`)."
            )

    def test_generate_pdf_report_html_not_found(self, tmp_path, sample_checkpoint_data):
        """Test PDF generation fails if the input HTML file doesn't exist."""
        collector = ResultsCollector(tmp_path, sample_checkpoint_data)
        summary_dir = tmp_path / "summary_reports"
        summary_dir.mkdir(exist_ok=True)
        non_existent_html_path = summary_dir / "report_does_not_exist.html"

        with patch.object(ResultsCollector.logger, 'error') as mock_log_error:
            pdf_path = collector.generate_pdf_report(non_existent_html_path, summary_dir)
            assert pdf_path is None
            # Check for error log related to reading HTML
            assert mock_log_error.call_count == 1
            assert "Failed to read HTML report file" in mock_log_error.call_args[0][0]

    @patch('epibench.pipeline.results_collector.weasyprint.HTML')
    def test_generate_pdf_report_css_not_found(self, mock_html_weasy, tmp_path, sample_checkpoint_data):
        """Test PDF generation proceeds with warning if print CSS is missing."""
        collector = ResultsCollector(tmp_path, sample_checkpoint_data)
        summary_dir = tmp_path / "summary_reports"
        summary_dir.mkdir(exist_ok=True)
        dummy_html_path = summary_dir / "report.html"
        dummy_html_path.write_text("<html>HTML</html>")
        
        # Ensure CSS path does NOT exist
        css_path = Path(__file__).parent.parent / "epibench" / "pipeline" / "templates" / "report_print.css"
        if css_path.exists(): css_path.unlink() # Make sure it's gone for the test

        mock_html_instance = MagicMock()
        mock_html_weasy.return_value = mock_html_instance

        with patch.object(ResultsCollector.logger, 'warning') as mock_log_warning:
            pdf_path = collector.generate_pdf_report(dummy_html_path, summary_dir)
            assert pdf_path is not None # Should still succeed
            mock_log_warning.assert_called_once_with(
                f"Print CSS file not found at {css_path}. PDF styling might be incorrect."
            )
            # write_pdf should be called with stylesheets=None
            mock_html_instance.write_pdf.assert_called_once_with(str(pdf_path), stylesheets=None)

    @patch('epibench.pipeline.results_collector.weasyprint.HTML')
    @patch('epibench.pipeline.results_collector.weasyprint.CSS')
    def test_generate_pdf_report_write_pdf_error(self, mock_css, mock_html_weasy, tmp_path, sample_checkpoint_data):
        """Test PDF generation fails if weasyprint write_pdf raises exception."""
        collector = ResultsCollector(tmp_path, sample_checkpoint_data)
        summary_dir = tmp_path / "summary_reports"
        summary_dir.mkdir(exist_ok=True)
        dummy_html_path = summary_dir / "report.html"
        dummy_html_path.write_text("<html>HTML</html>")
        css_path = self.template_dir / "report_print.css"
        css_path.touch() # Ensure CSS exists

        mock_css_instance = MagicMock()
        mock_css.return_value = mock_css_instance
        mock_html_instance = MagicMock()
        # Make write_pdf raise an error
        mock_html_instance.write_pdf.side_effect = Exception("WeasyPrint PDF write failed")
        mock_html_weasy.return_value = mock_html_instance

        with patch.object(ResultsCollector.logger, 'error') as mock_log_error:
            pdf_path = collector.generate_pdf_report(dummy_html_path, summary_dir)
            assert pdf_path is None
            mock_html_instance.write_pdf.assert_called_once()
            mock_log_error.assert_called_once_with(
                "WeasyPrint failed to generate PDF: WeasyPrint PDF write failed", 
                exc_info=True
            )

    # --- Test report generation orchestration with config ---
    @patch.object(ResultsCollector, 'generate_html_report')
    @patch.object(ResultsCollector, 'generate_pdf_report')
    def test_generate_reports_with_config(self, mock_generate_pdf, mock_generate_html, base_output_dir, sample_checkpoint_data, setup_output_dirs):
        """Test that _generate_reports calls HTML and PDF generation helpers with config."""
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=None)
        summary_dir = setup_output_dirs['base'] / "summary_reports"
        summary_dir.mkdir(exist_ok=True)
        
        # Mock HTML generation to return a path
        mock_html_path = summary_dir / "mock_report.html"
        mock_generate_html.return_value = mock_html_path
        mock_html_path.touch() # Create the dummy file
        
        # Create sample report context (simpler version needed)
        report_context = {
            "run_metadata": {"timestamp": datetime.now()},
            "summary_stats_dict": {"num_samples_with_metrics": 1, "metric_summaries": {"mse": {"mean": 0.1}}},
            "visualization_data": {},
            "styled_tables_html": {
                "summary_statistics": "<table>...</table>",
                "detailed_metrics": "<table>...</table>"
            },
            "failed_samples": []
        }
        
        collector._generate_reports(report_context, summary_dir)
        
        # Check main HTML generation method was called
        mock_generate_html.assert_called_once_with(report_context, summary_dir)
        # Check PDF generation method was called with the HTML path
        mock_generate_pdf.assert_called_once_with(mock_html_path, summary_dir)
            
        # Check that intermediate files are still saved
        assert (summary_dir / "report_context_snapshot.json").is_file()
        assert (summary_dir / "summary_statistics.csv").is_file()
        # Detailed CSV saving is now skipped in _generate_reports
        # assert (summary_dir / "detailed_metrics.csv").is_file()

    @patch.object(ResultsCollector, 'generate_html_report', return_value=None)
    @patch.object(ResultsCollector, 'generate_pdf_report')
    def test_generate_reports_html_fails(self, mock_generate_pdf, mock_generate_html, base_output_dir, sample_checkpoint_data):
        """Test _generate_reports when HTML generation fails."""
        config = {"output_formats": ["html", "pdf"]} # Request both
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=config)
        summary_dir = base_output_dir / "summary_reports"
        summary_dir.mkdir(exist_ok=True)
        report_context = {"run_metadata": {}} # Minimal context
        
        with patch.object(ResultsCollector.logger, 'warning') as mock_log_warning, \
             patch.object(ResultsCollector.logger, 'error') as mock_log_error:
            collector._generate_reports(report_context, summary_dir)
            
            mock_generate_html.assert_called_once()
            # PDF should not be attempted if HTML failed
            mock_generate_pdf.assert_not_called()
            # Check for warning about HTML failing
            assert any("HTML report generation failed or was skipped." in call.args[0] for call in mock_log_warning.call_args_list)
            # Check for warning about not being able to generate PDF due to missing HTML
            assert any("cannot generate PDF" in call.args[0] for call in mock_log_warning.call_args_list)

    @patch.object(ResultsCollector, 'generate_html_report')
    @patch.object(ResultsCollector, 'generate_pdf_report', return_value=None)
    def test_generate_reports_pdf_fails(self, mock_generate_pdf, mock_generate_html, base_output_dir, sample_checkpoint_data):
        """Test _generate_reports when PDF generation fails."""
        config = {"output_formats": ["html", "pdf"]} # Request both
        collector = ResultsCollector(base_output_dir, sample_checkpoint_data, report_config=config)
        summary_dir = base_output_dir / "summary_reports"
        summary_dir.mkdir(exist_ok=True)
        report_context = {"run_metadata": {}} # Minimal context
        
        # Make HTML succeed and return a path
        mock_html_path = summary_dir / "report.html"
        mock_html_path.touch()
        mock_generate_html.return_value = mock_html_path
        
        with patch.object(ResultsCollector.logger, 'warning') as mock_log_warning:
            collector._generate_reports(report_context, summary_dir)
            
            mock_generate_html.assert_called_once()
            mock_generate_pdf.assert_called_once()
            # Check for warning about PDF failing
            assert any("PDF report generation failed or was skipped." in call.args[0] for call in mock_log_warning.call_args_list)

        if mock_html_path.exists(): mock_html_path.unlink()

    # --- Update test signature --- 
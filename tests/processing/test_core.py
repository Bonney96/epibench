import pytest
import pandas as pd
import numpy as np
import subprocess
from unittest.mock import patch, MagicMock, ANY

from epibench.processing.core import (
    wilson_score_interval,
    calculate_confidence_intervals,
    run_methfast,
    process_sample
)

# --- Tests for wilson_score_interval ---

def test_wilson_score_interval_basic():
    # Example values (e.g., 8 successes in 10 trials)
    lower, upper = wilson_score_interval(8, 10, confidence=0.95)
    # Expected values can be calculated or taken from reliable sources
    # Using scipy.stats.binomtest as a reference (though it uses a different method)
    # binomtest(8, 10).proportion_confint(confidence_level=0.95, method='wilson') -> (0.490..., 0.943...)
    assert 0.490 < lower < 0.491
    assert 0.943 < upper < 0.944

def test_wilson_score_interval_zero_trials():
    lower, upper = wilson_score_interval(0, 0)
    assert np.isnan(lower)
    assert np.isnan(upper)

def test_wilson_score_interval_zero_successes():
    lower, upper = wilson_score_interval(0, 10)
    # Expected: lower bound is 0
    assert lower == 0.0
    assert 0.0 < upper < 1.0 # Upper bound should be > 0

def test_wilson_score_interval_all_successes():
    lower, upper = wilson_score_interval(10, 10)
    # Expected: upper bound is 1
    assert 0.0 < lower < 1.0 # Lower bound should be < 1
    assert np.isclose(upper, 1.0)

def test_wilson_score_interval_high_confidence():
    lower_95, upper_95 = wilson_score_interval(5, 10, confidence=0.95)
    lower_99, upper_99 = wilson_score_interval(5, 10, confidence=0.99)
    # Higher confidence should result in a wider interval
    assert lower_99 < lower_95
    assert upper_99 > upper_95

# --- Tests for calculate_confidence_intervals ---

def test_calc_ci_with_coverage():
    df_in = pd.DataFrame({
        'chromosome': ['chr1', 'chr1'],
        'position': [100, 200],
        'count_methylated': [8, 2],
        'coverage': [10, 5]
    })
    df_out = calculate_confidence_intervals(df_in.copy())
    assert 'ci_lower' in df_out.columns
    assert 'ci_upper' in df_out.columns
    assert pd.api.types.is_numeric_dtype(df_out['ci_lower'])
    assert pd.api.types.is_numeric_dtype(df_out['ci_upper'])
    # Check values for the first row (8/10)
    l1, u1 = wilson_score_interval(8, 10)
    assert np.isclose(df_out.loc[0, 'ci_lower'], l1)
    assert np.isclose(df_out.loc[0, 'ci_upper'], u1)
    # Check values for the second row (2/5)
    l2, u2 = wilson_score_interval(2, 5)
    assert np.isclose(df_out.loc[1, 'ci_lower'], l2)
    assert np.isclose(df_out.loc[1, 'ci_upper'], u2)

def test_calc_ci_with_unmethylated():
    df_in = pd.DataFrame({
        'chromosome': ['chr1'],
        'position': [100],
        'count_methylated': [3],
        'count_unmethylated': [7]
    })
    df_out = calculate_confidence_intervals(df_in.copy())
    assert 'coverage' in df_out.columns # Should be added
    assert df_out.loc[0, 'coverage'] == 10
    assert 'ci_lower' in df_out.columns
    assert 'ci_upper' in df_out.columns
    l, u = wilson_score_interval(3, 10)
    assert np.isclose(df_out.loc[0, 'ci_lower'], l)
    assert np.isclose(df_out.loc[0, 'ci_upper'], u)

def test_calc_ci_missing_count_methylated():
    df_in = pd.DataFrame({'chromosome': ['chr1'], 'coverage': [10]})
    df_out = calculate_confidence_intervals(df_in.copy())
    # Should return original df and print error (capture print if needed)
    assert 'ci_lower' not in df_out.columns
    pd.testing.assert_frame_equal(df_in, df_out)

def test_calc_ci_missing_coverage_and_unmethylated():
    df_in = pd.DataFrame({'chromosome': ['chr1'], 'count_methylated': [5]})
    df_out = calculate_confidence_intervals(df_in.copy())
    assert 'ci_lower' not in df_out.columns
    pd.testing.assert_frame_equal(df_in, df_out)

def test_calc_ci_non_numeric_coverage():
    df_in = pd.DataFrame({
        'chromosome': ['chr1'],
        'count_methylated': [5],
        'coverage': ['ten'] # Non-numeric
    })
    df_out = calculate_confidence_intervals(df_in.copy())
    assert 'ci_lower' not in df_out.columns
    pd.testing.assert_frame_equal(df_in, df_out)

def test_calc_ci_non_numeric_unmethylated():
    df_in = pd.DataFrame({
        'chromosome': ['chr1'],
        'count_methylated': [5],
        'count_unmethylated': ['five'] # Non-numeric
    })
    df_out = calculate_confidence_intervals(df_in.copy())
    assert 'ci_lower' not in df_out.columns
    pd.testing.assert_frame_equal(df_in, df_out)

def test_calc_ci_non_numeric_methylated():
    df_in = pd.DataFrame({
        'chromosome': ['chr1'],
        'count_methylated': ['five'], # Non-numeric
        'coverage': [10]
    })
    df_out = calculate_confidence_intervals(df_in.copy())
    assert 'ci_lower' not in df_out.columns
    pd.testing.assert_frame_equal(df_in, df_out)

def test_calc_ci_empty_dataframe():
    df_in = pd.DataFrame(columns=['chromosome', 'position', 'count_methylated', 'coverage'])
    df_out = calculate_confidence_intervals(df_in.copy())
    assert 'ci_lower' in df_out.columns
    assert 'ci_upper' in df_out.columns
    assert df_out.empty

# --- Tests for run_methfast ---

@patch('subprocess.run')
def test_run_methfast_success(mock_run):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Methfast output"
    mock_result.stderr = ""
    mock_run.return_value = mock_result

    result = run_methfast("input.bam", "output_prefix", methfast_path="/path/to/methfast", options=["-v"])

    assert result is True
    mock_run.assert_called_once_with(
        ["/path/to/methfast", "-v", "input.bam", "output_prefix"],
        check=True, capture_output=True, text=True
    )

@patch('subprocess.run')
def test_run_methfast_command_not_found(mock_run):
    mock_run.side_effect = FileNotFoundError
    result = run_methfast("input.bam", "output_prefix")
    assert result is False
    mock_run.assert_called_once()

@patch('subprocess.run')
def test_run_methfast_called_process_error(mock_run):
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", output="stdout", stderr="stderr")
    result = run_methfast("input.bam", "output_prefix")
    assert result is False
    mock_run.assert_called_once()

@patch('subprocess.run')
def test_run_methfast_unexpected_error(mock_run):
    mock_run.side_effect = Exception("Unexpected error")
    result = run_methfast("input.bam", "output_prefix")
    assert result is False
    mock_run.assert_called_once()

# --- Tests for process_sample ---

def test_process_sample_basic_placeholder():
    sample_id = "sample1"
    config = {
        'methylation_files': {'sample1': 'path/to/meth1.cov'},
        'histone_files': {'sample1': ['path/to/hist1.bw']},
        'sequence_file': 'path/to/genome.fa',
        'output_directory': '/tmp/output'
    }
    # Since it's a placeholder, just check it runs and returns the expected dict structure
    result = process_sample(sample_id, config)
    assert isinstance(result, dict)
    assert 'processed_matrix_path' in result
    assert result['processed_matrix_path'] == f"/tmp/output/{sample_id}_processed_matrix.h5"

def test_process_sample_missing_methylation_input():
    sample_id = "sample2"
    config = {
        # Missing methylation_files entry for sample2
        'histone_files': {'sample2': ['path/to/hist2.bw']},
        'sequence_file': 'path/to/genome.fa',
        'output_directory': '/tmp/output'
    }
    result = process_sample(sample_id, config)
    assert result is None # Should fail and return None

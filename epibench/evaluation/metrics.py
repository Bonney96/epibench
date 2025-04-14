"""Evaluation Metrics and Utilities.

This module provides functions for calculating various regression metrics,
visualizing model performance, and performing statistical tests.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import logging
from typing import Tuple, Union, Dict, Optional

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Stats imports
from scipy import stats as st

logger = logging.getLogger(__name__)

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Union[Dict[str, float], None]:
    """Calculates various regression metrics.

    Args:
        y_true: Array of true target values.
        y_pred: Array of predicted values.

    Returns:
        A dictionary containing the calculated metrics (MSE, MAE, R2, 
        Pearson Correlation, Spearman Correlation), or None if inputs are invalid.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        logger.error("Inputs must be numpy arrays.")
        return None
    if y_true.shape != y_pred.shape:
        logger.error(f"Input shapes must match. Got {y_true.shape} and {y_pred.shape}.")
        return None
    if len(y_true) < 2: # Need at least 2 points for correlation
        logger.warning("Need at least 2 data points to calculate correlation.")
        # Calculate other metrics if possible
        try:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'pearson_r': np.nan, # Cannot compute correlation
                'pearson_p': np.nan,
                'spearman_r': np.nan,
                'spearman_p': np.nan,
            }
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}", exc_info=True)
            return None

    metrics = {}
    try:
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Calculate correlations, handle potential errors (e.g., constant input)
        try:
            pearson_corr, pearson_p = pearsonr(y_true, y_pred)
            metrics['pearson_r'] = pearson_corr
            metrics['pearson_p'] = pearson_p
        except ValueError as e:
            logger.warning(f"Could not calculate Pearson correlation: {e}. Setting to NaN.")
            metrics['pearson_r'] = np.nan
            metrics['pearson_p'] = np.nan

        try:
            spearman_corr, spearman_p = spearmanr(y_true, y_pred)
            metrics['spearman_r'] = spearman_corr
            metrics['spearman_p'] = spearman_p
        except ValueError as e:
            logger.warning(f"Could not calculate Spearman correlation: {e}. Setting to NaN.")
            metrics['spearman_r'] = np.nan
            metrics['spearman_p'] = np.nan
            
        logger.debug(f"Calculated metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating regression metrics: {e}", exc_info=True)
        return None

# --- Visualization Utilities --- 

def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                               title: str = "Predicted vs. Actual Values", 
                               xlabel: str = "Actual Values", 
                               ylabel: str = "Predicted Values", 
                               save_path: Optional[str] = None) -> None:
    """Generates a scatter plot of predicted vs. actual values.

    Args:
        y_true: Array of true target values.
        y_pred: Array of predicted values.
        title: Title for the plot.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        save_path: Optional path to save the plot image. If None, shows the plot.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        logger.error("Inputs must be numpy arrays for plotting.")
        return
    if y_true.shape != y_pred.shape:
        logger.error(f"Input shapes must match for plotting. Got {y_true.shape} and {y_pred.shape}.")
        return

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, label="Data points")
    
    # Add a line for perfect predictions (y=x)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect prediction")
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.axis('equal') # Ensure aspect ratio is equal
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Plot saved to {save_path}")
            plt.close() # Close the figure after saving
        except Exception as e:
            logger.error(f"Failed to save plot to {save_path}: {e}", exc_info=True)
            plt.show() # Show plot if saving failed
    else:
        plt.show()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, 
                   title: str = "Residual Plot", 
                   xlabel: str = "Predicted Values", 
                   ylabel: str = "Residuals (Actual - Predicted)", 
                   save_path: Optional[str] = None) -> None:
    """Generates a scatter plot of residuals vs. predicted values.

    Args:
        y_true: Array of true target values.
        y_pred: Array of predicted values.
        title: Title for the plot.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        save_path: Optional path to save the plot image. If None, shows the plot.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        logger.error("Inputs must be numpy arrays for plotting.")
        return
    if y_true.shape != y_pred.shape:
        logger.error(f"Input shapes must match for plotting. Got {y_true.shape} and {y_pred.shape}.")
        return

    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', label="Zero residual")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Plot saved to {save_path}")
            plt.close() # Close the figure after saving
        except Exception as e:
            logger.error(f"Failed to save plot to {save_path}: {e}", exc_info=True)
            plt.show() # Show plot if saving failed
    else:
        plt.show()

# --- Statistical Analysis Utilities --- 

def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Union[Tuple[float, float], None]:
    """Calculates the confidence interval for a given dataset.

    Args:
        data: A numpy array of sample data.
        confidence: The confidence level (e.g., 0.95 for 95% CI).

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval,
        or None if the calculation fails (e.g., insufficient data).
    """
    if not isinstance(data, np.ndarray):
        logger.error("Input data must be a numpy array.")
        return None
    if len(data) < 2:
        logger.warning("Need at least 2 data points to calculate confidence interval.")
        return None
    if not (0 < confidence < 1):
        logger.error("Confidence level must be between 0 and 1.")
        return None

    try:
        # Use T-distribution for confidence interval as sample size might be small
        mean = np.mean(data)
        sem = st.sem(data) # Standard error of the mean
        if sem == 0: # Handle case with zero standard error (constant data)
            logger.warning("Data has zero standard error. Confidence interval is the mean itself.")
            return (mean, mean)
            
        degrees_freedom = len(data) - 1
        
        # Calculate interval using the t-distribution's percent point function (ppf)
        interval = st.t.interval(confidence, degrees_freedom, loc=mean, scale=sem)
        logger.debug(f"Calculated {confidence*100:.1f}% CI: {interval}")
        return interval
    except Exception as e:
        logger.error(f"Error calculating confidence interval: {e}", exc_info=True)
        return None

def perform_t_test(sample1: np.ndarray, sample2: np.ndarray, 
                   equal_var: bool = True) -> Union[Tuple[float, float], None]:
    """Performs an independent two-sample t-test.

    Args:
        sample1: First sample data (numpy array).
        sample2: Second sample data (numpy array).
        equal_var: If True (default), perform a standard independent 2 sample test
                     that assumes equal population variances. If False, perform
                     Welch's t-test, which does not assume equal population variance.

    Returns:
        A tuple containing the t-statistic and the p-value, or None if the
        test cannot be performed (e.g., insufficient data).
    """
    if not isinstance(sample1, np.ndarray) or not isinstance(sample2, np.ndarray):
        logger.error("Inputs must be numpy arrays for t-test.")
        return None
    if len(sample1) < 2 or len(sample2) < 2:
        logger.warning("Need at least 2 data points in each sample for t-test.")
        return None

    try:
        t_stat, p_value = st.ttest_ind(sample1, sample2, equal_var=equal_var, nan_policy='omit')
        logger.debug(f"Performed t-test (equal_var={equal_var}): t={t_stat:.4f}, p={p_value:.4f}")
        return t_stat, p_value
    except Exception as e:
        logger.error(f"Error performing t-test: {e}", exc_info=True)
        return None

# Example Usage (can be removed or moved to tests later)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    # Example data
    y_true_example = np.array([1, 2, 3, 4, 5, 6])
    y_pred_example = np.array([1.1, 1.9, 3.2, 3.8, 5.3, 6.1])
    y_const_example = np.array([3, 3, 3, 3, 3, 3])
    y_short_true = np.array([1])
    y_short_pred = np.array([1.1])

    print("--- Standard Example ---")
    metrics_result = calculate_regression_metrics(y_true_example, y_pred_example)
    if metrics_result:
        for k, v in metrics_result.items():
            print(f"{k}: {v:.4f}")

    print("\n--- Constant Prediction Example (Correlation Warning) ---")
    metrics_const = calculate_regression_metrics(y_true_example, y_const_example)
    if metrics_const:
         for k, v in metrics_const.items():
            print(f"{k}: {v:.4f}")

    print("\n--- Short Input Example (Correlation Warning) ---")
    metrics_short = calculate_regression_metrics(y_short_true, y_short_pred)
    if metrics_short:
         for k, v in metrics_short.items():
            print(f"{k}: {v:.4f}")
            
    print("\n--- Mismatched Shape Example (Error) ---")
    y_mismatch = np.array([1, 2])
    calculate_regression_metrics(y_true_example, y_mismatch) 
    
    print("\n--- Non-Array Input Example (Error) ---")
    calculate_regression_metrics([1, 2, 3], y_pred_example)
    
    print("\n--- Plotting Examples ---")
    # Ensure you have matplotlib and seaborn installed: pip install matplotlib seaborn
    plot_predictions_vs_actual(y_true_example, y_pred_example, save_path="pred_vs_actual.png")
    plot_residuals(y_true_example, y_pred_example, save_path="residuals.png")
    
    # Example showing plot without saving
    # plot_predictions_vs_actual(y_true_example, y_const_example, title="Constant Prediction")
    # plot_residuals(y_true_example, y_const_example, title="Residuals for Constant Prediction") 

    print("\n--- Statistical Analysis Examples ---")
    ci = calculate_confidence_interval(y_pred_example)
    if ci:
        print(f"95% Confidence Interval for y_pred: ({ci[0]:.4f}, {ci[1]:.4f})")
    
    # Example t-test data
    group1 = np.random.normal(loc=5.0, scale=1.0, size=30)
    group2 = np.random.normal(loc=5.5, scale=1.0, size=30) # Slightly different mean
    group3 = np.random.normal(loc=5.0, scale=1.5, size=30) # Different variance
    
    t_test_result = perform_t_test(group1, group2)
    if t_test_result:
        print(f"T-test (Group1 vs Group2, equal_var=True): t={t_test_result[0]:.4f}, p={t_test_result[1]:.4f}")
        
    welch_test_result = perform_t_test(group1, group3, equal_var=False)
    if welch_test_result:
        print(f"Welch's T-test (Group1 vs Group3, equal_var=False): t={welch_test_result[0]:.4f}, p={welch_test_result[1]:.4f}") 
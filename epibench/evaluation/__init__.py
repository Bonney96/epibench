# -*- coding: utf-8 -*-
"""EpiBench Evaluation Utilities Module.

This package contains modules for evaluating model performance,
including metric calculation, visualization, and statistical tests.
"""

from .metrics import (
    calculate_regression_metrics,
    plot_predictions_vs_actual,
    plot_residuals,
    calculate_confidence_interval,
    perform_t_test
)

__all__ = [
    "calculate_regression_metrics",
    "plot_predictions_vs_actual",
    "plot_residuals",
    "calculate_confidence_interval",
    "perform_t_test"
] 
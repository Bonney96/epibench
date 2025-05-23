"""Utility functions for calculating derived metrics from evaluation results."""

import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def calculate_derived_metrics(base_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate additional derived metrics from base evaluation metrics.
    
    Args:
        base_metrics: Dictionary containing base metrics (mse, r2, mae, etc.)
        
    Returns:
        Dictionary containing derived metrics
    """
    derived = {}
    
    try:
        # Calculate RMSE from MSE
        if 'mse' in base_metrics and base_metrics['mse'] is not None:
            derived['rmse'] = float(np.sqrt(base_metrics['mse']))
        
        # Calculate adjusted R-squared if sample size is available
        if 'r2' in base_metrics and base_metrics['r2'] is not None:
            r2 = base_metrics['r2']
            # Store basic R2 interpretation
            if r2 >= 0.9:
                derived['r2_interpretation'] = 'excellent'
            elif r2 >= 0.8:
                derived['r2_interpretation'] = 'good'
            elif r2 >= 0.7:
                derived['r2_interpretation'] = 'moderate'
            elif r2 >= 0.5:
                derived['r2_interpretation'] = 'weak'
            else:
                derived['r2_interpretation'] = 'poor'
        
        # Calculate normalized metrics if range is available
        if 'mae' in base_metrics and base_metrics['mae'] is not None:
            mae = base_metrics['mae']
            # Could normalize by target range if available
            derived['mae_normalized'] = mae  # Placeholder for future normalization
        
        # Correlation strength interpretation
        for corr_type in ['pearson_r', 'spearman_r']:
            if corr_type in base_metrics and base_metrics[corr_type] is not None:
                r_value = abs(base_metrics[corr_type])
                if r_value >= 0.9:
                    strength = 'very_strong'
                elif r_value >= 0.7:
                    strength = 'strong'
                elif r_value >= 0.5:
                    strength = 'moderate'
                elif r_value >= 0.3:
                    strength = 'weak'
                else:
                    strength = 'very_weak'
                    
                derived[f'{corr_type}_strength'] = strength
                
                # Check statistical significance
                p_key = corr_type.replace('_r', '_p')
                if p_key in base_metrics and base_metrics[p_key] is not None:
                    p_value = base_metrics[p_key]
                    derived[f'{corr_type}_significant'] = p_value < 0.05
                    derived[f'{corr_type}_highly_significant'] = p_value < 0.01
        
    except Exception as e:
        logger.error(f"Error calculating derived metrics: {e}")
    
    return derived


def calculate_performance_score(metrics: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> Optional[float]:
    """
    Calculate a composite performance score from multiple metrics.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        weights: Optional weights for each metric (default: equal weights)
        
    Returns:
        Composite performance score (0-1 scale) or None if calculation fails
    """
    if weights is None:
        # Default weights emphasizing prediction accuracy
        weights = {
            'r2': 0.3,
            'mse_inverse': 0.2,  # Will use 1/(1+mse)
            'mae_inverse': 0.2,  # Will use 1/(1+mae)
            'pearson_r': 0.15,
            'spearman_r': 0.15
        }
    
    try:
        score_components = []
        total_weight = 0
        
        # R-squared (already 0-1 scale)
        if 'r2' in metrics and metrics['r2'] is not None and 'r2' in weights:
            # Clip to [0, 1] range
            r2_score = max(0, min(1, metrics['r2']))
            score_components.append(r2_score * weights['r2'])
            total_weight += weights['r2']
        
        # MSE inverse score
        if 'mse' in metrics and metrics['mse'] is not None and 'mse_inverse' in weights:
            # Transform MSE to 0-1 scale (lower is better)
            mse_score = 1 / (1 + metrics['mse'])
            score_components.append(mse_score * weights['mse_inverse'])
            total_weight += weights['mse_inverse']
        
        # MAE inverse score
        if 'mae' in metrics and metrics['mae'] is not None and 'mae_inverse' in weights:
            # Transform MAE to 0-1 scale (lower is better)
            mae_score = 1 / (1 + metrics['mae'])
            score_components.append(mae_score * weights['mae_inverse'])
            total_weight += weights['mae_inverse']
        
        # Correlation scores (use absolute value, already -1 to 1)
        for corr_type in ['pearson_r', 'spearman_r']:
            if corr_type in metrics and metrics[corr_type] is not None and corr_type in weights:
                # Use absolute value and clip to [0, 1]
                corr_score = abs(metrics[corr_type])
                score_components.append(corr_score * weights[corr_type])
                total_weight += weights[corr_type]
        
        if total_weight > 0 and score_components:
            # Normalize by actual total weight
            composite_score = sum(score_components) / total_weight
            return float(composite_score)
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error calculating performance score: {e}")
        return None


def compare_metrics(metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two sets of metrics and calculate improvements.
    
    Args:
        metrics1: First set of metrics (baseline)
        metrics2: Second set of metrics (comparison)
        
    Returns:
        Dictionary containing comparison results
    """
    comparison = {}
    
    try:
        # Compare MSE (lower is better)
        if 'mse' in metrics1 and 'mse' in metrics2:
            mse1, mse2 = metrics1['mse'], metrics2['mse']
            if mse1 > 0:
                comparison['mse_improvement'] = float((mse1 - mse2) / mse1 * 100)
                comparison['mse_better'] = mse2 < mse1
        
        # Compare R2 (higher is better)
        if 'r2' in metrics1 and 'r2' in metrics2:
            r2_1, r2_2 = metrics1['r2'], metrics2['r2']
            comparison['r2_improvement'] = float(r2_2 - r2_1)
            comparison['r2_better'] = r2_2 > r2_1
        
        # Compare MAE (lower is better)
        if 'mae' in metrics1 and 'mae' in metrics2:
            mae1, mae2 = metrics1['mae'], metrics2['mae']
            if mae1 > 0:
                comparison['mae_improvement'] = float((mae1 - mae2) / mae1 * 100)
                comparison['mae_better'] = mae2 < mae1
        
        # Compare correlations (higher absolute value is better)
        for corr_type in ['pearson_r', 'spearman_r']:
            if corr_type in metrics1 and corr_type in metrics2:
                corr1 = abs(metrics1[corr_type])
                corr2 = abs(metrics2[corr_type])
                comparison[f'{corr_type}_improvement'] = float(corr2 - corr1)
                comparison[f'{corr_type}_better'] = corr2 > corr1
        
        # Overall assessment
        better_count = sum(1 for k, v in comparison.items() if k.endswith('_better') and v)
        total_metrics = sum(1 for k in comparison.keys() if k.endswith('_better'))
        
        if total_metrics > 0:
            comparison['overall_better'] = better_count > total_metrics / 2
            comparison['improvement_ratio'] = better_count / total_metrics
        
    except Exception as e:
        logger.error(f"Error comparing metrics: {e}")
    
    return comparison


def aggregate_sample_metrics(sample_metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple samples with advanced statistics.
    
    Args:
        sample_metrics_list: List of metric dictionaries from different samples
        
    Returns:
        Dictionary containing aggregated statistics
    """
    if not sample_metrics_list:
        return {}
    
    aggregated = {}
    metric_names = set()
    
    # Collect all metric names
    for metrics in sample_metrics_list:
        metric_names.update(metrics.keys())
    
    # Remove non-numeric fields
    metric_names.discard('sample_id')
    metric_names.discard('error')
    
    for metric_name in metric_names:
        values = []
        for metrics in sample_metrics_list:
            if metric_name in metrics and metrics[metric_name] is not None:
                try:
                    values.append(float(metrics[metric_name]))
                except (ValueError, TypeError):
                    continue
        
        if values:
            try:
                aggregated[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'q1': float(np.percentile(values, 25)),
                    'q3': float(np.percentile(values, 75)),
                    'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
                    'cv': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else None,
                    'count': len(values)
                }
                
                # Add outlier detection
                q1 = aggregated[metric_name]['q1']
                q3 = aggregated[metric_name]['q3']
                iqr = aggregated[metric_name]['iqr']
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = [v for v in values if v < lower_bound or v > upper_bound]
                aggregated[metric_name]['outlier_count'] = len(outliers)
                aggregated[metric_name]['outlier_ratio'] = len(outliers) / len(values) if values else 0
                
            except Exception as e:
                logger.error(f"Error aggregating metric {metric_name}: {e}")
    
    return aggregated 
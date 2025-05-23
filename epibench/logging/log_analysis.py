"""Advanced analysis functions for EpiBench logs including time series and correlation analysis."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from collections import defaultdict

logger = logging.getLogger(__name__)


class LogAnalyzer:
    """Advanced analysis methods for EpiBench logs."""
    
    def __init__(self, logs: List[Dict[str, Any]]):
        """
        Initialize analyzer with a list of logs.
        
        Args:
            logs: List of log dictionaries to analyze
        """
        self.logs = logs
        self._metrics_df = None
        self._config_df = None
        self._time_series_df = None
        
    def _prepare_metrics_dataframe(self) -> pd.DataFrame:
        """Prepare a DataFrame of metrics from logs."""
        if self._metrics_df is not None:
            return self._metrics_df
        
        metrics_data = []
        for log in self.logs:
            row = {
                'execution_id': log.get('execution_metadata', {}).get('execution_id'),
                'sample_id': log.get('input_configuration', {}).get('sample_id'),
                'timestamp': pd.to_datetime(
                    log.get('execution_metadata', {}).get('timestamp', {}).get('start', '')
                ),
                'duration': log.get('runtime_information', {}).get('duration_seconds')
            }
            
            # Add performance metrics
            perf_metrics = log.get('performance_metrics', {})
            row.update({
                'mse': perf_metrics.get('mse'),
                'r_squared': perf_metrics.get('r_squared'),
                'mae': perf_metrics.get('mae')
            })
            
            # Add correlations
            correlations = perf_metrics.get('correlations', {})
            if 'pearson' in correlations:
                row['pearson_r'] = correlations['pearson'].get('r')
            if 'spearman' in correlations:
                row['spearman_r'] = correlations['spearman'].get('r')
            
            metrics_data.append(row)
        
        self._metrics_df = pd.DataFrame(metrics_data)
        return self._metrics_df
    
    def _prepare_config_dataframe(self) -> pd.DataFrame:
        """Prepare a DataFrame of configuration parameters from logs."""
        if self._config_df is not None:
            return self._config_df
        
        config_data = []
        for log in self.logs:
            row = {
                'execution_id': log.get('execution_metadata', {}).get('execution_id')
            }
            
            # Extract key configuration parameters
            config_params = log.get('configuration_parameters', {})
            key_params = config_params.get('key_parameters', {})
            
            # Flatten key parameters
            for param_name, param_value in key_params.items():
                if isinstance(param_value, (int, float, str, bool)):
                    row[f'config_{param_name}'] = param_value
            
            config_data.append(row)
        
        self._config_df = pd.DataFrame(config_data)
        return self._config_df
    
    def time_series_analysis(self, metric: str, 
                           window_size: Optional[int] = None,
                           resample_freq: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform time series analysis on a specific metric.
        
        Args:
            metric: Name of the metric to analyze
            window_size: Size of rolling window for smoothing
            resample_freq: Frequency for resampling (e.g., '1D', '1H')
            
        Returns:
            Dictionary containing time series analysis results
        """
        df = self._prepare_metrics_dataframe()
        
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in logs")
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Filter out null values
        ts_data = df[['timestamp', metric]].dropna()
        
        if len(ts_data) < 2:
            logger.warning(f"Insufficient data points for time series analysis of {metric}")
            return {}
        
        results = {
            'metric': metric,
            'data_points': len(ts_data),
            'time_range': {
                'start': ts_data['timestamp'].min().isoformat(),
                'end': ts_data['timestamp'].max().isoformat()
            }
        }
        
        # Basic statistics
        results['statistics'] = {
            'mean': float(ts_data[metric].mean()),
            'std': float(ts_data[metric].std()),
            'min': float(ts_data[metric].min()),
            'max': float(ts_data[metric].max()),
            'trend': self._calculate_trend(ts_data['timestamp'], ts_data[metric])
        }
        
        # Resample if requested
        if resample_freq:
            ts_data.set_index('timestamp', inplace=True)
            resampled = ts_data[metric].resample(resample_freq).agg(['mean', 'std', 'count'])
            results['resampled_data'] = resampled.to_dict()
        
        # Rolling statistics if window size provided
        if window_size and len(ts_data) >= window_size:
            ts_data[f'{metric}_rolling_mean'] = ts_data[metric].rolling(window=window_size).mean()
            ts_data[f'{metric}_rolling_std'] = ts_data[metric].rolling(window=window_size).std()
            
            results['rolling_stats'] = {
                'window_size': window_size,
                'final_rolling_mean': float(ts_data[f'{metric}_rolling_mean'].iloc[-1]),
                'final_rolling_std': float(ts_data[f'{metric}_rolling_std'].iloc[-1])
            }
        
        # Detect anomalies
        anomalies = self._detect_anomalies(ts_data[metric].values)
        results['anomalies'] = {
            'count': len(anomalies),
            'indices': anomalies.tolist() if len(anomalies) > 0 else []
        }
        
        # Autocorrelation
        if len(ts_data) > 10:
            results['autocorrelation'] = self._calculate_autocorrelation(ts_data[metric].values)
        
        return results
    
    def _calculate_trend(self, timestamps: pd.Series, values: pd.Series) -> Dict[str, float]:
        """Calculate trend using linear regression."""
        # Convert timestamps to numeric values
        x = (timestamps - timestamps.min()).dt.total_seconds()
        
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        return {
            'slope': float(slope),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'trend_direction': 'increasing' if slope > 0 else 'decreasing'
        }
    
    def _detect_anomalies(self, values: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
        """Detect anomalies using z-score method."""
        if len(values) < 3:
            return np.array([])
        
        z_scores = np.abs(stats.zscore(values))
        return np.where(z_scores > z_threshold)[0]
    
    def _calculate_autocorrelation(self, values: np.ndarray, max_lag: int = 10) -> Dict[int, float]:
        """Calculate autocorrelation for different lags."""
        autocorr = {}
        n = len(values)
        
        for lag in range(1, min(max_lag + 1, n // 2)):
            if n > lag:
                correlation = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                if not np.isnan(correlation):
                    autocorr[lag] = float(correlation)
        
        return autocorr
    
    def correlation_analysis(self, target_metric: str = 'r_squared',
                           config_params: Optional[List[str]] = None,
                           method: str = 'pearson') -> Dict[str, Any]:
        """
        Analyze correlations between configuration parameters and a target metric.
        
        Args:
            target_metric: The performance metric to correlate against
            config_params: List of config parameters to analyze (None = all)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dictionary containing correlation analysis results
        """
        # Prepare dataframes
        metrics_df = self._prepare_metrics_dataframe()
        config_df = self._prepare_config_dataframe()
        
        # Merge on execution_id
        df = pd.merge(metrics_df, config_df, on='execution_id')
        
        if target_metric not in df.columns:
            raise ValueError(f"Target metric '{target_metric}' not found")
        
        # Filter out rows with missing target metric
        df = df.dropna(subset=[target_metric])
        
        if len(df) < 3:
            logger.warning("Insufficient data for correlation analysis")
            return {}
        
        # Determine config parameters to analyze
        if config_params is None:
            config_params = [col for col in df.columns if col.startswith('config_')]
        
        results = {
            'target_metric': target_metric,
            'method': method,
            'sample_size': len(df),
            'correlations': {}
        }
        
        # Calculate correlations
        for param in config_params:
            if param not in df.columns:
                continue
            
            # Convert to numeric if possible
            param_values = pd.to_numeric(df[param], errors='coerce')
            
            # Skip if not numeric or insufficient variation
            if param_values.isna().all() or param_values.nunique() < 2:
                continue
            
            # Calculate correlation
            valid_mask = ~param_values.isna()
            if valid_mask.sum() < 3:
                continue
            
            x = param_values[valid_mask]
            y = df[target_metric][valid_mask]
            
            try:
                if method == 'pearson':
                    corr, p_value = pearsonr(x, y)
                elif method == 'spearman':
                    corr, p_value = spearmanr(x, y)
                elif method == 'kendall':
                    corr, p_value = kendalltau(x, y)
                else:
                    raise ValueError(f"Unknown correlation method: {method}")
                
                results['correlations'][param] = {
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'n_samples': len(x)
                }
            except Exception as e:
                logger.warning(f"Failed to calculate correlation for {param}: {e}")
        
        # Sort by absolute correlation
        sorted_corr = sorted(
            results['correlations'].items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True
        )
        
        results['top_correlations'] = dict(sorted_corr[:10])
        
        return results
    
    def performance_trends_by_config(self, config_param: str, 
                                   metric: str = 'r_squared') -> Dict[str, Any]:
        """
        Analyze how performance metrics vary with a configuration parameter.
        
        Args:
            config_param: Name of configuration parameter
            metric: Performance metric to analyze
            
        Returns:
            Dictionary containing trend analysis
        """
        # Prepare data
        metrics_df = self._prepare_metrics_dataframe()
        config_df = self._prepare_config_dataframe()
        df = pd.merge(metrics_df, config_df, on='execution_id')
        
        config_col = f'config_{config_param}' if not config_param.startswith('config_') else config_param
        
        if config_col not in df.columns:
            raise ValueError(f"Configuration parameter '{config_param}' not found")
        
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found")
        
        # Group by config value
        grouped = df.groupby(config_col)[metric].agg(['mean', 'std', 'count'])
        grouped = grouped.dropna()
        
        results = {
            'config_param': config_param,
            'metric': metric,
            'groups': grouped.to_dict(),
            'overall_trend': None
        }
        
        # If numeric, calculate trend
        if pd.api.types.is_numeric_dtype(df[config_col]):
            valid_data = df[[config_col, metric]].dropna()
            if len(valid_data) > 2:
                trend = self._calculate_trend(
                    pd.Series(range(len(valid_data))),
                    valid_data[metric]
                )
                results['overall_trend'] = trend
        
        return results
    
    def experiment_clustering(self, n_clusters: int = 3,
                            features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Cluster experiments based on configuration and performance.
        
        Args:
            n_clusters: Number of clusters to create
            features: List of features to use for clustering
            
        Returns:
            Dictionary containing clustering results
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error("scikit-learn not installed. Cannot perform clustering.")
            return {}
        
        # Prepare data
        metrics_df = self._prepare_metrics_dataframe()
        config_df = self._prepare_config_dataframe()
        df = pd.merge(metrics_df, config_df, on='execution_id')
        
        # Select features
        if features is None:
            # Use numeric columns
            features = []
            for col in df.columns:
                if col.startswith('config_') or col in ['mse', 'r_squared', 'mae']:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        features.append(col)
        
        # Filter and prepare data
        feature_data = df[features].dropna()
        
        if len(feature_data) < n_clusters:
            logger.warning("Insufficient data for clustering")
            return {}
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Analyze clusters
        results = {
            'n_clusters': n_clusters,
            'features_used': features,
            'cluster_sizes': pd.Series(clusters).value_counts().to_dict(),
            'cluster_centers': {}
        }
        
        # Get cluster characteristics
        feature_data['cluster'] = clusters
        for cluster_id in range(n_clusters):
            cluster_data = feature_data[feature_data['cluster'] == cluster_id]
            results['cluster_centers'][f'cluster_{cluster_id}'] = {
                feat: float(cluster_data[feat].mean())
                for feat in features
            }
        
        # Calculate silhouette score
        if len(feature_data) > n_clusters:
            from sklearn.metrics import silhouette_score
            score = silhouette_score(scaled_features, clusters)
            results['silhouette_score'] = float(score)
        
        return results
    
    def statistical_tests(self, group_by: str, metric: str = 'r_squared',
                         test_type: str = 'auto') -> Dict[str, Any]:
        """
        Perform statistical tests to compare groups.
        
        Args:
            group_by: Configuration parameter to group by
            metric: Metric to compare
            test_type: Type of test ('auto', 't-test', 'anova', 'kruskal')
            
        Returns:
            Dictionary containing test results
        """
        # Prepare data
        metrics_df = self._prepare_metrics_dataframe()
        config_df = self._prepare_config_dataframe()
        df = pd.merge(metrics_df, config_df, on='execution_id')
        
        group_col = f'config_{group_by}' if not group_by.startswith('config_') else group_by
        
        if group_col not in df.columns:
            raise ValueError(f"Grouping parameter '{group_by}' not found")
        
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found")
        
        # Group data
        groups = []
        group_names = []
        for name, group in df.groupby(group_col):
            values = group[metric].dropna()
            if len(values) > 0:
                groups.append(values.values)
                group_names.append(str(name))
        
        if len(groups) < 2:
            logger.warning("Need at least 2 groups for statistical testing")
            return {}
        
        results = {
            'group_by': group_by,
            'metric': metric,
            'n_groups': len(groups),
            'group_sizes': {name: len(g) for name, g in zip(group_names, groups)}
        }
        
        # Choose test type
        if test_type == 'auto':
            if len(groups) == 2:
                test_type = 't-test'
            else:
                test_type = 'anova'
        
        # Perform test
        if test_type == 't-test' and len(groups) == 2:
            statistic, p_value = stats.ttest_ind(groups[0], groups[1])
            results['test'] = {
                'type': 't-test',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        elif test_type == 'anova':
            statistic, p_value = stats.f_oneway(*groups)
            results['test'] = {
                'type': 'one-way ANOVA',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        elif test_type == 'kruskal':
            statistic, p_value = stats.kruskal(*groups)
            results['test'] = {
                'type': 'Kruskal-Wallis',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        
        # Add effect size for t-test
        if test_type == 't-test' and len(groups) == 2:
            # Cohen's d
            pooled_std = np.sqrt(((len(groups[0])-1)*np.std(groups[0])**2 + 
                                 (len(groups[1])-1)*np.std(groups[1])**2) / 
                                (len(groups[0]) + len(groups[1]) - 2))
            cohens_d = (np.mean(groups[0]) - np.mean(groups[1])) / pooled_std
            results['test']['effect_size'] = float(cohens_d)
        
        return results
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of all analyses.
        
        Returns:
            Dictionary containing summary of all analyses
        """
        report = {
            'overview': {
                'total_experiments': len(self.logs),
                'date_range': None,
                'unique_samples': set(),
                'completion_rate': 0.0
            },
            'performance_summary': {},
            'best_configurations': {},
            'key_insights': []
        }
        
        # Calculate overview stats
        metrics_df = self._prepare_metrics_dataframe()
        
        if not metrics_df.empty:
            report['overview']['date_range'] = {
                'start': metrics_df['timestamp'].min().isoformat(),
                'end': metrics_df['timestamp'].max().isoformat()
            }
            report['overview']['unique_samples'] = len(metrics_df['sample_id'].unique())
            
            # Completion rate
            completed = sum(1 for log in self.logs 
                          if log.get('execution_metadata', {}).get('status') == 'completed')
            report['overview']['completion_rate'] = completed / len(self.logs) if self.logs else 0
        
        # Performance summary
        for metric in ['mse', 'r_squared', 'mae']:
            if metric in metrics_df.columns:
                values = metrics_df[metric].dropna()
                if len(values) > 0:
                    report['performance_summary'][metric] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median())
                    }
        
        # Find best configurations
        if 'r_squared' in metrics_df.columns:
            best_idx = metrics_df['r_squared'].idxmax()
            if not pd.isna(best_idx):
                best_log = self.logs[best_idx]
                report['best_configurations']['highest_r_squared'] = {
                    'execution_id': best_log.get('execution_metadata', {}).get('execution_id'),
                    'r_squared': float(metrics_df.loc[best_idx, 'r_squared']),
                    'key_params': best_log.get('configuration_parameters', {}).get('key_parameters', {})
                }
        
        # Generate insights
        if len(metrics_df) > 10:
            # Trend insight
            r2_trend = self._calculate_trend(
                pd.Series(range(len(metrics_df))),
                metrics_df['r_squared'].fillna(0)
            )
            if r2_trend['p_value'] < 0.05:
                direction = "improving" if r2_trend['slope'] > 0 else "declining"
                report['key_insights'].append(
                    f"Performance shows a statistically significant {direction} trend over time"
                )
        
        return report


# Convenience functions for common analyses
def analyze_experiment_series(log_directory: str, 
                            output_format: str = 'dict') -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Analyze a series of experiments from a log directory.
    
    Args:
        log_directory: Path to directory containing log files
        output_format: Format for results ('dict' or 'dataframe')
        
    Returns:
        Analysis results in requested format
    """
    from .log_query import LogQuery
    
    # Load all logs
    query = LogQuery(log_directory)
    logs = query.execute()
    
    if not logs:
        logger.warning("No logs found in directory")
        return {} if output_format == 'dict' else pd.DataFrame()
    
    # Perform analysis
    analyzer = LogAnalyzer(logs)
    
    results = {
        'summary': analyzer.generate_summary_report(),
        'r2_trends': analyzer.time_series_analysis('r_squared'),
        'correlations': analyzer.correlation_analysis()
    }
    
    if output_format == 'dataframe':
        # Convert to DataFrame format
        summary_df = pd.DataFrame([results['summary']['performance_summary']])
        return summary_df
    
    return results


def compare_experiment_groups(log_directory: str,
                            group_by: str,
                            metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compare different groups of experiments.
    
    Args:
        log_directory: Path to directory containing log files
        group_by: Configuration parameter to group by
        metrics: List of metrics to compare (default: all)
        
    Returns:
        Comparison results
    """
    from .log_query import LogQuery
    
    # Load logs
    query = LogQuery(log_directory)
    logs = query.execute()
    
    if not logs:
        return {}
    
    analyzer = LogAnalyzer(logs)
    
    if metrics is None:
        metrics = ['mse', 'r_squared', 'mae']
    
    results = {}
    for metric in metrics:
        try:
            test_results = analyzer.statistical_tests(group_by, metric)
            if test_results:
                results[metric] = test_results
        except Exception as e:
            logger.warning(f"Failed to compare {metric}: {e}")
    
    return results 
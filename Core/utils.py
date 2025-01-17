import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy.stats import trim_mean

def validate_input_data(df: pd.DataFrame) -> bool:
    """Validates input data format and quality."""
    if df.empty:
        raise ValueError("Empty dataset")
    if df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) > 0.3:
        raise ValueError("Too many missing values (>30%)")
    return True

def normalize_expression_data(df: pd.DataFrame, method: str = 'tmm', trim_quantile: float = 0.05) -> pd.DataFrame:
    """
    Enhanced normalization with robust statistical methods.
    
    Args:
        df: Expression data matrix
        method: Normalization method ('tmm', 'quantile', 'rle')
        trim_quantile: Trimming threshold for robust mean calculation
    """
    if method not in ['tmm', 'quantile', 'rle']:
        raise ValueError(f"Unsupported normalization method: {method}")
        
    if method == 'tmm':
        scaling_factors = calculate_tmm_factors(df, trim_quantile)
        return df.div(scaling_factors, axis=0)
    elif method == 'quantile':
        return pd.DataFrame(
            stats.mstats.rankdata(df, axis=1),
            index=df.index, columns=df.columns
        )
    else:  # RLE method
        geometric_means = stats.gmean(df, axis=1)
        factors = df.div(geometric_means, axis=0).median()
        return df.div(factors, axis=1)

def calculate_tmm_factors(df: pd.DataFrame, trim_quantile: float = 0.05) -> np.ndarray:
    """
    Enhanced TMM calculation with robust statistics.
    """
    reference_sample = trim_mean(df, trim_quantile, axis=1)
    factors = np.zeros(df.shape[1])
    
    for idx, sample in enumerate(df.columns):
        factors[idx] = _calc_tmm_factor(
            df[sample], 
            reference_sample,
            trim_quantile
        )
    
    return factors

def _calc_tmm_factor(sample: pd.Series, reference: pd.Series, trim_quantile: float) -> float:
    """
    Improved TMM factor calculation with outlier handling.
    """
    # Remove extreme values
    mask = (sample > np.quantile(sample, trim_quantile)) & \
           (sample < np.quantile(sample, 1 - trim_quantile))
    
    sample_clean = sample[mask]
    reference_clean = reference[mask]
    
    # Calculate weighted trimmed mean
    weights = 1 / (sample_clean.std() + reference_clean.std())
    return np.average(sample_clean / reference_clean, weights=weights)

def calculate_quality_metrics(df: pd.DataFrame) -> Dict[str, Union[float, np.ndarray]]:
    """
    Enhanced quality metrics with additional statistical measures.
    """
    metrics = {
        'expression_distribution': df.values.ravel(),
        'sample_correlation': df.corr(),
        'cv_distribution': df.std() / df.mean(),
        'gene_detection': (df > 0).sum(),
        'mad_scores': stats.median_abs_deviation(df, axis=1),
        'expression_entropy': -np.sum(df * np.log2(df + 1e-10), axis=1),
        'zero_proportion': (df == 0).sum() / df.size,
        'skewness': stats.skew(df, axis=1),
        'kurtosis': stats.kurtosis(df, axis=1)
    }
    return metrics

def plot_quality_metrics(metrics: Dict) -> None:
    """Creates quality metric visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot distributions
    sns.boxplot(data=metrics['expression_distribution'], ax=axes[0,0])
    axes[0,0].set_title('Expression Distribution')
    
    # Plot sample correlations
    sns.heatmap(metrics['sample_correlation'], ax=axes[0,1])
    axes[0,1].set_title('Sample Correlation')
    
    # Plot CV distribution
    sns.histplot(metrics['cv_distribution'], ax=axes[1,0])
    axes[1,0].set_title('Coefficient of Variation')
    
    # Plot gene detection rates
    sns.barplot(data=metrics['gene_detection'], ax=axes[1,1])
    axes[1,1].set_title('Gene Detection Rates')
    
    plt.tight_layout()
    return fig

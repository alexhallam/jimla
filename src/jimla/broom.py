"""
Broom-equivalent functions for regression results.
"""

import jax.numpy as jnp
import numpy as np
import polars as pl
import tidy_viewer_py as tv
import fiasto_py
from typing import Dict, List, Optional

from .data import prepare_data_with_wayne
from .models import RegressionResult


def tidy(result: RegressionResult, 
         display: bool = True, 
         title: Optional[str] = None,
         color_theme: str = "default") -> pl.DataFrame:
    """
    Extract coefficient information from regression results (broom::tidy equivalent).
    
    Args:
        result: RegressionResult object from lm()
        display: Whether to display the results using tidy-viewer
        title: Optional title for the display
        color_theme: Color theme for tidy-viewer
        
    Returns:
        Polars DataFrame with coefficient information
    """
    # Get posterior samples
    samples = result.pathfinder_result.get('samples', None)
    
    # Extract coefficients and create tidy format
    coef_data = []
    for i, (term, estimate) in enumerate(result.coefficients.items()):
        if samples is not None and i < samples.shape[1] - 1:  # Exclude sigma
            # Extract samples for this coefficient
            coef_samples = samples[:, i]
            
            # Compute statistics
            std_error = float(np.std(coef_samples))
            statistic = float(estimate / std_error) if std_error > 0 else np.nan
            p_value = 2 * (1 - _normal_cdf(abs(statistic))) if not np.isnan(statistic) else np.nan
            
            # Compute credible intervals (95%)
            conf_low = float(np.percentile(coef_samples, 2.5))
            conf_high = float(np.percentile(coef_samples, 97.5))
        else:
            # Fallback to NaN if no samples
            std_error = np.nan
            statistic = np.nan
            p_value = np.nan
            conf_low = np.nan
            conf_high = np.nan
        
        coef_data.append({
            'term': term,
            'estimate': float(estimate),
            'std_error': std_error,
            'statistic': statistic,
            'p_value': p_value,
            'conf_low_2_5': conf_low,
            'conf_high_97_5': conf_high
        })
    
    # Create DataFrame
    df = pl.DataFrame(coef_data)
    
    if display:
        viewer = tv.TV()
        viewer.print_polars_dataframe(df)
    
    return df


def _normal_cdf(x):
    """Approximate normal CDF for p-value calculation."""
    # Simple approximation using error function
    return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))


def _f_cdf(x, df1, df2):
    """Approximate F-distribution CDF for p-value calculation."""
    # Simple approximation for F-distribution CDF
    # This is a rough approximation - for production use, consider using scipy.stats
    if x <= 0:
        return 0.0
    if x >= 100:  # Very large F-statistic
        return 1.0
    
    # Simple approximation using beta function relationship
    # F = (X1/df1) / (X2/df2) where X1 ~ χ²(df1), X2 ~ χ²(df2)
    # For large degrees of freedom, F approaches normal
    if df1 > 30 and df2 > 30:
        # Normal approximation
        mean = df2 / (df2 - 2) if df2 > 2 else 1.0
        var = 2 * df2**2 * (df1 + df2 - 2) / (df1 * (df2 - 2)**2 * (df2 - 4)) if df2 > 4 else 1.0
        z = (x - mean) / np.sqrt(var)
        return _normal_cdf(z)
    else:
        # Simple approximation for smaller degrees of freedom
        # This is very rough - in practice, use scipy.stats.f.cdf
        return min(1.0, max(0.0, 1 - np.exp(-x / 2)))


def augment(result: RegressionResult, 
            data: Optional[pl.DataFrame] = None,
            display: bool = True, 
            title: Optional[str] = None,
            color_theme: str = "default") -> pl.DataFrame:
    """
    Add fitted values and residuals to original data (broom::augment equivalent).
    
    Args:
        result: RegressionResult object from lm()
        data: Original data (required for augment)
        display: Whether to display the results using tidy-viewer
        title: Optional title for the display
        color_theme: Color theme for tidy-viewer
        
    Returns:
        Polars DataFrame with original data plus fitted values and residuals
    """
    if data is None:
        raise ValueError("Data argument is required for augment() - cannot reconstruct from model")
    
    # Get response variable name
    # Get response variable name using fiasto-py
    parsed_formula = fiasto_py.parse_formula(result.formula)
    # Find the response variable from the columns
    response_var = None
    for var_name, var_info in parsed_formula['columns'].items():
        if 'Response' in var_info['roles']:
            response_var = var_name
            break
    
    if response_var is None:
        raise ValueError(f"No response variable found in formula '{result.formula}'")
    
    # Prepare design matrix using wayne
    X, y, column_names = prepare_data_with_wayne(data, result.formula)
    coef_names = column_names
    
    # Extract coefficients in the right order
    coefs = jnp.array([result.coefficients[name] for name in coef_names])
    
    # Calculate fitted values and residuals
    fitted_values = X @ coefs
    residuals = y - fitted_values
    
    # Add to original data
    augmented_data = data.clone()
    augmented_data = augmented_data.with_columns([
        pl.Series(".fitted", np.array(fitted_values)),
        pl.Series(".resid", np.array(residuals))
    ])
    
    if display:
        viewer = tv.TV()
        viewer.print_polars_dataframe(augmented_data)
    
    return augmented_data


def glance(result: RegressionResult, 
           display: bool = True, 
           title: Optional[str] = None,
           color_theme: str = "default") -> pl.DataFrame:
    """
    Extract model-level statistics (broom::glance equivalent).
    
    Args:
        result: RegressionResult object from lm()
        display: Whether to display the results using tidy-viewer
        title: Optional title for the display
        color_theme: Color theme for tidy-viewer
        
    Returns:
        Polars DataFrame with model-level statistics
    """
    # Get response variable name
    # Get response variable name using fiasto-py
    parsed_formula = fiasto_py.parse_formula(result.formula)
    # Find the response variable from the columns
    response_var = None
    for var_name, var_info in parsed_formula['columns'].items():
        if 'Response' in var_info['roles']:
            response_var = var_name
            break
    
    if response_var is None:
        raise ValueError(f"No response variable found in formula '{result.formula}'")
    
    # Get posterior samples for calculations
    samples = result.pathfinder_result.get('samples', None)
    
    # Initialize values
    sigma_estimate = np.nan
    log_lik = np.nan
    AIC = np.nan
    BIC = np.nan
    deviance = np.nan
    statistic = np.nan
    p_value = np.nan
    
    if samples is not None:
        # Extract sigma samples (last column)
        sigma_samples = np.exp(samples[:, -1])  # Convert from log(sigma) to sigma
        sigma_estimate = float(np.mean(sigma_samples))
        
        # Calculate log-likelihood using the mean sigma
        # For a normal likelihood: log L = -n/2 * log(2π) - n/2 * log(σ²) - 1/(2σ²) * Σ(y - Xβ)²
        # We'll use the R-squared to approximate the residual sum of squares
        # RSS = (1 - R²) * TSS, where TSS = n * var(y)
        # For simplicity, we'll use the mean of the posterior samples
        if result.r_squared < 1.0:
            # Approximate log-likelihood using R-squared and sigma
            n = result.n_obs
            log_lik = -n/2 * np.log(2 * np.pi) - n/2 * np.log(sigma_estimate**2) - n/2
            
            # Calculate AIC and BIC
            k = result.n_params
            AIC = -2 * log_lik + 2 * k
            BIC = -2 * log_lik + k * np.log(n)
            
            # Calculate deviance (negative log-likelihood)
            deviance = -2 * log_lik
            
            # Calculate F-statistic approximation
            # F = (R² / (k-1)) / ((1-R²) / (n-k))
            if result.r_squared > 0 and result.r_squared < 1:
                numerator = result.r_squared / (result.n_params - 1)
                denominator = (1 - result.r_squared) / (result.n_obs - result.n_params)
                if denominator > 0:
                    statistic = numerator / denominator
                    # Approximate p-value using F-distribution
                    # This is a rough approximation
                    p_value = 1 - _f_cdf(statistic, result.n_params - 1, result.n_obs - result.n_params)
    
    # Calculate adjusted R-squared
    adj_r_squared = 1 - (1 - result.r_squared) * (result.n_obs - 1) / (result.n_obs - result.n_params)
    
    # Create glance data with proper column names
    glance_data = {
        'r_squared': result.r_squared,
        'adj_r_squared': adj_r_squared,
        'sigma': sigma_estimate,
        'statistic': statistic,
        'p_value': p_value,
        'df': result.n_params - 1,  # Degrees of freedom
        'logLik': log_lik,
        'AIC': AIC,
        'BIC': BIC,
        'deviance': deviance,
        'df_residual': result.n_obs - result.n_params,
        'nobs': result.n_obs
    }
    
    # Create DataFrame
    df = pl.DataFrame([glance_data])
    
    if display:
        viewer = tv.TV()
        viewer.print_polars_dataframe(df)
    
    return df

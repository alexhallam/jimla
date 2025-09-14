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
    # Extract coefficients and create tidy format
    coef_data = []
    for term, estimate in result.coefficients.items():
        coef_data.append({
            'term': term,
            'estimate': float(estimate),
            'std.error': np.nan,  # Would need posterior samples to compute
            'statistic': np.nan,  # Would need posterior samples to compute
            'p.value': np.nan,    # Would need posterior samples to compute
            'conf.low': np.nan,   # Would need posterior samples to compute
            'conf.high': np.nan   # Would need posterior samples to compute
        })
    
    # Create DataFrame
    df = pl.DataFrame(coef_data)
    
    if display:
        display_title = title or "JIMLA Coefficients (tidy)"
        viewer = tv.TV()
        viewer.print_polars_dataframe(df, title=display_title, color_theme=color_theme)
    
    return df


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
        display_title = title or "JIMLA Augmented Data"
        viewer = tv.TV()
        viewer.print_polars_dataframe(augmented_data, title=display_title, color_theme=color_theme)
    
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
    
    # For log-likelihood calculation, we need the original data
    # Since we don't have it here, we'll use a placeholder
    log_lik = np.nan  # Would need original data to calculate properly
    
    # Create glance data
    glance_data = {
        'r.squared': result.r_squared,
        'adj.r.squared': np.nan,  # Would need to calculate
        'sigma': np.nan,          # Would need to extract from result
        'statistic': np.nan,      # F-statistic
        'p.value': np.nan,        # p-value for F-test
        'df': result.n_params - 1,  # Degrees of freedom
        'logLik': log_lik,
        'AIC': np.nan,            # Would need log-likelihood
        'BIC': np.nan,            # Would need log-likelihood
        'deviance': np.nan,       # Would need to calculate
        'df.residual': result.n_obs - result.n_params,
        'nobs': result.n_obs
    }
    
    # Create DataFrame
    df = pl.DataFrame([glance_data])
    
    if display:
        display_title = title or "JIMLA Model Summary (glance)"
        viewer = tv.TV()
        viewer.print_polars_dataframe(df, title=display_title, color_theme=color_theme)
    
    return df

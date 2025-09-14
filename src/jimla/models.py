"""
Core regression model functionality.
"""

import jax
import jax.numpy as jnp
import polars as pl
import blackjax.vi.pathfinder as pathfinder_module
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

from .data import prepare_data_with_wayne
from .priors import compute_autoscales, log_prior_autoscaled
from .likelihood import log_likelihood


@dataclass
class RegressionResult:
    """Container for regression results."""
    coefficients: Dict[str, float]
    r_squared: float
    formula: str
    n_obs: int
    n_params: int
    pathfinder_result: Dict


def lm(df: pl.DataFrame, formula: str, **kwargs) -> RegressionResult:
    """
    Fit a Bayesian linear regression model using variational inference.
    
    This function fits a linear regression model using JAX and blackjax for
    variational inference. It supports Wilkinson's notation for formulas
    and returns results in a format similar to R's broom::tidy().
    
    Args:
        df: Polars DataFrame containing the data
        formula: Wilkinson's formula string (e.g., "y ~ x1 + x2")
        **kwargs: Additional arguments (currently unused, for future compatibility)
            - maxiter: Maximum iterations for pathfinder (default: 1000)
            - tol: Convergence tolerance (default: 1e-6)
        
    Returns:
        RegressionResult object containing coefficients and model information
    """
    # Prepare data using wayne-trade
    X, y, column_names = prepare_data_with_wayne(df, formula)
    n_obs, n_params = X.shape
    
    # Compute automatic prior scales (rstanarm/brms style)
    beta_scales, intercept_loc, intercept_scale, sigma_scale = compute_autoscales(X, y, column_names)
    
    # Set up the model with autoscaling
    def logdensity_fn(params_and_logsigma):
        params = params_and_logsigma[:-1]
        log_sigma = params_and_logsigma[-1]
        sigma = jnp.exp(log_sigma)  # Transform back to sigma
        
        # Log-likelihood
        log_lik = log_likelihood(params, X, y, sigma)
        
        # Log-prior with autoscaling
        log_prior = log_prior_autoscaled(params, log_sigma, beta_scales, 
                                        intercept_loc, intercept_scale, sigma_scale)
        
        return log_lik + log_prior
    
    # Better initialization: use OLS estimates as starting point
    try:
        # Compute OLS estimates for better initialization
        XtX_inv = jnp.linalg.inv(X.T @ X)
        ols_coefs = XtX_inv @ X.T @ y
        residuals = y - X @ ols_coefs
        ols_sigma = jnp.sqrt(jnp.mean(residuals**2))
        
        # Initialize with OLS estimates
        init_params = jnp.concatenate([ols_coefs, jnp.array([jnp.log(ols_sigma)])])
    except:
        # Fallback to zeros if OLS fails
        init_params = jnp.zeros(n_params + 1)
    
    # Set up random key
    rng_key = jax.random.PRNGKey(42)
    
    # Set up progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=None,
        transient=True
    ) as progress:
        
        # Fit the model using pathfinder
        task = progress.add_task("Fitting JIMLA", total=None)
        
        start_time = time.time()
        # Run pathfinder
        pathfinder_result, _ = pathfinder_module.approximate(
            rng_key,
            logdensity_fn,
            init_params,
            maxiter=kwargs.get('maxiter', 1000),
            gtol=kwargs.get('tol', 1e-6)
        )
        
        # Sample from posterior for uncertainty estimates
        progress.update(task, description="Sampling from posterior")
        samples_tuple = pathfinder_module.sample(
            rng_key,
            pathfinder_result,
            num_samples=kwargs.get('num_samples', 1000)
        )
        # Extract the actual samples (first element of tuple)
        samples = samples_tuple[0] if isinstance(samples_tuple, tuple) else samples_tuple
        
        progress.update(task, completed=100)
    
    # Extract results
    coefs = pathfinder_result.position[:-1]  # All parameters except log(sigma)
    sigma = jnp.exp(pathfinder_result.position[-1])  # Transform log(sigma) back to sigma
    
    # Create coefficient dictionary using wayne column names
    coefficients = dict(zip(column_names, coefs))
    
    # Calculate R-squared
    y_pred = X @ coefs
    ss_res = jnp.sum((y - y_pred)**2)
    ss_tot = jnp.sum((y - jnp.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot
    
    # Store pathfinder result and samples for potential future use
    pathfinder_dict = {
        'position': pathfinder_result.position,
        'elbo': pathfinder_result.elbo,
        'grad_position': pathfinder_result.grad_position,
        'samples': samples
    }
    
    # Create result object
    result = RegressionResult(
        coefficients=coefficients,
        r_squared=float(r_squared),
        formula=formula,
        n_obs=n_obs,
        n_params=n_params,
        pathfinder_result=pathfinder_dict
    )
    
    # Automatically print tidy output
    _print_tidy_output(result)
    
    return result


def _print_tidy_output(result: RegressionResult):
    """Print tidy output automatically after fitting."""
    import polars as pl
    import numpy as np
    import tidy_viewer_py as tv
    
    # Get posterior samples
    samples = result.pathfinder_result.get('samples', None)
    
    # Create tidy format data
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
    
    # Display using tidy-viewer
    viewer = tv.TV()
    viewer.print_polars_dataframe(df)


def _normal_cdf(x):
    """Approximate normal CDF for p-value calculation."""
    import numpy as np
    # Simple approximation using error function
    return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

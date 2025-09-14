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
    
    # Store pathfinder result for potential future use
    pathfinder_dict = {
        'position': pathfinder_result.position,
        'elbo': pathfinder_result.elbo,
        'grad_position': pathfinder_result.grad_position
    }
    
    return RegressionResult(
        coefficients=coefficients,
        r_squared=float(r_squared),
        formula=formula,
        n_obs=n_obs,
        n_params=n_params,
        pathfinder_result=pathfinder_dict
    )

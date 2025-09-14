"""
Likelihood functions for regression models.
"""

import jax.numpy as jnp


def log_likelihood(params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, sigma: float) -> float:
    """
    Log-likelihood for linear regression.
    
    Args:
        params: Regression coefficients
        X: Design matrix
        y: Response vector
        sigma: Standard deviation of residuals
        
    Returns:
        Log-likelihood value
    """
    n = len(y)
    y_pred = X @ params
    log_lik = -0.5 * n * jnp.log(2 * jnp.pi * sigma**2) - 0.5 * jnp.sum((y - y_pred)**2) / sigma**2
    return log_lik


def log_posterior(params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, 
                  sigma: float, prior_scale: float = 10.0) -> float:
    """
    Log posterior for linear regression.
    
    Args:
        params: Regression coefficients
        X: Design matrix
        y: Response vector
        sigma: Standard deviation of residuals
        prior_scale: Scale parameter for coefficient priors
        
    Returns:
        Log posterior value
    """
    from .priors import log_prior
    
    log_lik = log_likelihood(params, X, y, sigma)
    log_prior_val = log_prior(params, sigma, prior_scale)
    return log_lik + log_prior_val

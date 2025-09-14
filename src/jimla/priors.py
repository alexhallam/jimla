"""
Prior distributions and autoscaling functionality.
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple


def compute_autoscales(X: jnp.ndarray, y: jnp.ndarray, colnames: list) -> Tuple[jnp.ndarray, float, float, float]:
    """
    Compute automatic prior scales based on data spread (rstanarm/brms style).
    
    Args:
        X: Design matrix
        y: Response vector
        colnames: Column names from wayne-trade
        
    Returns:
        Tuple of (beta_scales, intercept_loc, intercept_scale, sigma_scale)
    """
    y_np = np.asarray(y)
    # Robust spread of response (using MAD as fallback)
    sy = max(float(y_np.std(ddof=1)), 
             float(np.median(np.abs(y_np - y_np.mean())) * 1.4826), 
             1e-8)
    
    beta_scales = []
    for j, name in enumerate(colnames):
        xj = np.asarray(X[:, j])
        if name.lower() in ("(intercept)", "intercept", "Intercept"):
            beta_scales.append(np.nan)  # placeholder for intercept
            continue
        
        sx = float(xj.std(ddof=1))
        # Treat binary-like predictors as 0.5 (Gelman's recommendation)
        if np.isfinite(sx) and sx < 1e-12 and set(np.unique(xj)).issubset({0.0, 1.0}):
            sx = 0.5
        sx = max(sx, 1e-12)
        
        # rstanarm-style autoscale: 2.5 * sd(y) / sd(x)
        beta_scales.append(2.5 * sy / sx)
    
    intercept_loc = float(y_np.mean())
    intercept_scale = 2.5 * sy
    sigma_scale = sy
    
    return jnp.array(beta_scales), intercept_loc, intercept_scale, sigma_scale


def log_prior_autoscaled(params: jnp.ndarray, log_sigma: float, 
                        beta_scales: jnp.ndarray, intercept_loc: float, 
                        intercept_scale: float, sigma_scale: float) -> float:
    """
    Log prior for regression parameters with autoscaling.
    
    Args:
        params: Regression coefficients
        log_sigma: Log of standard deviation of residuals
        beta_scales: Per-coefficient prior scales (first element is nan for intercept)
        intercept_loc: Intercept prior location
        intercept_scale: Intercept prior scale
        sigma_scale: Sigma prior scale
        
    Returns:
        Log prior value
    """
    # Split intercept vs other coefficients
    alpha = params[0]
    betas = params[1:]
    
    # Normal prior on intercept (centered at mean of y)
    lp_alpha = -0.5 * ((alpha - intercept_loc) / intercept_scale) ** 2
    
    # Normal priors on standardized coefficients (skip the nan for intercept)
    valid_beta_scales = beta_scales[1:]  # Skip the nan for intercept
    lp_betas = -0.5 * jnp.sum((betas / valid_beta_scales) ** 2)
    
    # Log-normal prior on sigma (equivalent to normal on log(sigma))
    lp_logsigma = -0.5 * ((log_sigma - jnp.log(sigma_scale)) / 1.0) ** 2
    
    return lp_alpha + lp_betas + lp_logsigma


def log_prior(params: jnp.ndarray, sigma: float, prior_scale: float = 10.0) -> float:
    """
    Log prior for regression coefficients and residual standard deviation.
    
    Args:
        params: Regression coefficients
        sigma: Standard deviation of residuals
        prior_scale: Scale of the normal prior (larger = weaker prior)
        
    Returns:
        Log prior value
    """
    # Weak normal prior for coefficients (closer to maximum likelihood)
    log_prior_coefs = -0.5 * jnp.sum(params**2) / (prior_scale**2)  # N(0, prior_scale^2) prior
    # Weak inverse gamma prior for sigma
    log_prior_sigma = -2 * jnp.log(sigma)  # Inverse gamma(1, 1) prior
    return log_prior_coefs + log_prior_sigma

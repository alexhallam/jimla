"""
Core regression functionality using fiasto-py and blackjax.
"""

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import fiasto_py
import blackjax
import tidy_viewer_py as tv
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
import time


@dataclass
class RegressionResult:
    """Container for regression results."""
    coefficients: Dict[str, float]
    r_squared: float
    formula: str
    n_obs: int
    n_params: int
    pathfinder_result: Dict


def _parse_formula(formula: str) -> Dict:
    """
    Parse formula using fiasto-py and extract relevant information.
    
    Args:
        formula: Wilkinson's formula string (e.g., "y ~ x1 + x2")
        
    Returns:
        Parsed formula metadata
    """
    try:
        result = fiasto_py.parse_formula(formula)
        return result
    except Exception as e:
        raise ValueError(f"Failed to parse formula '{formula}': {e}")


def _extract_variables(parsed_formula: Dict) -> Tuple[List[str], List[str]]:
    """
    Extract response and predictor variables from parsed formula.
    
    Args:
        parsed_formula: Result from fiasto_py.parse_formula()
        
    Returns:
        Tuple of (response_vars, predictor_vars)
    """
    response_vars = []
    predictor_vars = []
    
    for col, details in parsed_formula["columns"].items():
        if "Response" in details["roles"]:
            response_vars.append(col)
        elif "FixedEffect" in details["roles"]:
            predictor_vars.append(col)
    
    return response_vars, predictor_vars


def _prepare_data(df: pl.DataFrame, response_vars: List[str], 
                 predictor_vars: List[str], has_intercept: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Prepare data matrices for regression.
    
    Args:
        df: Polars DataFrame
        response_vars: List of response variable names
        predictor_vars: List of predictor variable names
        has_intercept: Whether to include intercept
        
    Returns:
        Tuple of (X, y) as JAX arrays
    """
    # Check that all variables exist in DataFrame
    all_vars = response_vars + predictor_vars
    missing_vars = [var for var in all_vars if var not in df.columns]
    if missing_vars:
        raise ValueError(f"Variables not found in DataFrame: {missing_vars}")
    
    # Extract data
    y = df.select(response_vars).to_numpy()
    X = df.select(predictor_vars).to_numpy()
    
    # Handle multiple response variables (flatten for now)
    if y.ndim > 1:
        y = y.ravel()
    
    # Add intercept if needed
    if has_intercept:
        X = np.column_stack([np.ones(X.shape[0]), X])
    
    return jnp.array(X), jnp.array(y)


def _log_likelihood(params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, 
                   sigma: float) -> float:
    """
    Log likelihood for linear regression.
    
    Args:
        params: Regression coefficients
        X: Design matrix
        y: Response vector
        sigma: Standard deviation of residuals
        
    Returns:
        Log likelihood value
    """
    n = X.shape[0]
    y_pred = X @ params
    log_lik = -0.5 * n * jnp.log(2 * jnp.pi * sigma**2) - 0.5 * jnp.sum((y - y_pred)**2) / sigma**2
    return log_lik


def _log_prior(params: jnp.ndarray, sigma: float, prior_scale: float = 10.0) -> float:
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


def _log_posterior(params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, 
                  sigma: float, prior_scale: float = 10.0) -> float:
    """
    Log posterior for linear regression.
    
    Args:
        params: Regression coefficients
        X: Design matrix
        y: Response vector
        sigma: Standard deviation of residuals
        prior_scale: Scale of the normal prior (larger = weaker prior)
        
    Returns:
        Log posterior value
    """
    return _log_likelihood(params, X, y, sigma) + _log_prior(params, sigma, prior_scale)


def lm(df: pl.DataFrame, formula: str, **kwargs) -> RegressionResult:
    """
    Fit a Bayesian linear regression model using blackjax pathfinder.
    
    Args:
        df: Polars DataFrame containing the data
        formula: Wilkinson's formula string (e.g., "y ~ x1 + x2")
        **kwargs: Additional arguments for fine-tuning:
            - num_samples: Number of posterior samples (default: 2000)
            - pathfinder_samples: Number of pathfinder samples (default: 100)
            - maxiter: Maximum iterations for optimization (default: 1000)
            - tol: Convergence tolerance (default: 1e-6)
        
    Returns:
        RegressionResult object containing coefficients and model information
    """
    # Parse formula
    parsed_formula = _parse_formula(formula)
    response_vars, predictor_vars = _extract_variables(parsed_formula)
    has_intercept = parsed_formula["metadata"]["has_intercept"]
    
    if len(response_vars) != 1:
        raise ValueError("Only single response variable is currently supported")
    
    if len(predictor_vars) == 0 and not has_intercept:
        raise ValueError("Model must have at least one predictor or intercept")
    
    # Prepare data
    X, y = _prepare_data(df, response_vars, predictor_vars, has_intercept)
    n_obs, n_params = X.shape
    
    # Use a weak prior scale internally (not exposed to users)
    prior_scale = 10.0
    
    # Set up the model
    def logdensity_fn(params_and_sigma):
        params = params_and_sigma[:-1]
        sigma = jnp.exp(params_and_sigma[-1])  # Log transform for positivity
        return _log_posterior(params, X, y, sigma, prior_scale)
    
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
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=False  # Keep progress visible
    ) as progress:
        
        # Task for pathfinder approximation
        pathfinder_task = progress.add_task(
            "Fitting JIMLA", 
            total=100
        )
        
        # Simulate progress for pathfinder (since it's very fast)
        for i in range(0, 50, 10):
            progress.update(pathfinder_task, completed=i)
            time.sleep(0.1)
        
        # Run pathfinder with optimized parameters
        pathfinder_kwargs = {
            'num_samples': kwargs.get('pathfinder_samples', 100),
            'maxiter': kwargs.get('maxiter', 1000),
            'tol': kwargs.get('tol', 1e-6),  # Use 'tol' instead of 'atol'/'rtol'
        }
        # Remove our custom kwargs to avoid conflicts
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['num_samples', 'pathfinder_samples', 'maxiter', 'tol']}
        pathfinder_kwargs.update(filtered_kwargs)
        
        pathfinder_state, _ = blackjax.vi.pathfinder.approximate(
            rng_key, logdensity_fn, init_params, **pathfinder_kwargs
        )
        
        # Complete pathfinder task
        progress.update(pathfinder_task, completed=100)
        
        # Task for sampling
        sampling_task = progress.add_task(
            "Sampling from posterior.", 
            total=100
        )
        
        # Simulate progress for sampling
        for i in range(0, 100, 20):
            progress.update(sampling_task, completed=i)
            time.sleep(0.05)
        
        # Sample from the posterior (increase default samples for better accuracy)
        num_samples = kwargs.get('num_samples', 2000)
        samples, _ = blackjax.vi.pathfinder.sample(
            rng_key, pathfinder_state, num_samples
        )
        
        # Complete the sampling task
        progress.update(sampling_task, completed=100)
        
        # Brief pause to show completion
        time.sleep(0.2)
    
    # Extract results
    mean_params = jnp.mean(samples, axis=0)
    
    # Transform back from log space
    coefs = mean_params[:-1]
    sigma = jnp.exp(mean_params[-1])
    
    # Create coefficient dictionary
    coef_names = []
    if has_intercept:
        coef_names.append("(Intercept)")
    coef_names.extend(predictor_vars)
    
    coefficients = dict(zip(coef_names, coefs))
    
    # Calculate R-squared
    y_pred = X @ coefs
    ss_res = jnp.sum((y - y_pred)**2)
    ss_tot = jnp.sum((y - jnp.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return RegressionResult(
        coefficients=coefficients,
        r_squared=float(r_squared),
        formula=formula,
        n_obs=n_obs,
        n_params=n_params,
        pathfinder_result={"samples": samples, "state": pathfinder_state}
    )


def tidy(result: RegressionResult, 
         display: bool = True,
         title: Optional[str] = None,
         color_theme: str = "default") -> pl.DataFrame:
    """
    Create a tidy summary of regression results, similar to broom::tidy().
    Uses tidy-viewer for enhanced display by default.
    
    Args:
        result: RegressionResult from lm()
        display: Whether to display the results using tidy-viewer
        title: Optional title for the display
        color_theme: Color theme for display ("default", "dracula", etc.)
        
    Returns:
        Polars DataFrame with term, estimate, and other statistics
    """
    # Extract samples from pathfinder result
    samples = result.pathfinder_result["samples"]
    
    # Get coefficient samples (exclude sigma)
    coef_samples = samples[:, :-1]  # All but last column (log_sigma)
    
    # Calculate statistics
    estimates = jnp.mean(coef_samples, axis=0)
    std_errors = jnp.std(coef_samples, axis=0)
    
    # Calculate credible intervals (2.5% and 97.5%)
    lower_ci = jnp.percentile(coef_samples, 2.5, axis=0)
    upper_ci = jnp.percentile(coef_samples, 97.5, axis=0)
    
    # Calculate t-statistics (estimate / std_error)
    t_statistics = estimates / std_errors
    
    # For Bayesian inference, we can calculate "Bayesian p-values" 
    # as the probability that the coefficient is greater than 0
    p_values = jnp.mean(coef_samples > 0, axis=0)
    # For negative coefficients, p-value is probability of being less than 0
    p_values = jnp.where(estimates < 0, jnp.mean(coef_samples < 0, axis=0), p_values)
    # Two-sided p-value
    p_values = 2 * jnp.minimum(p_values, 1 - p_values)
    
    # Create coefficient names
    terms = list(result.coefficients.keys())
    
    # Create the DataFrame
    tidy_df = pl.DataFrame({
        "term": terms,
        "estimate": [float(x) for x in estimates],
        "std_error": [float(x) for x in std_errors],
        "statistic": [float(x) for x in t_statistics],
        "p_value": [float(x) for x in p_values],
        "p_025": [float(x) for x in lower_ci],
        "p_975": [float(x) for x in upper_ci],
    })
    
    # Display using tidy-viewer if requested
    if display:
        # Create title if not provided
        if title is None:
            title = f"Regression Results: {result.formula}"
        
        # Configure tidy-viewer
        viewer = tv.tv()
        
        if title:
            viewer = viewer.title(title)
        
        if color_theme != "default":
            viewer = viewer.color_theme(color_theme)
        
        # Display the results
        viewer.print_polars_dataframe(tidy_df)
        
        # Print summary information
        print(f"\nModel Summary:")
        print(f"  Formula: {result.formula}")
        print(f"  R-squared: {result.r_squared:.4f}")
        print(f"  Observations: {result.n_obs}")
        print(f"  Parameters: {result.n_params}")
    
    return tidy_df


def augment(result: RegressionResult, 
            data: Optional[pl.DataFrame] = None,
            display: bool = True,
            title: Optional[str] = None,
            color_theme: str = "default") -> pl.DataFrame:
    """
    Add columns to the original data with model information (Bayesian equivalent of broom::augment).
    
    Args:
        result: RegressionResult from lm()
        data: Original DataFrame (if None, will attempt to reconstruct from model)
        display: Whether to display the result using tidy-viewer
        title: Optional title for display
        color_theme: Color theme for tidy-viewer
        
    Returns:
        DataFrame with original data plus model columns
    """
    # Extract samples for uncertainty quantification
    samples = result.pathfinder_result["samples"]
    coef_samples = samples[:, :-1]  # All but last column (sigma)
    
    # Parse formula to get variable information
    parsed_formula = _parse_formula(result.formula)
    response_vars, predictor_vars = _extract_variables(parsed_formula)
    has_intercept = parsed_formula["metadata"]["has_intercept"]
    
    if data is None:
        raise ValueError("Data argument is required for augment() - cannot reconstruct from model")
    
    # Prepare design matrix
    X, y = _prepare_data(data, response_vars, predictor_vars, has_intercept)
    
    # Get coefficient names
    coef_names = []
    if has_intercept:
        coef_names.append("(Intercept)")
    coef_names.extend(predictor_vars)
    
    # Calculate fitted values for each sample
    fitted_samples = X @ coef_samples.T  # Shape: (n_obs, n_samples)
    
    # Calculate point estimates (mean of posterior samples)
    fitted_values = jnp.mean(fitted_samples, axis=1)
    residuals = y.ravel() - fitted_values
    
    # Calculate uncertainty measures
    fitted_std = jnp.std(fitted_samples, axis=1)
    fitted_low = jnp.percentile(fitted_samples, 2.5, axis=1)
    fitted_high = jnp.percentile(fitted_samples, 97.5, axis=1)
    
    # Calculate leverage (hat values) - simplified version
    try:
        XtX_inv = jnp.linalg.inv(X.T @ X)
        hat_values = jnp.diag(X @ XtX_inv @ X.T)
    except:
        # Fallback if matrix is singular
        hat_values = jnp.zeros(X.shape[0])
    
    # Calculate standardized residuals
    sigma_samples = jnp.exp(samples[:, -1])  # Transform back from log space
    sigma_mean = jnp.mean(sigma_samples)
    std_residuals = residuals / sigma_mean
    
    # Create augmented DataFrame
    augmented_data = data.clone()
    
    # Add model columns (following broom conventions with . prefix)
    # Convert JAX arrays to numpy arrays for Polars compatibility
    augmented_data = augmented_data.with_columns([
        pl.Series(".fitted", np.array(fitted_values)),
        pl.Series(".resid", np.array(residuals)),
        pl.Series(".fitted_std", np.array(fitted_std)),
        pl.Series(".fitted_low", np.array(fitted_low)),
        pl.Series(".fitted_high", np.array(fitted_high)),
        pl.Series(".hat", np.array(hat_values)),
        pl.Series(".std.resid", np.array(std_residuals)),
        pl.Series(".sigma", np.full(len(fitted_values), float(sigma_mean)))
    ])
    
    # Display using tidy-viewer if requested
    if display:
        if title is None:
            title = "Augmented Data with Model Information"
        
        # Configure tidy-viewer
        viewer = tv.tv()
        
        if title:
            viewer = viewer.title(title)
        
        if color_theme != "default":
            viewer = viewer.color_theme(color_theme)
        
        print(f"\n{title}")
        print("=" * len(title))
        viewer.print_polars_dataframe(augmented_data)
        
        # Print summary
        print(f"\nModel Summary:")
        print(f"  Formula: {result.formula}")
        print(f"  R-squared: {result.r_squared:.4f}")
        print(f"  Observations: {result.n_obs}")
        print(f"  Parameters: {result.n_params}")
        print(f"  Added columns: .fitted, .resid, .fitted_std, .fitted_low, .fitted_high, .hat, .std.resid, .sigma")
    
    return augmented_data


def glance(result: RegressionResult,
           display: bool = True,
           title: Optional[str] = None,
           color_theme: str = "default") -> pl.DataFrame:
    """
    Return a one-row summary of the model (Bayesian equivalent of broom::glance).
    
    Args:
        result: RegressionResult from lm()
        display: Whether to display the result using tidy-viewer
        title: Optional title for display
        color_theme: Color theme for tidy-viewer
        
    Returns:
        One-row DataFrame with model summary statistics
    """
    # Extract samples for additional statistics
    samples = result.pathfinder_result["samples"]
    coef_samples = samples[:, :-1]
    sigma_samples = jnp.exp(samples[:, -1])
    
    # Calculate additional Bayesian statistics
    sigma_mean = jnp.mean(sigma_samples)
    sigma_std = jnp.std(sigma_samples)
    
    # Calculate effective sample size (simplified)
    n_eff = len(samples)  # In practice, would calculate ESS from samples
    
    # Calculate log-likelihood at posterior mean
    parsed_formula = _parse_formula(result.formula)
    response_vars, predictor_vars = _extract_variables(parsed_formula)
    has_intercept = parsed_formula["metadata"]["has_intercept"]
    
    # For log-likelihood calculation, we need the original data
    # Since we don't have it here, we'll use a placeholder
    log_lik = np.nan  # Would need original data to calculate properly
    
    # Create one-row summary
    glance_data = {
        "r_squared": result.r_squared,
        "adj_r_squared": 1 - (1 - result.r_squared) * (result.n_obs - 1) / (result.n_obs - result.n_params),
        "sigma": float(sigma_mean),
        "sigma_std": float(sigma_std),
        "n_obs": result.n_obs,
        "n_params": result.n_params,
        "df_residual": result.n_obs - result.n_params,
        "n_samples": len(samples),
        "n_eff": n_eff,
        "formula": result.formula,
        "method": "Bayesian (blackjax pathfinder)"
    }
    
    glance_df = pl.DataFrame([glance_data])
    
    # Display using tidy-viewer if requested
    if display:
        if title is None:
            title = "Model Summary (Glance)"
        
        # Configure tidy-viewer
        viewer = tv.tv()
        
        if title:
            viewer = viewer.title(title)
        
        if color_theme != "default":
            viewer = viewer.color_theme(color_theme)
        
        print(f"\n{title}")
        print("=" * len(title))
        viewer.print_polars_dataframe(glance_df)
        
        print(f"\nBayesian Model Information:")
        print(f"  Method: Variational inference with blackjax pathfinder")
        print(f"  Posterior samples: {len(samples)}")
        print(f"  R-squared: {result.r_squared:.4f}")
        print(f"  Residual std dev: {sigma_mean:.4f} ± {sigma_std:.4f}")
    
    return glance_df



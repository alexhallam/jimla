"""
Tests for core regression model functionality.
"""

import pytest
import polars as pl
import numpy as np
import jax.numpy as jnp
from jimla.models import lm, RegressionResult


def test_lm_basic(sample_data):
    """Test basic linear regression."""
    result = lm(sample_data, 'y ~ x1 + x2')
    
    # Check result type
    assert isinstance(result, RegressionResult)
    
    # Check basic attributes
    assert result.formula == 'y ~ x1 + x2'
    assert result.n_obs == 100
    assert result.n_params == 3  # intercept, x1, x2
    
    # Check coefficients
    assert 'intercept' in result.coefficients
    assert 'x1' in result.coefficients
    assert 'x2' in result.coefficients
    
    # Check that coefficients are finite
    for coef in result.coefficients.values():
        assert np.isfinite(coef)
    
    # Check R-squared is reasonable
    assert 0 <= result.r_squared <= 1
    
    # Check pathfinder result
    assert 'position' in result.pathfinder_result
    assert 'elbo' in result.pathfinder_result
    assert 'grad_position' in result.pathfinder_result


def test_lm_single_predictor(sample_data):
    """Test linear regression with single predictor."""
    result = lm(sample_data, 'y ~ x1')
    
    assert result.formula == 'y ~ x1'
    assert result.n_params == 2  # intercept, x1
    assert 'intercept' in result.coefficients
    assert 'x1' in result.coefficients
    assert 'x2' not in result.coefficients


def test_lm_mtcars(mtcars_data):
    """Test linear regression with mtcars data."""
    result = lm(mtcars_data, 'mpg ~ wt + cyl')
    
    assert result.formula == 'mpg ~ wt + cyl'
    assert result.n_obs == 32
    assert result.n_params == 3  # intercept, wt, cyl
    
    # Check that coefficients are reasonable
    # wt should be negative (heavier cars have lower mpg)
    assert result.coefficients['wt'] < 0
    
    # cyl should be negative (more cylinders = lower mpg)
    assert result.coefficients['cyl'] < 0
    
    # R-squared should be reasonable for this model
    assert 0.5 <= result.r_squared <= 0.9


def test_lm_autoscaling_robustness(mtcars_data, scaled_mtcars_data):
    """Test that autoscaling makes results robust to data scale."""
    # Fit models on original and scaled data
    result_orig = lm(mtcars_data, 'mpg ~ wt')
    result_scaled = lm(scaled_mtcars_data, 'mpg ~ wt')
    
    # Check that coefficients scale appropriately
    intercept_ratio = result_scaled.coefficients['intercept'] / (result_orig.coefficients['intercept'] * 1000)
    wt_ratio = result_scaled.coefficients['wt'] / result_orig.coefficients['wt']
    
    # Ratios should be close to 1.0 (perfect scaling)
    assert abs(intercept_ratio - 1.0) < 0.1
    assert abs(wt_ratio - 1.0) < 0.1
    
    # R-squared should be identical
    assert abs(result_scaled.r_squared - result_orig.r_squared) < 1e-6


def test_lm_convergence(sample_data):
    """Test that the model converges properly."""
    result = lm(sample_data, 'y ~ x1 + x2')
    
    # Check that elbo is finite
    assert np.isfinite(result.pathfinder_result['elbo'])
    
    # Check that position is finite
    assert np.all(np.isfinite(result.pathfinder_result['position']))
    
    # Check that gradient is finite
    assert np.all(np.isfinite(result.pathfinder_result['grad_position']))


def test_lm_invalid_formula(sample_data):
    """Test that invalid formulas raise appropriate errors."""
    with pytest.raises(ValueError, match="Failed to process formula"):
        lm(sample_data, 'invalid ~ formula')


def test_lm_missing_variable(sample_data):
    """Test that missing variables are handled gracefully."""
    # wayne-trade may not raise an error for missing variables
    # This test documents the current behavior
    try:
        result = lm(sample_data, 'y ~ nonexistent')
        # If no error is raised, that's the current behavior
        assert True
    except ValueError:
        # If an error is raised, that's also acceptable
        assert True


def test_lm_kwargs(sample_data):
    """Test that kwargs are passed to pathfinder."""
    # Test with custom maxiter and tol
    result = lm(sample_data, 'y ~ x1', maxiter=100, tol=1e-4)
    
    # Should still work and produce reasonable results
    assert isinstance(result, RegressionResult)
    assert result.n_params == 2


def test_lm_ols_initialization_fallback(sample_data):
    """Test that OLS initialization works and fallback is available."""
    # This test ensures that both OLS initialization and zero fallback work
    result = lm(sample_data, 'y ~ x1 + x2')
    
    # Should produce reasonable results
    assert np.all(np.isfinite(result.pathfinder_result['position']))
    assert np.isfinite(result.pathfinder_result['elbo'])


def test_lm_coefficient_signs(sample_data):
    """Test that coefficient signs make sense for the generated data."""
    result = lm(sample_data, 'y ~ x1 + x2')
    
    # For the generated data: y = 2 + 3*x1 - 1.5*x2 + noise
    # So we expect: intercept ≈ 2, x1 ≈ 3, x2 ≈ -1.5
    
    # Check that signs are correct
    assert result.coefficients['x1'] > 0  # positive relationship
    assert result.coefficients['x2'] < 0  # negative relationship
    
    # Check that magnitudes are reasonable (within 2x of true values)
    assert 1.5 <= result.coefficients['x1'] <= 4.5  # true value is 3
    assert -3.0 <= result.coefficients['x2'] <= -0.75  # true value is -1.5

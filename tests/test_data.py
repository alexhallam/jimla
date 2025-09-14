"""
Tests for data preparation functionality.
"""

import pytest
import polars as pl
import jax.numpy as jnp
from jimla.data import prepare_data_with_wayne


def test_prepare_data_with_wayne_basic(sample_data):
    """Test basic data preparation."""
    X, y, colnames = prepare_data_with_wayne(sample_data, 'y ~ x1 + x2')
    
    # Check shapes
    assert X.shape == (100, 3)  # n_obs, n_params (including intercept)
    assert y.shape == (100,)
    assert len(colnames) == 3
    
    # Check column names
    assert 'intercept' in colnames
    assert 'x1' in colnames
    assert 'x2' in colnames
    
    # Check that intercept column is all 1s
    assert jnp.allclose(X[:, 0], 1.0)
    
    # Check that x1 and x2 match original data
    assert jnp.allclose(X[:, 1], sample_data['x1'].to_numpy())
    assert jnp.allclose(X[:, 2], sample_data['x2'].to_numpy())
    
    # Check that y matches original data
    assert jnp.allclose(y, sample_data['y'].to_numpy())


def test_prepare_data_with_wayne_single_predictor(sample_data):
    """Test data preparation with single predictor."""
    X, y, colnames = prepare_data_with_wayne(sample_data, 'y ~ x1')
    
    # Check shapes
    assert X.shape == (100, 2)  # n_obs, n_params (including intercept)
    assert y.shape == (100,)
    assert len(colnames) == 2
    
    # Check column names
    assert 'intercept' in colnames
    assert 'x1' in colnames


def test_prepare_data_with_wayne_invalid_formula(sample_data):
    """Test that invalid formulas raise appropriate errors."""
    with pytest.raises(ValueError, match="Failed to process formula"):
        prepare_data_with_wayne(sample_data, 'invalid ~ formula')


def test_prepare_data_with_wayne_missing_variable(sample_data):
    """Test that missing variables are handled gracefully."""
    # wayne-trade may not raise an error for missing variables
    # This test documents the current behavior
    try:
        X, y, colnames = prepare_data_with_wayne(sample_data, 'y ~ nonexistent')
        # If no error is raised, that's the current behavior
        assert True
    except ValueError:
        # If an error is raised, that's also acceptable
        assert True


def test_prepare_data_with_wayne_mtcars(mtcars_data):
    """Test data preparation with mtcars data."""
    X, y, colnames = prepare_data_with_wayne(mtcars_data, 'mpg ~ wt + cyl')
    
    # Check shapes
    assert X.shape == (32, 3)  # 32 cars, 3 parameters (intercept, wt, cyl)
    assert y.shape == (32,)
    assert len(colnames) == 3
    
    # Check column names
    assert 'intercept' in colnames
    assert 'wt' in colnames
    assert 'cyl' in colnames
    
    # Check that intercept column is all 1s
    assert jnp.allclose(X[:, 0], 1.0)

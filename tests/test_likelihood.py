"""
Tests for likelihood functions.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from jimla.likelihood import log_likelihood, log_posterior


def test_log_likelihood_basic():
    """Test basic log likelihood computation."""
    # Create simple test data
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])  # intercept, x
    y = jnp.array([1.0, 2.0, 3.0])
    params = jnp.array([1.0, 1.0])  # intercept=1, slope=1
    sigma = 0.5
    
    log_lik = log_likelihood(params, X, y, sigma)
    
    # Should return a finite value
    assert jnp.isfinite(log_lik)
    
    # Should be negative (log of probability)
    assert log_lik < 0


def test_log_likelihood_perfect_fit():
    """Test log likelihood with perfect fit."""
    # Create data that fits perfectly
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = jnp.array([1.0, 2.0, 3.0])  # y = 1 + 1*x
    params = jnp.array([1.0, 1.0])  # Perfect fit
    sigma = 0.1  # Small noise
    
    log_lik = log_likelihood(params, X, y, sigma)
    
    # Should be finite and relatively high (less negative)
    assert jnp.isfinite(log_lik)
    assert log_lik > -10  # Should be reasonably high


def test_log_likelihood_poor_fit():
    """Test log likelihood with poor fit."""
    # Create data that fits poorly
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = jnp.array([1.0, 2.0, 3.0])
    params = jnp.array([0.0, 0.0])  # Poor fit (predicts all zeros)
    sigma = 0.1  # Small noise
    
    log_lik = log_likelihood(params, X, y, sigma)
    
    # Should be finite but very negative (low probability)
    assert jnp.isfinite(log_lik)
    assert log_lik < -50  # Should be very low


def test_log_likelihood_sigma_effect():
    """Test that larger sigma gives lower log likelihood."""
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = jnp.array([1.0, 2.0, 3.0])
    params = jnp.array([1.0, 1.0])
    
    log_lik_small_sigma = log_likelihood(params, X, y, sigma=0.1)
    log_lik_large_sigma = log_likelihood(params, X, y, sigma=1.0)
    
    # Smaller sigma should give higher log likelihood
    assert log_lik_small_sigma > log_lik_large_sigma


def test_log_posterior_basic():
    """Test basic log posterior computation."""
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = jnp.array([1.0, 2.0, 3.0])
    params = jnp.array([1.0, 1.0])
    sigma = 0.5
    prior_scale = 10.0
    
    log_post = log_posterior(params, X, y, sigma, prior_scale)
    
    # Should return a finite value
    assert jnp.isfinite(log_post)
    
    # Log posterior can be positive if likelihood is high enough
    # Just check that it's finite and reasonable
    assert -1000 < log_post < 1000


def test_log_posterior_prior_scale_effect():
    """Test that prior scale affects log posterior."""
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = jnp.array([1.0, 2.0, 3.0])
    params = jnp.array([1.0, 1.0])
    sigma = 0.5
    
    log_post_weak_prior = log_posterior(params, X, y, sigma, prior_scale=100.0)
    log_post_strong_prior = log_posterior(params, X, y, sigma, prior_scale=1.0)
    
    # Weak prior should give higher log posterior (closer to likelihood)
    assert log_post_weak_prior > log_post_strong_prior


def test_log_likelihood_vectorized():
    """Test that log likelihood works with vectorized inputs."""
    # Test with multiple parameter vectors
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = jnp.array([1.0, 2.0, 3.0])
    sigma = 0.5
    
    # Test with different parameter sets
    params1 = jnp.array([1.0, 1.0])
    params2 = jnp.array([0.0, 2.0])
    
    log_lik1 = log_likelihood(params1, X, y, sigma)
    log_lik2 = log_likelihood(params2, X, y, sigma)
    
    # Both should be finite
    assert jnp.isfinite(log_lik1)
    assert jnp.isfinite(log_lik2)
    
    # They should be different (unless by coincidence)
    # This test might occasionally fail due to numerical precision
    # but it's unlikely for these specific values
    assert not jnp.allclose(log_lik1, log_lik2, atol=1e-10)

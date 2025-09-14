"""
Pytest configuration and fixtures for JIMLA tests.
"""

import pytest
import polars as pl
import numpy as np
import jax.numpy as jnp


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    y = 2 + 3*x1 - 1.5*x2 + np.random.normal(0, 0.5, n)
    
    return pl.DataFrame({
        'y': y,
        'x1': x1,
        'x2': x2
    })


@pytest.fixture
def mtcars_data():
    """Load mtcars data for testing."""
    mtcars_path = 'https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv'
    return pl.read_csv(mtcars_path)


@pytest.fixture
def scaled_mtcars_data():
    """Load scaled mtcars data for testing autoscaling."""
    mtcars_path = 'https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv'
    df = pl.read_csv(mtcars_path)
    return df.with_columns([
        (pl.col('mpg') * 1000).alias('mpg'),
        (pl.col('wt') * 1000).alias('wt')
    ])

"""
Tests for broom-equivalent functions.
"""

import pytest
import polars as pl
import numpy as np
from jimla.models import lm
from jimla.broom import tidy, augment, glance


def test_tidy_basic(sample_data):
    """Test basic tidy functionality."""
    result = lm(sample_data, 'y ~ x1 + x2')
    df = tidy(result, display=False)
    
    # Check DataFrame structure
    assert isinstance(df, pl.DataFrame)
    assert df.shape[0] == 3  # intercept, x1, x2
    assert df.shape[1] == 7  # term, estimate, std.error, statistic, p.value, conf.low, conf.high
    
    # Check column names
    expected_cols = ['term', 'estimate', 'std.error', 'statistic', 'p.value', 'conf.low', 'conf.high']
    assert df.columns == expected_cols
    
    # Check that terms are correct
    terms = df['term'].to_list()
    assert 'intercept' in terms
    assert 'x1' in terms
    assert 'x2' in terms
    
    # Check that estimates are finite
    estimates = df['estimate'].to_list()
    for est in estimates:
        assert np.isfinite(est)
    
    # Check that other columns are NaN (not implemented yet)
    for col in ['std.error', 'statistic', 'p.value', 'conf.low', 'conf.high']:
        assert df[col].is_nan().all()


def test_tidy_single_predictor(sample_data):
    """Test tidy with single predictor."""
    result = lm(sample_data, 'y ~ x1')
    df = tidy(result, display=False)
    
    assert df.shape[0] == 2  # intercept, x1
    terms = df['term'].to_list()
    assert 'intercept' in terms
    assert 'x1' in terms
    assert 'x2' not in terms


def test_tidy_mtcars(mtcars_data):
    """Test tidy with mtcars data."""
    result = lm(mtcars_data, 'mpg ~ wt + cyl')
    df = tidy(result, display=False)
    
    assert df.shape[0] == 3  # intercept, wt, cyl
    terms = df['term'].to_list()
    assert 'intercept' in terms
    assert 'wt' in terms
    assert 'cyl' in terms
    
    # Check that estimates make sense
    estimates = df.filter(pl.col('term') == 'wt')['estimate'].item()
    assert estimates < 0  # wt should be negative
    
    estimates = df.filter(pl.col('term') == 'cyl')['estimate'].item()
    assert estimates < 0  # cyl should be negative


def test_augment_basic(sample_data):
    """Test basic augment functionality."""
    result = lm(sample_data, 'y ~ x1 + x2')
    df = augment(result, data=sample_data, display=False)
    
    # Check DataFrame structure
    assert isinstance(df, pl.DataFrame)
    assert df.shape[0] == 100  # same as original data
    assert df.shape[1] == 5  # original columns + .fitted + .resid
    
    # Check that original columns are preserved
    assert 'y' in df.columns
    assert 'x1' in df.columns
    assert 'x2' in df.columns
    
    # Check that new columns are added
    assert '.fitted' in df.columns
    assert '.resid' in df.columns
    
    # Check that fitted values and residuals are finite
    assert df['.fitted'].is_finite().all()
    assert df['.resid'].is_finite().all()


def test_augment_missing_data():
    """Test that augment raises error when data is missing."""
    # Create a dummy result (we won't actually fit it)
    class DummyResult:
        def __init__(self):
            self.formula = 'y ~ x1'
    
    result = DummyResult()
    
    with pytest.raises(ValueError, match="Data argument is required"):
        augment(result, data=None, display=False)


def test_augment_mtcars(mtcars_data):
    """Test augment with mtcars data."""
    result = lm(mtcars_data, 'mpg ~ wt + cyl')
    df = augment(result, data=mtcars_data, display=False)
    
    assert df.shape[0] == 32  # same as mtcars
    assert '.fitted' in df.columns
    assert '.resid' in df.columns
    
    # Check that fitted values are reasonable
    fitted = df['.fitted'].to_numpy()
    assert np.all(fitted > 0)  # mpg should be positive
    assert np.all(fitted < 50)  # reasonable upper bound
    
    # Check that residuals have reasonable magnitude
    residuals = df['.resid'].to_numpy()
    assert np.std(residuals) < 10  # residuals shouldn't be too large


def test_glance_basic(sample_data):
    """Test basic glance functionality."""
    result = lm(sample_data, 'y ~ x1 + x2')
    df = glance(result, display=False)
    
    # Check DataFrame structure
    assert isinstance(df, pl.DataFrame)
    assert df.shape[0] == 1  # single row
    assert df.shape[1] == 12  # all the glance columns
    
    # Check column names
    expected_cols = ['r.squared', 'adj.r.squared', 'sigma', 'statistic', 'p.value', 
                    'df', 'logLik', 'AIC', 'BIC', 'deviance', 'df.residual', 'nobs']
    assert df.columns == expected_cols
    
    # Check that r.squared is computed correctly
    r_squared = df['r.squared'].item()
    assert 0 <= r_squared <= 1
    assert abs(r_squared - result.r_squared) < 1e-10  # should match result
    
    # Check that nobs is correct
    assert df['nobs'].item() == result.n_obs
    
    # Check that df.residual is correct
    assert df['df.residual'].item() == result.n_obs - result.n_params


def test_glance_mtcars(mtcars_data):
    """Test glance with mtcars data."""
    result = lm(mtcars_data, 'mpg ~ wt + cyl')
    df = glance(result, display=False)
    
    # Check that r.squared is reasonable for this model
    r_squared = df['r.squared'].item()
    assert 0.5 <= r_squared <= 0.9
    
    # Check that nobs is correct
    assert df['nobs'].item() == 32
    
    # Check that df is correct (n_params - 1)
    assert df['df'].item() == 2  # 3 params - 1


def test_broom_functions_consistency(sample_data):
    """Test that broom functions are consistent with each other."""
    result = lm(sample_data, 'y ~ x1 + x2')
    
    # Get results from all three functions
    tidy_df = tidy(result, display=False)
    augment_df = augment(result, data=sample_data, display=False)
    glance_df = glance(result, display=False)
    
    # Check that r.squared is consistent
    tidy_r2 = None  # tidy doesn't have r.squared
    glance_r2 = glance_df['r.squared'].item()
    
    assert abs(glance_r2 - result.r_squared) < 1e-10
    
    # Check that nobs is consistent
    glance_nobs = glance_df['nobs'].item()
    assert glance_nobs == result.n_obs
    
    # Check that coefficients are consistent between tidy and result
    for row in tidy_df.iter_rows(named=True):
        term = row['term']
        estimate = row['estimate']
        assert abs(estimate - result.coefficients[term]) < 1e-10


def test_broom_functions_display_off(sample_data):
    """Test that display=False works for all broom functions."""
    result = lm(sample_data, 'y ~ x1 + x2')
    
    # These should not raise errors and should return DataFrames
    tidy_df = tidy(result, display=False)
    augment_df = augment(result, data=sample_data, display=False)
    glance_df = glance(result, display=False)
    
    assert isinstance(tidy_df, pl.DataFrame)
    assert isinstance(augment_df, pl.DataFrame)
    assert isinstance(glance_df, pl.DataFrame)

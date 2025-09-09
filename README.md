# jimla

A Python package for Bayesian linear regression using variational inference, inspired by R's `lm()` function and the `broom` package.

## Features

- **Formula parsing**: Uses [fiasto-py](https://github.com/alexhallam/fiasto-py) to parse Wilkinson's notation formulas
- **Bayesian inference**: Uses [blackjax](https://github.com/blackjax-devs/blackjax) pathfinder for variational inference
- **Complete Broom API**: Full equivalents of `tidy()`, `augment()`, and `glance()` functions
- **Enhanced display**: Results displayed using tidy-viewer for beautiful, formatted output
- **Progress tracking**: Rich progress bars show variational inference progress
- **Polars integration**: Works seamlessly with Polars DataFrames
- **Uncertainty quantification**: Bayesian credible intervals and uncertainty measures

## Installation

```bash
pip install jimla
```

## Quick Start

```python
import polars as pl
import numpy as np
from jimla import lm, tidy, augment, glance

# Create sample data
np.random.seed(42)
n = 100
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
y = 2 + 1.5 * x1 + 0.8 * x2 + np.random.normal(0, 0.5, n)

df = pl.DataFrame({
    "y": y,
    "x1": x1,
    "x2": x2
})

# Fit regression model
result = lm(df, "y ~ x1 + x2")

# Get tidy output (coefficients and statistics)
tidy_result = tidy(result)

# Augment original data with model information
augmented_data = augment(result, df)

# Get one-row model summary
model_summary = glance(result)
```

## API Reference

### `lm(df: pl.DataFrame, formula: str, **kwargs) -> RegressionResult`

Fit a Bayesian linear regression model using blackjax pathfinder.

**Parameters:**
- `df`: Polars DataFrame containing the data
- `formula`: Wilkinson's formula string (e.g., "y ~ x1 + x2")
- `**kwargs`: Additional arguments passed to blackjax pathfinder

**Returns:**
- `RegressionResult`: Object containing coefficients, R-squared, and model information

### `tidy(result: RegressionResult, display: bool = True, title: str = None, color_theme: str = "default") -> pl.DataFrame`

Create a tidy summary of regression results, similar to `broom::tidy()`.
Uses tidy-viewer for enhanced display by default.

**Parameters:**
- `result`: RegressionResult from `lm()`
- `display`: Whether to display the results using tidy-viewer (default: True)
- `title`: Optional title for the display
- `color_theme`: Color theme for display ("default", "dracula", etc.)

**Returns:**
- `pl.DataFrame`: DataFrame with columns: term, estimate, std.error, statistic, p.value, conf.low, conf.high

### `augment(result: RegressionResult, data: pl.DataFrame, display: bool = True, title: str = None, color_theme: str = "default") -> pl.DataFrame`

Add model information to the original data, similar to `broom::augment()`.

**Parameters:**
- `result`: RegressionResult from `lm()`
- `data`: Original Polars DataFrame
- `display`: Whether to display results using tidy-viewer (default: True)
- `title`: Optional title for the display
- `color_theme`: Color theme for display ("default", "dracula", etc.)

**Returns:**
- `pl.DataFrame`: Original data plus model columns: .fitted, .resid, .fitted_std, .fitted_low, .fitted_high, .hat, .std.resid, .sigma

### `glance(result: RegressionResult, display: bool = True, title: str = None, color_theme: str = "default") -> pl.DataFrame`

Create a one-row model summary, similar to `broom::glance()`.

**Parameters:**
- `result`: RegressionResult from `lm()`
- `display`: Whether to display results using tidy-viewer (default: True)
- `title`: Optional title for the display
- `color_theme`: Color theme for display ("default", "dracula", etc.)

**Returns:**
- `pl.DataFrame`: One-row DataFrame with: r_squared, adj_r_squared, sigma, sigma_std, n_obs, n_params, df_residual, n_samples, n_eff, formula, method

## Supported Formula Syntax

jimla supports Wilkinson's notation through fiasto-py:

- **Basic formulas**: `y ~ x1 + x2`
- **Interactions**: `y ~ x1 * x2`
- **Intercept control**: `y ~ x1 + x2 - 1` (no intercept)

## Example Output

```
Formula: y ~ x1 + x2
R-squared: 0.8951
Number of observations: 100
Number of parameters: 3

Coefficients:
  (Intercept): 2.0370
  x1: 1.6048
  x2: 0.7877

Tidy output (with tidy-viewer):
```
Regression Results: y ~ x1 + x2

        tv dim: 3 x 7
        term        estimate std.error statistic p.value conf.low conf.high 
        <str>       <f64>    <f64>     <f64>     <f64>   <f64>    <f64>     
     1  (Intercept) 2.04     0.0463    43.9      0       1.94     2.12      
     2  x1          1.60     0.0592    27.1      0       1.49     1.72      
     3  x2          0.788    0.0654    12.0      0       0.662    0.912     

Model Summary:
  Formula: y ~ x1 + x2
  R-squared: 0.8951
  Observations: 100
  Parameters: 3
```
```

## Dependencies

- [blackjax](https://github.com/blackjax-devs/blackjax) - Bayesian inference
- [fiasto-py](https://github.com/alexhallam/fiasto-py) - Formula parsing
- [polars](https://github.com/pola-rs/polars) - Data manipulation
- [jax](https://github.com/google/jax) - Numerical computing
- [tidy-viewer-py](https://github.com/alexhallam/tv/tree/main/tidy-viewer-py) - Enhanced data display
- [rich](https://github.com/Textualize/rich) - Progress bars and terminal formatting

## License

MIT License

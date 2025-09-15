<h1 align="center">jimla</h1>

<p align="center">
  <img src="https://openmoji.org/data/color/svg/1F47E.svg" alt="logo" width="240">
</p>

---


A Python package for Linear regression using "pathfinder variational inference" using polars dataframes.

## Supported Formula Syntax

jimla supports Wilkinson's notation through fiasto-py and wayne-trade:

- **Basic formulas**: `y ~ x1 + x2`
- **Interactions**: `y ~ x1 * x2`
- **Polynomials**: `y ~ poly(x1, 2)`
- **Intercept control**: `y ~ x1 + x2 - 1` (no intercept)
- **Complex interactions**: `y ~ x1 + x2*x3 + poly(x1, 2)`

## Features

- **Formula parsing**: Uses [fiasto-py](https://github.com/alexhallam/fiasto-py) for formula parsing and [wayne-trade](https://github.com/alexhallam/wayne) for model matrix generation
- **Bayesian inference**: Uses [blackjax](https://github.com/blackjax-devs/blackjax) pathfinder for variational inference
- **Automatic prior scaling**: Stan/brms-style autoscaling makes models robust to data scales
- **Enhanced display**: Results displayed using tidy-viewer for beautiful, formatted output
- **Progress tracking**: Rich progress bars show variational inference progress
- **Polars integration**: Works seamlessly with Polars DataFrames
- **Scale invariant**: Works with any data scale (dollars, inches, milliseconds) without manual tuning via autoscaling

## Installation

```bash
pip install jimla
# I like:
# uv pip install jimla
```

## Quick Start

```python
import polars as pl
import numpy as np
from jimla import lm, augment, glance

mtcars_path = 'https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv'
df = pl.read_csv(mtcars_path)

# lm() automatically prints tidy output
result = lm(df, "mpg ~ cyl + wt*hp - 1")

# Show augmented data
augment(result, df)

# Show model summary
glance(result)
```

![output](img/output.png)




Fit a Linear regression model using [blackjax](https://blackjax-devs.github.io/blackjax/)[pathfinder](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/vi/pathfinder/index.html).


## Motivation - A Second Best At Everything

This is not the fastest way to do regression and it is not the most accurate way to do regression.

The fastest options will be frequentist. I use this all the time.

I made this package to get something that would give some uncertainty quantification (UQ), but I did not want to wait for MCMC to converge.

## How does this fit in with other regression options?

I am not going to oversell this regression package. I am going to give you the lay of the land. I have a table below that probably generalizes more than experts would, but I think it is a good starting point.

| Axis                      | Pathfinder VI                                                                                                  | R `lm()` (OLS)                                                                 | NUTS (dynamic HMC)                                                                     | Laplace approximation                                            |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| What it computes          | Approx. **posterior draws** via quasi-Newton path variational fit; selects best normal (and can **multi-start + importance-resample**). | **Point estimates** and asymptotic SEs/CI under OLS assumptions; no posterior. | **Asymptotically exact posterior** samples via MCMC trajectories.                      | **Single Gaussian** at MAP from local curvature (Hessian).       |
| Uncertainty               | **Bayesian**, from samples (can be close to short-run HMC; > ADVI on many tests).                                                       | **Frequentist** SEs; relies on homoskedasticity/CLT.                           | **Bayesian**, gold standard if mixed well.                                             | **Bayesian-ish**, sometimes gets a bad rap for being **too narrow** / symmetric. My opinion is that it is a good approximation for many problems.          |
| Geometry & multi-modality | Better than Laplace (fits **along a path**; multiple starts help); still VI—can **miss minor modes** without restarts.                  | N/A (optimization).                                                            | Handles **strong skew/correlations** and multi-modal posteriors (with enough compute). | **Local & unimodal** by design; struggles with skew/heavy tails. |
| Speed & scale             | **1–2 orders of magnitude fewer grads** than ADVI/short HMC in paper benchmarks; embarrassingly parallel. Great for big N, moderate D.  | **Fastest** for linear models; closed forms.                                   | **Slowest** per effective sample—gradients per step + tuning; excellent but costly.    | **Very fast**; just one mode’s Hessian.                          |
| Output for users          | **Posterior draws, ELBO/KL diagnostics**, WAIC/LOO feasible; can act as **initializer for MCMC**.                                       | Coefs, SEs, t-tests, R²; classic diagnostics.                                  | Posterior draws + rich diagnostics (ESS, R-hat, divergences).                          | Gaussian approx (mean/cov), log-evidence approx.                 |
| Where it shines           | GLMs & hierarchical regression when you want **near-Bayesian UQ** at **interactive speeds**; GPU-friendly with JAX.                     | Simple linear regression with OLS assumptions; quick inference/explanation.    | Final-mile inference where **calibration is critical** (tails, decisions).             | Quick marginal-likelihoods, well-behaved unimodal posteriors.    |
| Where it struggles        | **Extreme** multi-modality/heavy-tails; may **underestimate variance** (common to VI).                                                  | Misspecified errors, small-N inference, complex priors.                        | Compute/time; tricky posteriors need careful diagnostics.                              | Anything non-Gaussian/strongly skew or multi-modal.              |



## Motivation - A Second Best At Everything


Bayesian-style uncertainty quantification (UQ) that is faster than MCMC, but slower than Laplace Approximation. From this we get posterior draws (credible intervals, predictions) faster than NUTS and with less local bias than Laplace approximation, enabling Bayesian regression for practitioners who can’t afford full MCMC on every model.

## But We Already Have Stan/Brms/PyMC/Bambi/rstanarm/etc.

1. A focused package is an easy to maintain package.

I had an experience with one Bayesian linear regression package which offered all MCMC options as well as variational inference and laplace. I was clear that the package really focused on one flavour or Bayesian ingerence. When I attempted to fit a simple model with Laplace approcimation the package crashed. A sensible person would just open an issue. I am not sensible. I am stubborn.

2. Many Bayesian packages have C++ dependancies/toolchains which take some care to install.

3. I was just curious. How far can we take VI via pathfinder before it breaks?

My hunch is that this this is a good approximation for my practical regression needs. I am not going to claim that it is a good approximation for all problems.

4. I like pretty things.

I wanted pretty output. I wanted a nice progress bar. I wanted a nice way to display data.

5. I like to use Polars.

I wanted a Polars first regression package. I can see myself integrating Narwhals in the future. At while I am developing I am using Polars.

6. JAX is cool.

I wanted to use JAX. I wanted to use pathfinder. I wanted to use blackjax. It is a minimal way to do Bayesian inference.

7. I am testing some fomula parsing

This sits on top of fiasto-py and wayne-trade. I have a hypothesis that decoupling parsing from the model matrix generation will be a good thing. I am probably wrong, but I am going to try it anyway. Yes, I am aware of formulaic and formulae. I have fiasto-py sitting on top of `fiasto`, a rust package that offers lexing and parsing of Wilkinson's formulas.


**Parameters:**
- `df`: Polars DataFrame containing the data
- `formula`: Wilkinson's formula string (e.g., "y ~ x1 + x2")
- `**kwargs`: Additional arguments passed to blackjax pathfinder

**Returns:**
- `RegressionResult`: Object containing coefficients, R-squared, and model information

### `tidy(result: RegressionResult, display: bool = True, title: str = None, color_theme: str = "default") -> pl.DataFrame`

Create a tidy summary of regression results, similar to `broom::tidy()`.
Note: Tidy output is automatically printed when calling `lm()`, but you can call this function manually if needed.

**Parameters:**
- `result`: RegressionResult from `lm()`
- `display`: Whether to display the results using tidy-viewer (default: True)
- `title`: Optional title for the display
- `color_theme`: Color theme for display ("default", "dracula", etc.)

**Returns:**
- `pl.DataFrame`: DataFrame with columns: term, estimate, std_error, statistic, p_value, conf_low_2_5, conf_high_97_5

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
- `pl.DataFrame`: One-row DataFrame with: r_squared, adj_r_squared, sigma, statistic, p_value, df, logLik, AIC, BIC, deviance, df_residual, nobs


```

## Dependencies

- [blackjax](https://github.com/blackjax-devs/blackjax) - Bayesian inference
- [fiasto-py](https://github.com/alexhallam/fiasto-py) - Formula parsing
- [wayne-trade](https://github.com/alexhallam/wayne) - Model matrix generation
- [polars](https://github.com/pola-rs/polars) - Data manipulation
- [jax](https://github.com/google/jax) - Numerical computing
- [tidy-viewer-py](https://github.com/alexhallam/tv/tree/main/tidy-viewer-py) - Enhanced data display
- [rich](https://github.com/Textualize/rich) - Progress bars and terminal formatting

## License

MIT License

## What is jimla?

I am glade you asked. You will be disappointed in the answer.

Many packages of cool acronyms or recursive names. Clever people make clever names. I am not clever.

I just finished reading [11/22/63](https://en.wikipedia.org/wiki/11/22/63). In the book, “jimla” is a mispronunciation of the main character’s name, Jim, by the “yellow card man,” a drunk guardian of time. The word then takes on a life of its own, becoming a nightmarish, disembodied form that grows angrier as Jim continues to change the obdurate past.

One could say that this name connects with the purpose of the package. MCMC is the proper order of things a pure Monte Carlo Chain. In contrast, the jimla is observed as renegade time streams ricochet across a timeline. By side stepping MCMC will we awaken the jimla?

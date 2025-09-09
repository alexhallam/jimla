"""
Example usage of jimla for Bayesian linear regression.
"""

import polars as pl
import numpy as np
from jimla import lm, tidy

# Create sample data
np.random.seed(42)
n = 100000

# Generate data with known relationship
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
y = 2 + 1.5 * x1 + 0.8 * x2 + np.random.normal(0, 0.5, n)

# Create polars DataFrame
df = pl.DataFrame({
    "y": y,
    "x1": x1,
    "x2": x2
})

print("Sample data:")
print(df.head())

# Fit regression model
print("\n" + "="*50)
print("Fitting Bayesian linear regression model")
print("="*50)

formula = "y ~ x1 + x2"
result = lm(df, formula)

print(f"Formula: {result.formula}")
print(f"R-squared: {result.r_squared:.4f}")
print(f"Number of observations: {result.n_obs}")
print(f"Number of parameters: {result.n_params}")

# Display coefficients
print("\nCoefficients:")
for term, coef in result.coefficients.items():
    print(f"  {term}: {coef:.4f}")

# Create tidy output (now uses tidy-viewer by default)
print("\n" + "="*50)
print("Tidy output with tidy-viewer")
print("="*50)
tidy_result = tidy(result)

# Example with single predictor
print("\n" + "="*50)
print("Single predictor model")
print("="*50)

single_result = lm(df, "y ~ x1")
print(f"Formula: {single_result.formula}")
print(f"R-squared: {single_result.r_squared:.4f}")

# Display with different color theme
print("\n" + "="*50)
print("Single predictor with dracula theme")
print("="*50)
tidy_single = tidy(single_result, title="Single Predictor Model")

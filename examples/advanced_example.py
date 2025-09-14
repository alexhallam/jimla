"""
Advanced example comparing jimla with traditional OLS regression.
"""

import polars as pl
import numpy as np
from jimla import lm, tidy

def compare_with_ols():
    """Compare jimla results with traditional OLS."""
    
    # Create more realistic data
    np.random.seed(123)
    n = 200
    
    # Generate correlated predictors
    x1 = np.random.normal(0, 1, n)
    x2 = 0.5 * x1 + np.random.normal(0, 0.8, n)  # Correlated with x1
    x3 = np.random.normal(0, 1, n)
    
    # True relationship
    y = 1.5 + 2.0 * x1 + 1.2 * x2 - 0.8 * x3 + np.random.normal(0, 0.5, n)
    
    df = pl.DataFrame({
        "y": y,
        "x1": x1,
        "x2": x2,
        "x3": x3
    })
    
    print("Dataset summary:")
    print(df.describe())
    
    # Fit with jimla
    print("\n" + "="*60)
    print("jimla (Bayesian with blackjax pathfinder)")
    print("="*60)
    
    result = lm(df, "y ~ x1 + x2 + x3")
    print(f"Formula: {result.formula}")
    print(f"R-squared: {result.r_squared:.4f}")
    print(f"Observations: {result.n_obs}")
    print(f"Parameters: {result.n_params}")
    
    # Show tidy output
    tidy_result = tidy(result)
    print("\nTidy output:")
    print(tidy_result)
    
    # Compare with traditional OLS (using numpy)
    print("\n" + "="*60)
    print("Traditional OLS (for comparison)")
    print("="*60)
    
    # Manual OLS calculation
    X = df.select(["x1", "x2", "x3"]).to_numpy()
    y_vals = df.select("y").to_numpy().ravel()
    
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    
    # Solve normal equations
    XTX = X_with_intercept.T @ X_with_intercept
    XTy = X_with_intercept.T @ y_vals
    ols_coefs = np.linalg.solve(XTX, XTy)
    
    # Calculate R-squared
    y_pred_ols = X_with_intercept @ ols_coefs
    ss_res = np.sum((y_vals - y_pred_ols)**2)
    ss_tot = np.sum((y_vals - np.mean(y_vals))**2)
    r_squared_ols = 1 - (ss_res / ss_tot)
    
    print(f"R-squared: {r_squared_ols:.4f}")
    
    # Create comparison DataFrame
    ols_df = pl.DataFrame({
        "term": ["(Intercept)", "x1", "x2", "x3"],
        "estimate": ols_coefs,
        "method": ["OLS"] * 4
    })
    
    jimla_df = tidy_result.select(["term", "estimate"]).with_columns(
        pl.lit("jimla").alias("method")
    )
    
    comparison = pl.concat([jimla_df, ols_df])
    print("\nComparison of estimates:")
    print(comparison.pivot(index="term", columns="method", values="estimate"))
    
    # Show differences
    jimla_coefs = dict(zip(tidy_result["term"], tidy_result["estimate"]))
    ols_coefs_dict = dict(zip(["x1", "x2", "x3"], ols_coefs[1:]))
    ols_coefs_dict["(Intercept)"] = ols_coefs[0]
    
    print("\nDifferences (jimla - OLS):")
    for term in ["(Intercept)", "x1", "x2", "x3"]:
        diff = jimla_coefs[term] - ols_coefs_dict[term]
        print(f"  {term}: {diff:.6f}")

if __name__ == "__main__":
    compare_with_ols()

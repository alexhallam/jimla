#!/usr/bin/env python3
"""
Basic usage example for JIMLA - Bayesian linear regression with variational inference.
"""

import polars as pl
import numpy as np
from jimla import lm, tidy, augment, glance

def main():
    print("🚀 JIMLA Basic Usage Example")
    print("=" * 50)
    
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
    
    print("📊 Sample Data:")
    print(df.head())
    print()
    
    # Fit regression model
    print("🔬 Fitting Bayesian Linear Regression...")
    result = lm(df, "y ~ x1 + x2")
    print()
    
    # Display results
    print("📈 Model Results:")
    print(f"Formula: {result.formula}")
    print(f"R-squared: {result.r_squared:.4f}")
    print(f"Observations: {result.n_obs}")
    print(f"Parameters: {result.n_params}")
    print()
    
    print("🎯 Coefficients:")
    for term, coef in result.coefficients.items():
        print(f"  {term}: {coef:.4f}")
    print()
    
    # Tidy output is automatically printed by lm()
    print("📋 Tidy Output (coefficients and statistics):")
    print("(Already displayed above by lm())")
    print()
    
    # Augment original data
    print("🔧 Augmented Data (original + fitted values and residuals):")
    augmented_data = augment(result, df, display=False)
    print(augmented_data.head())
    print()
    
    # Get model summary
    print("📊 Model Summary:")
    model_summary = glance(result, display=False)
    print(model_summary)
    print()
    
    print("✅ Example completed successfully!")

if __name__ == "__main__":
    main()

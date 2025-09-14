#!/usr/bin/env python3
"""
Showcase JIMLA with wayne-trade integration for complex formula parsing.

This example demonstrates the power of wayne-trade for handling complex
statistical formulas including interactions, polynomials, and more.
"""

import numpy as np
import polars as pl
from jimla import lm, tidy, augment, glance

def main():
    print("🤠 JIMLA + Wayne-Trade Formula Showcase 🤠")
    print("=" * 50)
    
    # Create realistic sample data
    np.random.seed(42)
    n = 300
    
    # Generate correlated variables (avoiding reserved names like 'c')
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)
    x4 = np.random.normal(0, 1, n)
    
    # Create true relationships with interactions and non-linearities
    y = (2.0 +                    # intercept
         1.5 * x1 +              # linear effect
         0.8 * x2 +              # linear effect
         0.3 * x3 +              # linear effect
         0.5 * x1 * x2 +         # interaction
         -0.2 * x1**2 +          # quadratic effect
         0.1 * x3 * x4 +         # another interaction
         np.random.normal(0, 0.3, n))  # noise
    
    df = pl.DataFrame({
        'y': y,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4
    })
    
    print(f"📊 Dataset: {n} observations, 4 predictors")
    print(f"📈 True model includes interactions and quadratic terms")
    print()
    
    # Example 1: Basic linear model
    print("1️⃣ Basic Linear Model")
    print("-" * 30)
    result1 = lm(df, 'y ~ x1 + x2 + x3 + x4')
    tidy(result1, title='Basic Linear Model')
    print()
    
    # Example 2: Model with interactions
    print("2️⃣ Model with Interactions")
    print("-" * 30)
    result2 = lm(df, 'y ~ x1*x2 + x3*x4')
    tidy(result2, title='Model with Interactions')
    print()
    
    # Example 3: Model with polynomials
    print("3️⃣ Model with Polynomials")
    print("-" * 30)
    result3 = lm(df, 'y ~ poly(x1, 2) + x2 + x3')
    tidy(result3, title='Model with Polynomials')
    print()
    
    # Example 4: Complex model with interactions and polynomials
    print("4️⃣ Complex Model (Interactions + Polynomials)")
    print("-" * 30)
    result4 = lm(df, 'y ~ x1*x2 + poly(x1, 2) + x3*x4')
    tidy(result4, title='Complex Model')
    print()
    
    # Example 5: Model without intercept
    print("5️⃣ Model without Intercept")
    print("-" * 30)
    result5 = lm(df, 'y ~ x1 + x2 - 1')
    tidy(result5, title='Model without Intercept')
    print()
    
    # Compare model performance
    print("📊 Model Comparison")
    print("-" * 30)
    models = [
        ("Basic Linear", result1),
        ("With Interactions", result2),
        ("With Polynomials", result3),
        ("Complex Model", result4)
    ]
    
    for name, result in models:
        print(f"{name:20} | R² = {result.r_squared:.4f} | n_params = {result.n_params}")

if __name__ == "__main__":
    main()

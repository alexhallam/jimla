#!/usr/bin/env python3
"""
Demonstration of JIMLA's automatic prior scaling feature.
Shows how the package is robust to different data scales.
"""

import polars as pl
import numpy as np
from jimla import lm, tidy

def main():
    print("🎯 JIMLA Autoscaling Demonstration")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    y = 2 + 1.5 * x1 + 0.8 * x2 + np.random.normal(0, 0.5, n)
    
    # Original data
    df_orig = pl.DataFrame({
        "y": y,
        "x1": x1,
        "x2": x2
    })
    
    # Scaled data (1000x larger)
    df_scaled = pl.DataFrame({
        "y": y * 1000,
        "x1": x1 * 1000,
        "x2": x2 * 1000
    })
    
    print("📊 Original Data (small scale):")
    print(f"  y range: {df_orig['y'].min():.2f} to {df_orig['y'].max():.2f}")
    print(f"  x1 range: {df_orig['x1'].min():.2f} to {df_orig['x1'].max():.2f}")
    print()
    
    print("📊 Scaled Data (1000x larger):")
    print(f"  y range: {df_scaled['y'].min():.0f} to {df_scaled['y'].max():.0f}")
    print(f"  x1 range: {df_scaled['x1'].min():.0f} to {df_scaled['x1'].max():.0f}")
    print()
    
    # Fit models on both datasets
    print("🔬 Fitting models on both datasets...")
    result_orig = lm(df_orig, "y ~ x1 + x2")
    result_scaled = lm(df_scaled, "y ~ x1 + x2")
    print()
    
    # Compare results
    print("📈 Model Comparison:")
    print("Original Data:")
    print(f"  Intercept: {result_orig.coefficients['intercept']:.4f}")
    print(f"  x1: {result_orig.coefficients['x1']:.4f}")
    print(f"  x2: {result_orig.coefficients['x2']:.4f}")
    print(f"  R²: {result_orig.r_squared:.4f}")
    print()
    
    print("Scaled Data (1000x):")
    print(f"  Intercept: {result_scaled.coefficients['intercept']:.0f}")
    print(f"  x1: {result_scaled.coefficients['x1']:.4f}")
    print(f"  x2: {result_scaled.coefficients['x2']:.4f}")
    print(f"  R²: {result_scaled.r_squared:.4f}")
    print()
    
    # Check scaling relationships
    intercept_ratio = result_scaled.coefficients['intercept'] / (result_orig.coefficients['intercept'] * 1000)
    x1_ratio = result_scaled.coefficients['x1'] / result_orig.coefficients['x1']
    x2_ratio = result_scaled.coefficients['x2'] / result_orig.coefficients['x2']
    
    print("🔍 Scaling Analysis:")
    print(f"  Intercept ratio (should be ≈ 1.0): {intercept_ratio:.3f}")
    print(f"  x1 ratio (should be ≈ 1.0): {x1_ratio:.3f}")
    print(f"  x2 ratio (should be ≈ 1.0): {x2_ratio:.3f}")
    print(f"  R² difference: {abs(result_scaled.r_squared - result_orig.r_squared):.6f}")
    print()
    
    if (abs(intercept_ratio - 1.0) < 0.1 and 
        abs(x1_ratio - 1.0) < 0.1 and 
        abs(x2_ratio - 1.0) < 0.1):
        print("✅ SUCCESS: Autoscaling works perfectly!")
        print("   JIMLA is robust to data scales like Stan/brms!")
    else:
        print("❌ ISSUE: Autoscaling needs adjustment")
    
    print()
    print("🎯 Key Benefits of Autoscaling:")
    print("  • No manual prior tuning required")
    print("  • Works with any data scale (dollars, inches, milliseconds)")
    print("  • Consistent results across different units")
    print("  • Matches behavior of mature Bayesian packages")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Demonstration of how prior scale affects Bayesian regression with different data scales.
"""

import polars as pl
import numpy as np
import jax.numpy as jnp
from jimla import lm, tidy

def demonstrate_prior_scale():
    print("🔬 PRIOR SCALE DEMONSTRATION")
    print("=" * 50)
    
    # Load mtcars data
    mtcars_path = "https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv"
    df = pl.read_csv(mtcars_path)
    
    # Create scaled versions
    df_original = df
    df_scaled_10x = df.with_columns([
        (pl.col('mpg') * 10).alias('mpg'),
        (pl.col('wt') * 10).alias('wt')
    ])
    df_scaled_100x = df.with_columns([
        (pl.col('mpg') * 100).alias('mpg'),
        (pl.col('wt') * 100).alias('wt')
    ])
    df_scaled_1000x = df.with_columns([
        (pl.col('mpg') * 1000).alias('mpg'),
        (pl.col('wt') * 1000).alias('wt')
    ])
    
    datasets = [
        ("Original", df_original),
        ("10x scaled", df_scaled_10x),
        ("100x scaled", df_scaled_100x),
        ("1000x scaled", df_scaled_1000x)
    ]
    
    print("📊 Data Scale Analysis:")
    for name, data in datasets:
        mpg_range = f"{data['mpg'].min():.1f} to {data['mpg'].max():.1f}"
        wt_range = f"{data['wt'].min():.1f} to {data['wt'].max():.1f}"
        print(f"  {name:12} | mpg: {mpg_range:>15} | wt: {wt_range:>15}")
    
    print("\n🎯 The Problem:")
    print("With fixed prior_scale=100, the priors become:")
    print("  - Too weak for original data (good)")
    print("  - Too strong for scaled data (bad)")
    print()
    
    # Show what the priors look like
    print("📈 Prior Distributions (N(0, prior_scale²)):")
    prior_scales = [100, 1000, 10000, 100000]
    for scale in prior_scales:
        range_95 = 1.96 * scale
        print(f"  prior_scale={scale:6} → 95% range: ±{range_95:8.0f}")
    
    print("\n💡 The Solution:")
    print("Prior scale should be proportional to data scale!")
    print("Rule of thumb: prior_scale ≈ 10 × data_std")
    
    # Calculate recommended prior scales
    print("\n📏 Recommended Prior Scales:")
    for name, data in datasets:
        mpg_std = data['mpg'].std()
        wt_std = data['wt'].std()
        recommended = max(mpg_std, wt_std) * 10
        print(f"  {name:12} | mpg_std: {mpg_std:8.1f} | wt_std: {wt_std:8.1f} | recommended: {recommended:8.0f}")
    
    print("\n🔧 Current Implementation:")
    print("JIMLA uses a fixed prior_scale=100, which works well for")
    print("typical data scales but may need adjustment for very large/small data.")
    print()
    print("For production use, consider:")
    print("1. Auto-scaling based on data standard deviation")
    print("2. Making prior_scale a user parameter")
    print("3. Using standardized data internally")

if __name__ == "__main__":
    demonstrate_prior_scale()

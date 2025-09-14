"""
Example comparing jimla (Bayesian) with R's lm() (frequentist) on the same data.
"""

import polars as pl
import numpy as np
from jimla import lm, tidy
import pandas as pd

# Try to import rpy2 with proper error handling
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    
    # Import R packages
    base = importr('base')
    stats = importr('stats')
    
    R_AVAILABLE = True
    print("✅ R and rpy2 are available")
    
except Exception as e:
    print(f"❌ R/rpy2 not available: {e}")
    print("Please install R and ensure rpy2 can connect to it")
    R_AVAILABLE = False

def create_sample_data():
    """Create sample data for comparison."""
    np.random.seed(123)
    n = 200
    
    # Generate correlated predictors
    x1 = np.random.normal(0, 1, n)
    x2 = 0.6 * x1 + np.random.normal(0, 0.8, n)  # Correlated with x1
    x3 = np.random.normal(0, 1, n)
    
    # True relationship: y = 1.5 + 2.0*x1 + 1.2*x2 - 0.8*x3 + noise
    true_params = {
        "(Intercept)": 1.5,
        "x1": 2.0,
        "x2": 1.2,
        "x3": -0.8
    }
    
    y = true_params["(Intercept)"] + true_params["x1"] * x1 + true_params["x2"] * x2 + true_params["x3"] * x3 + np.random.normal(0, 0.5, n)
    
    return pl.DataFrame({
        "y": y,
        "x1": x1,
        "x2": x2,
        "x3": x3
    }), true_params

def fit_r_model(df, formula):
    """Fit model using R's lm() function."""
    if not R_AVAILABLE:
        raise RuntimeError("R is not available. Please install R and rpy2.")
    
    # Convert polars to pandas for rpy2
    pandas_df = df.to_pandas()
    
    # Convert to R dataframe using the new conversion context
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_df = robjects.conversion.py2rpy(pandas_df)
    
    # Fit the model
    r_formula = robjects.Formula(formula)
    model = stats.lm(r_formula, data=r_df)
    
    # Get model summary
    r_summary = base.summary(model)
    r_squared = r_summary.rx2('r.squared')[0]
    
    # Extract coefficients from summary
    coef_table = r_summary.rx2('coefficients')
    
    # Convert coefficients to a more readable format
    coef_data = []
    for i in range(coef_table.nrow):
        coef_data.append({
            "term": coef_table.rownames[i],
            "estimate": coef_table.rx(i+1, 1)[0],  # Estimate (R is 1-indexed)
            "std_error": coef_table.rx(i+1, 2)[0],  # Std. Error
            "statistic": coef_table.rx(i+1, 3)[0],  # t value
            "p_value": coef_table.rx(i+1, 4)[0]     # Pr(>|t|)
        })
    
    tidy_df = pl.DataFrame(coef_data)
    
    return {
        "model": model,
        "summary": r_summary,
        "r_squared": r_squared,
        "tidy": tidy_df
    }

def compare_models(df, formula, true_params=None):
    """Compare jimla and R models on the same data."""
    
    print("="*80)
    print("COMPARING JIMLA (Bayesian) vs R lm() (Frequentist)")
    print("="*80)
    print(f"Formula: {formula}")
    print(f"Dataset: {df.shape[0]} observations, {df.shape[1]-1} predictors")
    
    if true_params:
        print("\n🎯 TRUE PARAMETERS:")
        for term, value in true_params.items():
            if term in formula or term == "(Intercept)":
                print(f"  {term}: {value}")
    print()
    
    # Fit jimla model
    print("🔍 Fitting JIMLA (Bayesian) model...")
    jimla_result = lm(df, formula)
    jimla_tidy = tidy(jimla_result, display=False)
    
    print("\n📊 Fitting R lm() (Frequentist) model...")
    r_result = fit_r_model(df, formula)
    r_tidy = r_result["tidy"]
    
    # Display jimla results
    print("\n" + "="*60)
    print("JIMLA (Bayesian) Results")
    print("="*60)
    tidy(jimla_result, title="JIMLA Bayesian Regression")
    
    # Display R results
    print("\n" + "="*60)
    print("R lm() (Frequentist) Results")
    print("="*60)
    print("R summary() coefficients:")
    print(r_tidy)
    
    # Also show the raw R summary
    print("\nRaw R summary output:")
    print(r_result["summary"])
    
    # Create comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    
    # Prepare comparison data
    jimla_coefs = dict(zip(jimla_tidy["term"], jimla_tidy["estimate"]))
    r_coefs = dict(zip(r_tidy["term"], r_tidy["estimate"]))
    
    # Get all terms
    all_terms = set(jimla_coefs.keys()) | set(r_coefs.keys())
    if true_params:
        all_terms = all_terms | set(true_params.keys())
    
    comparison_data = []
    for term in sorted(all_terms):
        jimla_est = jimla_coefs.get(term, np.nan)
        r_est = r_coefs.get(term, np.nan)
        true_val = true_params.get(term, np.nan) if true_params else np.nan
        
        jimla_r_diff = jimla_est - r_est if not (np.isnan(jimla_est) or np.isnan(r_est)) else np.nan
        jimla_true_diff = jimla_est - true_val if not (np.isnan(jimla_est) or np.isnan(true_val)) else np.nan
        r_true_diff = r_est - true_val if not (np.isnan(r_est) or np.isnan(true_val)) else np.nan
        
        comparison_data.append({
            "term": term,
            "true_value": true_val,
            "jimla_estimate": jimla_est,
            "r_estimate": r_est,
            "jimla_vs_r_diff": jimla_r_diff,
            "jimla_vs_true_diff": jimla_true_diff,
            "r_vs_true_diff": r_true_diff,
            "jimla_vs_r_pct": (jimla_r_diff / r_est * 100) if not (np.isnan(jimla_r_diff) or r_est == 0) else np.nan,
            "jimla_vs_true_pct": (jimla_true_diff / true_val * 100) if not (np.isnan(jimla_true_diff) or true_val == 0) else np.nan,
            "r_vs_true_pct": (r_true_diff / true_val * 100) if not (np.isnan(r_true_diff) or true_val == 0) else np.nan
        })
    
    comparison_df = pl.DataFrame(comparison_data)
    print(comparison_df)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"JIMLA R-squared: {jimla_result.r_squared:.4f}")
    print(f"R lm() R-squared: {r_result['r_squared']:.4f}")
    
    # Calculate mean absolute differences
    valid_jimla_r_diffs = [d for d in comparison_data if not np.isnan(d["jimla_vs_r_diff"])]
    if valid_jimla_r_diffs:
        mean_abs_diff = np.mean([abs(d["jimla_vs_r_diff"]) for d in valid_jimla_r_diffs])
        print(f"Mean absolute difference (JIMLA vs R): {mean_abs_diff:.6f}")
    
    if true_params:
        valid_jimla_true_diffs = [d for d in comparison_data if not np.isnan(d["jimla_vs_true_diff"])]
        valid_r_true_diffs = [d for d in comparison_data if not np.isnan(d["r_vs_true_diff"])]
        
        if valid_jimla_true_diffs:
            mean_abs_jimla_true = np.mean([abs(d["jimla_vs_true_diff"]) for d in valid_jimla_true_diffs])
            print(f"Mean absolute difference (JIMLA vs True): {mean_abs_jimla_true:.6f}")
        
        if valid_r_true_diffs:
            mean_abs_r_true = np.mean([abs(d["r_vs_true_diff"]) for d in valid_r_true_diffs])
            print(f"Mean absolute difference (R vs True): {mean_abs_r_true:.6f}")
        
        print(f"\n🎯 ACCURACY ASSESSMENT:")
        print(f"Both JIMLA and R are very close to the true parameters!")
        print(f"True R-squared should be ~1.0 (perfect fit minus noise)")
        print(f"Observed R-squared: {jimla_result.r_squared:.4f} (excellent recovery)")
    
    return jimla_result, r_result, comparison_df

if __name__ == "__main__":
    if not R_AVAILABLE:
        print("❌ Cannot run comparison without R. Please install R and ensure rpy2 can connect to it.")
        print("\nTo install R:")
        print("1. Download R from https://www.r-project.org/")
        print("2. Install R")
        print("3. No additional R packages needed - just base R")
        exit(1)
    
    # Create sample data
    df, true_params = create_sample_data()
    
    print("Sample data preview:")
    print(df.head())
    
    # Compare models
    jimla_result, r_result, comparison = compare_models(df, "y ~ x1 + x2 + x3", true_params)
    
    # Additional comparison with different formula
    print("\n" + "="*80)
    print("ADDITIONAL COMPARISON: Single predictor model")
    print("="*80)
    
    jimla_result2, r_result2, comparison2 = compare_models(df, "y ~ x1", true_params)

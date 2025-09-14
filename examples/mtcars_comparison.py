"""
Example comparing jimla (Bayesian) with R's lm() (frequentist) on the mtcars dataset.
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

def load_mtcars():
    """Load the mtcars dataset from the provided URL."""
    mtcars_path = "https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv"
    
    # Load with polars
    df = pl.read_csv(mtcars_path)
    
    print(f"📊 mtcars dataset: {df.shape[0]} observations, {df.shape[1]} variables")
    
    return df

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

def compare_models(df, formula):
    """Compare jimla and R models on the same data."""
    
    print(f"\n🔍 Formula: {formula}")
    
    # Fit jimla model
    jimla_result = lm(df, formula)
    jimla_tidy = tidy(jimla_result, display=False)
    
    # Fit R model
    r_result = fit_r_model(df, formula)
    r_tidy = r_result["tidy"]
    
    # Display jimla results
    print("\n" + "="*50)
    print("JIMLA (Bayesian) Results")
    print("="*50)
    tidy(jimla_result, title="JIMLA Bayesian Regression")
    
    # Display R results
    print("\n" + "="*50)
    print("R lm() (Frequentist) Results")
    print("="*50)
    print("R summary() coefficients:")
    print(r_tidy)
    
    # Create comparison table
    print("\n" + "="*50)
    print("COMPARISON TABLE")
    print("="*50)
    
    # Prepare comparison data
    jimla_coefs = dict(zip(jimla_tidy["term"], jimla_tidy["estimate"]))
    r_coefs = dict(zip(r_tidy["term"], r_tidy["estimate"]))
    
    # Standardize intercept naming for comparison
    if "intercept" in jimla_coefs and "(Intercept)" in r_coefs:
        jimla_coefs["(Intercept)"] = jimla_coefs.pop("intercept")
    elif "(Intercept)" in jimla_coefs and "intercept" in r_coefs:
        r_coefs["(Intercept)"] = r_coefs.pop("intercept")
    
    # Get all terms
    all_terms = set(jimla_coefs.keys()) | set(r_coefs.keys())
    
    comparison_data = []
    for term in sorted(all_terms):
        jimla_est = jimla_coefs.get(term, np.nan)
        r_est = r_coefs.get(term, np.nan)
        
        jimla_r_diff = jimla_est - r_est if not (np.isnan(jimla_est) or np.isnan(r_est)) else np.nan
        
        comparison_data.append({
            "term": term,
            "jimla_estimate": jimla_est,
            "r_estimate": r_est,
            "difference": jimla_r_diff,
            "relative_diff_pct": (jimla_r_diff / r_est * 100) if not (np.isnan(jimla_r_diff) or r_est == 0) else np.nan
        })
    
    comparison_df = pl.DataFrame(comparison_data)
    print(comparison_df)
    
    # Summary statistics
    print(f"\n📊 R-squared: JIMLA={jimla_result.r_squared:.4f}, R={r_result['r_squared']:.4f}")
    
    return jimla_result, r_result, comparison_df

if __name__ == "__main__":
    if not R_AVAILABLE:
        print("❌ Cannot run comparison without R. Please install R and ensure rpy2 can connect to it.")
        exit(1)
    
    # Load mtcars dataset
    df = load_mtcars()
    
    # Compare models on the specified formula
    formula = "mpg ~ wt + cyl"
    jimla_result, r_result, comparison = compare_models(df, formula)
    
    # Additional comparison with different formula
    print("\n" + "="*60)
    print("ADDITIONAL COMPARISON: Single predictor model")
    print("="*60)
    
    jimla_result2, r_result2, comparison2 = compare_models(df, "mpg ~ wt")

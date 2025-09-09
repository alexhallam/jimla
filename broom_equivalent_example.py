"""
Example demonstrating JIMLA's broom-equivalent functions: augment() and glance().
These functions provide Bayesian equivalents of R's broom::augment() and broom::glance().
"""

import polars as pl
import numpy as np
from jimla import lm, tidy, augment, glance

def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(123)
    n = 100
    
    # Generate predictors
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # True relationship: y = 2 + 1.5*x1 + 0.8*x2 + noise
    y = 2 + 1.5 * x1 + 0.8 * x2 + np.random.normal(0, 0.5, n)
    
    return pl.DataFrame({
        "y": y,
        "x1": x1,
        "x2": x2,
        "id": range(n)  # Add ID column for reference
    })

def demonstrate_broom_equivalents():
    """Demonstrate all three broom-equivalent functions."""
    
    print()
    
    # Create sample data
    df = create_sample_data()
    formula = "y ~ x1 + x2"
    
    print("Sample Data:")
    print(df.head())
    print()
    
    # Fit the model
    print("Fitting Bayesian model...")
    result = lm(df, formula)
    
    # 1. TIDY - Model coefficients (equivalent to broom::tidy)
    print("\n" + "="*60)
    print("1. TIDY() - Model Coefficients")
    print("="*60)
    print()
    
    tidy_result = tidy(result, title="Model Coefficients")
    
    # 2. AUGMENT - Original data + model info (equivalent to broom::augment)
    print("\n" + "="*60)
    print("2. AUGMENT() - Original Data + Model Information")
    print("="*60)
    print()
    
    augment_result = augment(result, df, title="Augmented Data with Model Information")
    
    # Show just the model-related columns
    print("\nModel-related columns only:")
    model_cols = [".fitted", ".resid", ".fitted_std", ".fitted_low", ".fitted_high", ".hat", ".std.resid"]
    model_data = augment_result.select(["id", "y"] + model_cols)
    print(model_data.head(10))
    
    # 3. GLANCE - One-row model summary (equivalent to broom::glance)
    print("\n" + "="*60)
    print("3. GLANCE() - One-Row Model Summary")
    print("="*60)
    print("Equivalent to: broom::glance(model)")
    print()
    
    glance_result = glance(result, title="Model Summary (Glance)")
    
    return tidy_result, augment_result, glance_result

def compare_with_mtcars():
    """Compare with mtcars data to show real-world usage."""
    
    print("\n" + "="*80)
    print("REAL-WORLD EXAMPLE: MTCARS DATASET")
    print("="*80)
    
    # Load mtcars data
    mtcars_path = "https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv"
    df = pl.read_csv(mtcars_path)
    
    print("📊 mtcars dataset:")
    print(f"Shape: {df.shape}")
    print(df.head())
    
    # Fit model
    formula = "mpg ~ wt + cyl"
    result = lm(df, formula)
    
    # Show all three functions
    print("\n🔍 Model Coefficients (tidy):")
    tidy(result, title="mtcars: Model Coefficients")
    
    print("\n📈 Augmented Data (augment):")
    augment(result, df, title="mtcars: Augmented Data")
    
    print("\n📋 Model Summary (glance):")
    glance(result, title="mtcars: Model Summary")

if __name__ == "__main__":
    # Run the main demonstration
    tidy_result, augment_result, glance_result = demonstrate_broom_equivalents()
    
    # Run the mtcars example
    compare_with_mtcars()

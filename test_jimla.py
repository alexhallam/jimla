"""
Simple test script for jimla functionality.
"""

import polars as pl
import numpy as np

def test_basic_functionality():
    """Test basic lm() and tidy() functionality."""
    try:
        from jimla import lm, tidy
        
        # Create simple test data
        np.random.seed(42)
        n = 50
        x = np.random.normal(0, 1, n)
        y = 1 + 2 * x + np.random.normal(0, 0.1, n)
        
        df = pl.DataFrame({"y": y, "x": x})
        
        # Test lm()
        result = lm(df, "y ~ x")
        
        # Check basic properties
        assert result.formula == "y ~ x"
        assert result.n_obs == n
        assert result.n_params == 2  # intercept + slope
        assert len(result.coefficients) == 2
        assert "(Intercept)" in result.coefficients
        assert "x" in result.coefficients
        
        # Test tidy()
        tidy_result = tidy(result)
        assert tidy_result.shape[0] == 2  # 2 coefficients
        assert "term" in tidy_result.columns
        assert "estimate" in tidy_result.columns
        
        print("✅ Basic functionality test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure to install dependencies: pip install -e .")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_basic_functionality()

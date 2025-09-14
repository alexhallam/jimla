"""
Data preparation and formula parsing functionality.
"""

import jax.numpy as jnp
import polars as pl
import wayne
import fiasto_py
from typing import Tuple, List


def prepare_data_with_wayne(df: pl.DataFrame, formula: str) -> Tuple[jnp.ndarray, jnp.ndarray, List[str]]:
    """
    Prepare data for regression using wayne-trade to create design matrix and response vector.
    
    Args:
        df: Polars DataFrame
        formula: Wilkinson's formula string (e.g., "y ~ x1 + x2")
        
    Returns:
        Tuple of (design_matrix, response_vector, column_names)
    """
    try:
        # Use wayne-trade to convert formula to model matrix
        model_matrix = wayne.trade_formula_for_matrix(df, formula)
        
        # Extract response variable using fiasto-py
        parsed_formula = fiasto_py.parse_formula(formula)
        # Find the response variable from the columns
        response_var = None
        for var_name, var_info in parsed_formula['columns'].items():
            if 'Response' in var_info['roles']:
                response_var = var_name
                break
        
        if response_var is None:
            raise ValueError(f"No response variable found in formula '{formula}'")
        
        y = df.select(response_var).to_numpy().ravel()
        
        # Get design matrix (all columns from wayne - this includes intercept and all terms)
        X = model_matrix.to_numpy()
        
        # Get column names from wayne-trade
        column_names = model_matrix.columns
        
        return jnp.array(X), jnp.array(y), column_names
        
    except Exception as e:
        raise ValueError(f"Failed to process formula '{formula}' with wayne-trade: {e}")

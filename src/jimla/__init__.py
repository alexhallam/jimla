"""
jimla: regression with variational inference

A Python package for Bayesian linear regression using fiasto-py for formula parsing
and blackjax for variational inference, with broom-style tidy output.
"""

from .models import lm, RegressionResult
from .broom import tidy, augment, glance

__version__ = "0.1.0"
__all__ = ["lm", "tidy", "augment", "glance", "RegressionResult"]
"""Shared evaluation contract for held-out utility and fairness measurements."""

from .contract import EvaluationContract, EvaluationError
from .splitting import split_source_records

__all__ = ["EvaluationContract", "EvaluationError"]

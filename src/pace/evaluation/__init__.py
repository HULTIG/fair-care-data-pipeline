"""Shared evaluation contract for held-out utility and fairness measurements."""

from .contract import EvaluationContract, EvaluationError

__all__ = ["EvaluationContract", "EvaluationError"]

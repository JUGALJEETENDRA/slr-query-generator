"""Compatibility entry point for the local-AI gold-set evaluator."""

from evaluation.local_ai_benchmark import evaluate_files as evaluate_benchmark

__all__ = ["evaluate_benchmark"]

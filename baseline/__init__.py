"""
Baseline Module
===============
Implementation of traditional optimization methods for ITSN.

This module contains:
- BaselineZFSDROptimizer: Zero-Forcing + Semi-Definite Relaxation optimizer
- evaluate_baseline: Evaluation scripts for baseline performance
"""

from .baseline_optimizer import BaselineZFSDROptimizer

__all__ = ['BaselineZFSDROptimizer']
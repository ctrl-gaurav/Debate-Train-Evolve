"""
DTE Framework: Debate, Train, Evolve

A complete implementation of the Debate, Train, Evolve framework for improving
language model reasoning through multi-agent debate and iterative training.

This package provides:
- Multi-agent debate with RCR (Reflect-Critique-Refine) prompting
- GRPO (Group Relative Policy Optimization) training
- End-to-end pipeline orchestration
- Comprehensive evaluation and monitoring

Author: DTE Research Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "DTE Research Team"

# Core imports - only import what doesn't require external ML dependencies
from .core.config import DTEConfig

# Conditional imports for components that require external dependencies
try:
    from .core.pipeline import DTEPipeline
    from .debate.manager import DebateManager
    from .training.grpo_trainer import GRPOTrainer
    _FULL_IMPORTS_AVAILABLE = True
except ImportError:
    DTEPipeline = None
    DebateManager = None
    GRPOTrainer = None
    _FULL_IMPORTS_AVAILABLE = False

__all__ = [
    "DTEConfig",
]

# Add to __all__ only if imports succeeded
if _FULL_IMPORTS_AVAILABLE:
    __all__.extend([
        "DTEPipeline",
        "DebateManager",
        "GRPOTrainer",
    ])
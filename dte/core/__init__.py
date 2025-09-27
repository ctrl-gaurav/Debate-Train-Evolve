"""Core DTE Framework components."""

from .config import DTEConfig
from .logger import DTELogger

# Conditional import for pipeline
try:
    from .pipeline import DTEPipeline
    __all__ = ["DTEConfig", "DTEPipeline", "DTELogger"]
except ImportError:
    __all__ = ["DTEConfig", "DTELogger"]
"""Multi-agent debate components with structured prompting."""

from .agent import DebateAgent
from .manager import DebateManager
from .prompts import DebatePromptManager

__all__ = ["DebateAgent", "DebateManager", "DebatePromptManager"]
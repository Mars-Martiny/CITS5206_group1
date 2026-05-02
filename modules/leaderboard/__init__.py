"""Leaderboard module — tracks model runs in a persistent CSV."""

from .logger import log_run
from .runner import resolve_runner

__all__ = ["log_run", "resolve_runner"]

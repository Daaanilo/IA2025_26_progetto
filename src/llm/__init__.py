"""Lazy LLM package initialization.

This module avoids importing heavy LLM-related dependencies at import time.
Call `Helper(config)` or `Reviewer(config, device)` to create instances; the
actual classes will be imported lazily when needed.
"""

from typing import Any, Dict


def Helper(config: Dict[str, Any]):
	"""Factory that returns a Helper instance (imports lazily)."""
	from src.llm.helper import Helper as _Helper

	return _Helper(config)


def Reviewer(config: Dict[str, Any], device: str = "cuda"):
	"""Factory that returns a Reviewer instance (imports lazily)."""
	from src.llm.reviewer import Reviewer as _Reviewer

	return _Reviewer(config, device=device)


__all__ = ["Helper", "Reviewer"]

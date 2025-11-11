"""
HeRoN Crafter Project
=====================

An implementation of the HeRoN (Helper-Reviewer-NPC) architecture
for adaptive NPC behavior in the Crafter environment.
"""
__version__ = "0.1.0"

# Import main components for easy access
from src.agents import DQNAgent, DQNNetwork, ReplayBuffer
from src.environment import CrafterWrapper, make_crafter_env
from src.llm import Helper, Reviewer

__all__ = [
    # Version
    '__version__',
    # Agents
    'DQNAgent',
    'DQNNetwork', 
    'ReplayBuffer',
    # Environment
    'CrafterWrapper',
    'make_crafter_env',
    # LLM
    'Helper',
    'Reviewer',
]

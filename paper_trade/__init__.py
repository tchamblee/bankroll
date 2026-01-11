"""
Paper Trade Module

Virtual paper trading with IBKR data feed.
Supports directional barriers and volatility scaling.
"""
from .app import PaperTradeApp, main
from .data_manager import LiveDataManager
from .execution import ExecutionEngine
from .utils import logger, play_sound

__all__ = [
    'PaperTradeApp',
    'LiveDataManager',
    'ExecutionEngine',
    'main',
    'logger',
    'play_sound',
]

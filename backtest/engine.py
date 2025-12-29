from .core import BacktestEngineBase
from .mixins.generation import SignalGenerationMixin
from .mixins.simulation import SimulationMixin
from .mixins.evaluation import EvaluationMixin

class BacktestEngine(BacktestEngineBase, SignalGenerationMixin, SimulationMixin, EvaluationMixin):
    """
    High-Performance Backtester using Centralized Trade Simulator.
    Composed of modular mixins for Generation, Simulation, and Evaluation.
    """
    pass
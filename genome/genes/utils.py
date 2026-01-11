from .relational import RelationalGene
from .delta import DeltaGene
from .flux import FluxGene
from .divergence import DivergenceGene
from .efficiency import EfficiencyGene
from .zscore import ZScoreGene
from .soft_zscore import SoftZScoreGene
from .correlation import CorrelationGene
from .time import TimeGene
from .seasonality import SeasonalityGene
from .consecutive import ConsecutiveGene
from .cross import CrossGene
from .persistence import PersistenceGene
from .squeeze import SqueezeGene
from .event import EventGene
from .extrema import ExtremaGene
from .mean_reversion import MeanReversionGene
from .hysteresis import HysteresisGene
from .proximity import ProximityGene
from .validity import ValidityGene

def gene_from_dict(d):
    """Factory to restore gene from dictionary."""
    if d['type'] == 'relational':
        return RelationalGene(d['feature_left'], d['operator'], d['feature_right'])
    elif d['type'] == 'delta':
        return DeltaGene(d['feature'], d['operator'], d['threshold'], d['lookback'])
    elif d['type'] == 'flux':
        return FluxGene(d['feature'], d['operator'], d['threshold'], d['lag'])
    elif d['type'] == 'divergence':
        return DivergenceGene(d['feature_a'], d['feature_b'], d['window'])
    elif d['type'] == 'efficiency':
        return EfficiencyGene(d['feature'], d['operator'], d['threshold'], d['window'])
    elif d['type'] == 'zscore':
        return ZScoreGene(d['feature'], d['operator'], d['threshold'], d['window'])
    elif d['type'] == 'soft_zscore':
        return SoftZScoreGene(d['feature'], d['operator'], d['threshold'], d['window'], d.get('slope', 1.0))
    elif d['type'] == 'correlation':
        return CorrelationGene(d['feature_left'], d['feature_right'], d['operator'], d['threshold'], d['window'])
    elif d['type'] == 'time':
        return TimeGene(d['mode'], d['operator'], d['value'])
    elif d['type'] == 'seasonality':
        return SeasonalityGene(d['operator'], d['threshold'])
    elif d['type'] == 'consecutive':
        return ConsecutiveGene(d['direction'], d['operator'], d['count'])
    elif d['type'] == 'cross':
        return CrossGene(d['feature_left'], d['direction'], d['feature_right'])
    elif d['type'] == 'persistence':
        return PersistenceGene(d['feature'], d['operator'], d['threshold'], d['window'])
    elif d['type'] == 'squeeze':
        return SqueezeGene(d['feature_short'], d['feature_long'], d['multiplier'])
    elif d['type'] == 'event':
        return EventGene(d['feature'], d['operator'], d['threshold'], d['window'])
    elif d['type'] == 'extrema':
        return ExtremaGene(d['feature'], d['mode'], d['window'])
    elif d['type'] == 'mean_reversion':
        return MeanReversionGene(d['feature'], d['regime_feature'], d['threshold'], d['regime_threshold'], d['direction'], d['window'])
    elif d['type'] == 'hysteresis':
        return HysteresisGene(d['feature'], d['operator'], d['window'])
    elif d['type'] == 'proximity':
        return ProximityGene(d['feature'], d['mode'], d['threshold'], d['window'])
    elif d['type'] == 'validity':
        return ValidityGene(d['feature'], d['operator'], d['threshold'], d['window'], d['percentage'])
    return None

import importlib.metadata
__version__ = importlib.metadata.version('ParticlePhaseSpace')

# from ParticlePhaseSpace._ParticlePhaseSpace import PhaseSpace
from ParticlePhaseSpace.__unit_config__ import ParticlePhaseSpaceUnits, UnitSet
from ParticlePhaseSpace._ParticlePhaseSpace import PhaseSpace

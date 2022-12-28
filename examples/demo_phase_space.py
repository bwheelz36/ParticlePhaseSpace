
from ParticlePhaseSpace.ParticlePhaseSpace import ParticlePhaseSpace
from pathlib import Path


PS = ParticlePhaseSpace()
test_data = Path(r'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0').resolve()
PS.DataLoaders.TopasDataLoader(test_data)

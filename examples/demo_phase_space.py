
from ParticlePhaseSpace import DataLoaders
from ParticlePhaseSpace.ParticlePhaseSpace import ParticlePhaseSpace

from pathlib import Path

# load topas data
test_data_loc = Path(r'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp').absolute()
ps_data = DataLoaders.LoadTopasData(test_data_loc)


PS = ParticlePhaseSpace(ps_data._data)
electrons_only = PS('electrons')

from pathlib import Path
from ParticlePhaseSpace import DataLoaders
from ParticlePhaseSpace._ParticlePhaseSpace import PhaseSpace

data_loc  = Path(r'/home/brendan/Dropbox (Sydney Uni)/Projects/PhaserSims/LinacPhaseSpace/atExit_Jul29_2020_tpsImport.phsp')
data = DataLoaders.LoadTopasData(data_loc)

PS = PhaseSpace(data)
PS.calculate_twiss_parameters()
PS.assess_density_versus_r()
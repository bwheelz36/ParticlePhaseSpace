from pathlib import Path

import numpy as np

from ParticlePhaseSpace import DataLoaders
from ParticlePhaseSpace._ParticlePhaseSpace import PhaseSpace


data_loc = Path('../tests/test_data/coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
data = DataLoaders.LoadTopasData(data_loc)
PS = PhaseSpace(data)

from ParticlePhaseSpace.DataExporters import NewDataExporter

NewDataExporter(PS,'.','test_new_exporter.dat')

with open('test_new_exporter.dat') as f:
    # file_contents = f.read()
    fuck = np.loadtxt(f, header)

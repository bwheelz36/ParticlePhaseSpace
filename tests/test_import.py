import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
this_file_loc = Path(__file__).parent
sys.path.insert(0, str(this_file_loc.parent))
from ParticlePhaseSpace import DataExporters
from ParticlePhaseSpace import DataLoaders
from ParticlePhaseSpace import PhaseSpace
import pytest



def test_pandas_import_fails_when_particles_specfied():

    demo_data = pd.DataFrame(
        {'x [mm]': [0, 1, 2],
         'y [mm]': [0, 1, 2],
         'z [mm]': [0, 1, 2],
         'px [MeV/c]': [0, 1, 2],
         'py [MeV/c]': [0, 1, 2],
         'pz [MeV/c]': [0, 1, 2],
         'particle type [pdg_code]': [11, 11, 11],
         'weight': [0, 1, 2],
         'particle id': [0, 1, 2],
         'time [ps]': [0, 1, 2]})

    with pytest.raises(Exception):
        data = DataLoaders.LoadPandasData(demo_data)


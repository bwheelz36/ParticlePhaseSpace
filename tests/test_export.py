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

def test_topas_export():

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

    data = DataLoaders.LoadPandasData(demo_data, particle_type='electrons')
    PS = PhaseSpace(data)
    # ok: can we export this data:
    DataExporters.Topas_Exporter(PS,output_location='.', output_name='test.phsp')
    # now check we can read it back in:
    # data = DataLoaders.LoadTopasData('test.phsp')
    # PS2 = PhaseSpace(data)
    # assert np.allclose(PS.ps_data, PS2.ps_data)



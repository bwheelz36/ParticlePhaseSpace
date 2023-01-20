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

def test_topas_export():

    demo_data = pd.DataFrame(
                    {'x [mm]': [0, 1, 2],
                     'y [mm]': [0, 1, 2],
                     'z [mm]': [0, 1, 2],
                     'px [MeV/c]': [1, 1, 2],
                     'py [MeV/c]': [1, 1, 2],
                     'pz [MeV/c]': [1, 1, 2],
                     'particle type [pdg_code]': [11, 11, 11],
                     'weight': [0, 1, 2],
                     'particle id': [0, 1, 2],
                     'time [ps]': [0, 0, 0]})

    data = DataLoaders.LoadPandasData(demo_data)
    PS = PhaseSpace(data)
    # ok: can we export this data:
    DataExporters.Topas_Exporter(PS,output_location='.', output_name='test.phsp')
    # now check we can read it back in:
    data = DataLoaders.LoadTopasData('test.phsp')
    PS2 = PhaseSpace(data)
    PS.reset_phase_space()
    gah = PS.ps_data - PS2.ps_data
    assert all(gah.max() < 1e-5)


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
from ParticlePhaseSpace import ParticlePhaseSpaceUnits
import pytest
from urllib import request
'''
note that the topas data loader functionality is inhenerently tested in test_ParticlePhaseSpace
'''

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
        data = DataLoaders.Load_PandasData(demo_data)


def test_tibaray_import():
    data_loc = this_file_loc / 'test_data' / 'tibaray_test.dat'
    data = DataLoaders.Load_TibarayData(data_loc, particle_type='electrons')
    PS = PhaseSpace(data)
    # check that energy is stable
    PS.calculate_energy_statistics()
    with open(this_file_loc / 'test_data' / 'tibaray_energy.json') as f:
        old_data = json.load(f)
    for particle_key in old_data:
        for energy_key  in old_data[particle_key]:
            np.allclose(old_data[particle_key][energy_key],
                        PS.energy_stats[particle_key][energy_key])


def test_tibaray_import_fails_when_no_particles_specified():
    data_loc = this_file_loc / 'test_data' / 'tibaray_test.dat'
    with pytest.raises(Exception):
        data = DataLoaders.Load_TibarayData(data_loc)


def test_p2sat_txt():
    # get data file:
    available_units = ParticlePhaseSpaceUnits()
    data_url = 'https://raw.githubusercontent.com/lesnat/p2sat/master/examples/ExamplePhaseSpace.csv'
    file_name = 'p2sat_txt_test.csv'
    request.urlretrieve(data_url, file_name)
    # read in
    ps_data = DataLoaders.Load_p2sat_txt(file_name, particle_type='electrons', units=available_units('p2_sat_UHI'))
    PS = PhaseSpace(ps_data)
    energy_dict = PS.calculate_energy_statistics()
    original_values = {'electrons':
                           {'number': 4488,
                            'min energy': 0.0008218226687650709,
                            'max energy': 11.461781660179325,
                            'mean energy': 1.486820558657867,
                            'std mean': 1.5383426055210419,
                            'median energy': 1.036878156474783,
                            'energy spread IQR': 1.507862639804363}
                       }
    for key in original_values['electrons'].keys():
        assert np.allclose(original_values['electrons'][key], PS.energy_stats['electrons'][key])
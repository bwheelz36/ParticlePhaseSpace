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
        data = DataLoaders.Load_PandasData(demo_data, particle_type='electrons')


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


def test_iaea():
    """
    this tests we can read one particular IAEA file that we know the format for
    :return:
    """
    file_name = this_file_loc / 'test_data' / 'test_iaea.phsp'
    # read in
    data_schema = np.dtype([
        ('particle type', 'i1'),
        ('Ek', 'f4'),
        ('x', 'f4'),
        ('y', 'f4'),
        ('Cosine X', 'f4'),
        ('Cosine Y', 'f4')])

    constants = {'z': np.float32(26.7), 'weight': np.int8(1)}
    ps_data = DataLoaders.Load_IAEA(data_schema=data_schema, constants=constants, input_data=file_name)
    PS = PhaseSpace(ps_data)
    PS.calculate_energy_statistics()
    original_values = {'gammas': {'number': 9919,
                          'min energy': 0.013054816,
                          'max energy': 6.090453,
                          'mean energy': 1.288256936058273,
                          'std mean': 1.150713867434605,
                          'median energy': 0.8709397,
                          'energy spread IQR': 1.3069977313280108},
                     'electrons': {'number': 79,
                          'min energy': 0.010767639,
                          'max energy': 3.2847178,
                          'mean energy': 0.7908395317536366,
                          'std mean': 0.740570250465219,
                          'median energy': 0.5534066,
                          'energy spread IQR': 0.8670903742313385},
                     'positrons': {'number': 2,
                          'min energy': 0.8010578,
                          'max energy': 2.5652504,
                          'mean energy': 1.6831541061401367,
                          'std mean': 0.8820962954544692,
                          'median energy': 1.6831541,
                          'energy spread IQR': 1.7641925811767578}}
    for particle in original_values.keys():
        for key in original_values[particle].keys():
            assert np.allclose(original_values[particle][key], PS.energy_stats[particle][key])

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
this_file_loc = Path(__file__)
sys.path.insert(0, str(this_file_loc.parent.parent))
from ParticlePhaseSpace import DataLoaders
from ParticlePhaseSpace import PhaseSpace
import ParticlePhaseSpace.__phase_space_config__ as ps_cfg
import ParticlePhaseSpace.__particle_config__ as particle_cfg

test_data_loc = Path('test_data/coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp').absolute()
if not Path('test_data').is_dir():
    print('i smell like balls')

import os
print(f'we are currently in {os.getcwd()}')
os.listdir(os.getcwd())
data = DataLoaders.LoadTopasData(test_data_loc)
PS = PhaseSpace(data)

def test_all_allowed_columns_can_be_filled():

    for col in ps_cfg.allowed_columns:
        if col == 'Ek [MeV]':
            if col == 'Ek [MeV]':
                PS.fill_kinetic_E()
            elif col == 'rest mass [MeV/c^2]':
                PS.fill_rest_mass()
            elif col in 'gamma, beta':
                PS.fill_beta_and_gamma()
            elif col in 'vx [m/s], vy [m/s], vz [m/s]':
                PS.fill_velocity()
            elif col in 'Direction Cosine X, Direction Cosine Y, Direction Cosine Z':
                PS.fill_direction_cosines()
            else:
                raise Exception(f'unable to fill required column {col}')

def test_downsample_phase_space():

    PS_downsample = PS.get_downsampled_phase_space(downsample_factor=10)
    assert len(PS)/len(PS_downsample) > 9.5 and len(PS)/len(PS_downsample) < 10.5  # to account for rounding

def test_twiss():

    PS.calculate_twiss_parameters(beam_direction='z')
    # compare with previous calc
    with open('test_data/twiss_stability.json') as f:
        old_data = json.load(f)

    for particle_key in old_data:
        for direction_key in old_data[particle_key]:
            for twiss_key  in old_data[particle_key][direction_key]:
                np.allclose(old_data[particle_key][direction_key][twiss_key],
                            PS.twiss_parameters[particle_key][direction_key][twiss_key])

def test_project_particles():
    if not 'vx [m/s]' in PS.ps_data.columns:
        PS.fill_velocity()

    # manual projection:
    project_dist = 100
    x1 = PS.ps_data['x [mm]'][0]
    y1 = PS.ps_data['y [mm]'][0]
    z1 = PS.ps_data['z [mm]'][0]
    vx1 = PS.ps_data['vx [m/s]'][0]
    vy1 = PS.ps_data['vy [m/s]'][0]
    vz1 = PS.ps_data['vz [m/s]'][0]

    x2 = x1 + ((vx1/vz1) * project_dist)
    y2 = y1 + ((vy1 / vz1) * project_dist)
    z2 = z1 + ((vz1 / vz1) * project_dist)
    PS.project_particles(beam_direction='z', distance=project_dist)
    # compare:
    np.allclose([x2,y2,z2], [PS.ps_data['x [mm]'][0], PS.ps_data['y [mm]'][0], PS.ps_data['z [mm]'][0]])

def test_reset_phase_space():
    PS.reset_phase_space()
    # following this only required columns should be included:
    for col_name in PS.ps_data.columns:
        if not col_name in ps_cfg.required_columns:
            raise AttributeError(f'non allowed column "{col_name}" in data.')

def test_get_particle_density():

    particle_density = PS.assess_density_versus_r()
    old_density = pd.read_csv('test_data/particle_density.csv', index_col=0).squeeze("columns")
    np.allclose(particle_density, old_density)

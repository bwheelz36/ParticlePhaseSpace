import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

this_file_loc = Path(__file__).parent
sys.path.insert(0, str(this_file_loc.parent))
from ParticlePhaseSpace import DataLoaders
from ParticlePhaseSpace import PhaseSpace
import ParticlePhaseSpace.__phase_space_config__ as ps_cfg
import pytest
import ParticlePhaseSpace.__particle_config__ as particle_cfg
from ParticlePhaseSpace import ParticlePhaseSpaceUnits

test_data_loc = this_file_loc / 'test_data'
data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
PS = PhaseSpace(data)


def test_all_allowed_columns_can_be_filled():
    for col in list(ps_cfg.allowed_columns.keys()):
        try:
            PS.fill.__getattribute__(ps_cfg.allowed_columns[col])()
        except AttributeError:
            raise AttributeError(f'unable to fill required column {col}')


def test_downsample_phase_space():
    PS_downsample = PS.get_downsampled_phase_space(downsample_factor=10)
    assert len(PS) / len(PS_downsample) > 9.5 and len(PS) / len(PS_downsample) < 10.5  # to account for rounding


def test_twiss_stability():
    PS.calculate_twiss_parameters(beam_direction='z')
    # compare with previous calc
    with open(test_data_loc / 'twiss_stability.json') as f:
        old_data = json.load(f)

    for particle_key in old_data:
        for direction_key in old_data[particle_key]:
            for twiss_key in old_data[particle_key][direction_key]:
                np.allclose(old_data[particle_key][direction_key][twiss_key],
                            PS.twiss_parameters[particle_key][direction_key][twiss_key])


def test_energy_stats_stability():
    PS.calculate_energy_statistics()
    # compare with previous calc
    with open(test_data_loc / 'energy_stats.json') as f:
        old_data = json.load(f)
    for particle_key in old_data:
        for energy_key in old_data[particle_key]:
            np.allclose(old_data[particle_key][energy_key],
                        PS.energy_stats[particle_key][energy_key])


def test_project_particles():
    # manual projection in z direction:
    project_dist = 100
    x1 = PS.ps_data['x [mm]'][0]
    y1 = PS.ps_data['y [mm]'][0]
    z1 = PS.ps_data['z [mm]'][0]
    px1 = PS.ps_data['px [MeV/c]'][0]
    py1 = PS.ps_data['py [MeV/c]'][0]
    pz1 = PS.ps_data['pz [MeV/c]'][0]

    x2 = x1 + ((px1 / pz1) * project_dist)
    y2 = y1 + ((py1 / pz1) * project_dist)
    z2 = z1 + ((pz1 / pz1) * project_dist)
    PS_projected = PS.transform.project(direction='z', distance=project_dist)
    # compare:
    assert np.allclose([x2, y2, z2], [PS_projected.ps_data['x [mm]'][0], PS_projected.ps_data['y [mm]'][0],
                                      PS_projected.ps_data['z [mm]'][0]])

    # manual projection in x direction:
    project_dist = 100
    x1 = PS.ps_data['x [mm]'][0]
    y1 = PS.ps_data['y [mm]'][0]
    z1 = PS.ps_data['z [mm]'][0]
    px1 = PS.ps_data['px [MeV/c]'][0]
    py1 = PS.ps_data['py [MeV/c]'][0]
    pz1 = PS.ps_data['pz [MeV/c]'][0]

    x2 = x1 + ((px1 / px1) * project_dist)
    y2 = y1 + ((py1 / px1) * project_dist)
    z2 = z1 + ((pz1 / px1) * project_dist)
    PS_projected = PS.transform.project(direction='x', distance=project_dist)
    # compare:
    assert np.allclose([x2, y2, z2], [PS_projected.ps_data['x [mm]'][0], PS_projected.ps_data['y [mm]'][0],
                                      PS_projected.ps_data['z [mm]'][0]])

    # manual projection in y direction:
    project_dist = 100
    x1 = PS.ps_data['x [mm]'][0]
    y1 = PS.ps_data['y [mm]'][0]
    z1 = PS.ps_data['z [mm]'][0]
    px1 = PS.ps_data['px [MeV/c]'][0]
    py1 = PS.ps_data['py [MeV/c]'][0]
    pz1 = PS.ps_data['pz [MeV/c]'][0]

    x2 = x1 + ((px1 / py1) * project_dist)
    y2 = y1 + ((py1 / py1) * project_dist)
    z2 = z1 + ((pz1 / py1) * project_dist)
    PS_projected = PS.transform.project(direction='y', distance=project_dist)
    # compare:
    assert np.allclose([x2, y2, z2], [PS_projected.ps_data['x [mm]'][0], PS_projected.ps_data['y [mm]'][0],
                                      PS_projected.ps_data['z [mm]'][0]])

    # now in place:
    x1 = PS.ps_data['x [mm]'][0]
    y1 = PS.ps_data['y [mm]'][0]
    z1 = PS.ps_data['z [mm]'][0]
    px1 = PS.ps_data['px [MeV/c]'][0]
    py1 = PS.ps_data['py [MeV/c]'][0]
    pz1 = PS.ps_data['pz [MeV/c]'][0]

    x2 = x1 + ((px1 / pz1) * project_dist)
    y2 = y1 + ((py1 / pz1) * project_dist)
    z2 = z1 + ((pz1 / pz1) * project_dist)
    PS.transform.project(direction='z', distance=project_dist, in_place=True)
    # compare:
    assert np.allclose([x2, y2, z2], [PS.ps_data['x [mm]'][0], PS.ps_data['y [mm]'][0],
                                      PS.ps_data['z [mm]'][0]])


def test_reset_phase_space():
    PS.reset_phase_space()
    # following this only required columns should be included:
    for col_name in PS.ps_data.columns:
        if not col_name in ps_cfg.get_required_column_names(PS._units):
            raise AttributeError(f'non allowed column "{col_name}" in data.')


def test_get_particle_density():
    particle_density = PS.assess_density_versus_r(verbose=False)
    old_density = pd.read_csv(test_data_loc / 'particle_density.csv', index_col=0).squeeze("columns")
    np.allclose(particle_density, old_density)


def test_get_seperated_phase_space():
    # had to reload the data for some reason:
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)

    electron_PS = PS('electrons')
    electron_PS2 = PS(11)
    assert np.allclose(electron_PS.ps_data, electron_PS2.ps_data)
    assert len(electron_PS) == 2853

    no_electron_PS = PS - electron_PS


def test_add_phase_space():
    # had to reload the data for some reason:
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)

    gamma_PS, electron_PS = PS(['gammas', 'electrons'])

    combine_PS = gamma_PS + electron_PS
    assert len(combine_PS) == len(gamma_PS) + len(electron_PS)
    with pytest.raises(Exception):
        wont_work_PS = gamma_PS + gamma_PS


def test_manipulate_data():
    old_x_mean = PS.ps_data['x [mm]'].mean()
    PS.ps_data['x [mm]'] = PS.ps_data['x [mm]'] + 2
    assert np.allclose(PS.ps_data['x [mm]'].mean(), old_x_mean + 2)


def test_filter_by_time():
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)
    PS.ps_data['time [ps]'] = np.arange(len(PS))
    filtered_PS = PS.filter_by_time(t_start=0, t_finish=np.floor(len(PS) / 2))
    assert len(filtered_PS) == np.ceil(len(PS) / 2)


def test_PS_reset():
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)
    PS.calculate_twiss_parameters()
    assert PS.twiss_parameters
    PS.calculate_energy_statistics()
    assert PS.energy_stats
    PS.ps_data = PS.ps_data
    # this should have removed the twiss parameters and energy stats
    assert not PS.twiss_parameters
    assert not PS.energy_stats


def test_beta_gamma_momentum_relation():
    """
    for charged particles, should have

    pz = beta_z * gamma * rest_mass

    :return:
    """
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)
    PS_electrons = PS('electrons')

    PS_electrons.fill.beta_and_gamma()
    PS_electrons.fill.rest_mass()
    px = np.multiply(np.multiply(PS_electrons.ps_data['beta_x'], PS_electrons.ps_data['gamma']),
                     PS_electrons.ps_data['rest mass [MeV/c^2]'])
    assert np.allclose(px, PS_electrons.ps_data['px [MeV/c]'], atol=1e-03)
    py = np.multiply(np.multiply(PS_electrons.ps_data['beta_y'], PS_electrons.ps_data['gamma']),
                     PS_electrons.ps_data['rest mass [MeV/c^2]'])
    assert np.allclose(py, PS_electrons.ps_data['py [MeV/c]'], atol=1e-03)
    pz = np.multiply(np.multiply(PS_electrons.ps_data['beta_z'], PS_electrons.ps_data['gamma']),
                     PS_electrons.ps_data['rest mass [MeV/c^2]'])
    assert np.allclose(pz, PS_electrons.ps_data['pz [MeV/c]'], atol=1e-03)


def test_resample_kde():
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)
    PS = PS('gammas')  # take only the gammas for this example

    resample_factor = 0.5
    new_PS = PS.resample_via_gaussian_kde(n_new_particles_factor=resample_factor, interpolate_weights=False)

    assert np.allclose(resample_factor, len(new_PS) / len(PS))
    PS.calculate_energy_statistics()
    new_PS.calculate_energy_statistics()
    original_energy_stats = PS.energy_stats
    new_energy_stats = new_PS.energy_stats

    energy_cut_off = 3  # MeV; dont expect a great match here, just looking for wild outliers
    for energy_key in original_energy_stats['gammas']:
        if energy_key == 'number':
            continue
        assert abs(original_energy_stats['gammas'][energy_key] - new_energy_stats['gammas'][energy_key]) \
               < energy_cut_off


def test_regrid():
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)
    # regrid 100 bins
    quantities = ['x', 'y', 'px', 'py', 'pz']
    new_PS = PS.transform.regrid(n_bins=100, quantities=quantities)

    columns = new_PS._quantities_to_column_names(quantities)
    for col in columns:
        assert len(np.unique(new_PS.ps_data[col])) <= 100

    # regrid x into 10 bins
    new_PS = PS.transform.regrid(quantities='x', n_bins=10)
    assert len(np.unique(new_PS.ps_data['y [mm]'])) == len(np.unique(PS.ps_data['y [mm]']))
    assert len(np.unique(new_PS.ps_data['x [mm]'])) == 10
    # compare in Place to Note in Place
    new_PS = PS.transform.regrid()
    PS.transform.regrid(in_place=True)
    assert np.allclose(new_PS.ps_data, PS.ps_data)


def test_merge():
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)

    new_PS = PS.transform.regrid(n_bins=10, in_place=False)
    new_PS.merge(in_place=True)

    # test merge didn't effect energy stats:
    PS.calculate_energy_statistics()
    new_PS.calculate_energy_statistics()
    original_energy_stats = PS.energy_stats
    for particle_key in original_energy_stats:
        for energy_key in original_energy_stats[particle_key]:
            np.allclose(original_energy_stats[particle_key][energy_key],
                        new_PS.energy_stats[particle_key][energy_key])
    # test weight preservation
    assert np.isclose(PS.ps_data['weight'].sum(), new_PS.ps_data['weight'].sum())

    # test size of merge
    assert len(new_PS) == 2043  # stability check


def test_sort():
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)
    x_temp = PS.ps_data['x [mm]']
    PS.sort('x')
    assert np.allclose(np.sort(x_temp), PS.ps_data['x [mm]'])
    # test default
    PS.sort()
    # test quantities
    quantities = ['x', 'y', 'px', 'py', 'pz']
    PS.sort(quantities_to_sort=quantities)


def test_print_methods():
    # this only tests that the method runs
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)
    PS.plot.get_methods()
    PS.transform.get_methods()
    PS.fill.get_methods()


def test_filter_by_boolean_index():
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)

    index_1 = np.ones(len(PS)) > 0
    new_PS = PS.filter_by_boolean_index(index_1)  # this should do nothing
    assert np.allclose(PS.ps_data, new_PS.ps_data)

    index_1[round(len(PS)/2):] = False
    PS1, PS2 = PS.filter_by_boolean_index(index_1, split=True)
    assert np.isclose(len(PS1) / len(PS2), 1)
    # in place
    old_length = len(PS)
    PS.filter_by_boolean_index(index_1, in_place=True)
    assert np.isclose(old_length / len(PS), 2)


def test_translate():
    # test that mean coordinate matches translation in all directions
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)

    new_PS  = PS.transform.translate(direction='x', distance=100)
    assert np.allclose(new_PS.ps_data[new_PS.columns['x']], PS.ps_data[PS.columns['x']] + 100)
    new_PS  = PS.transform.translate(direction='y', distance=50)
    assert np.allclose(new_PS.ps_data[new_PS.columns['y']], PS.ps_data[PS.columns['y']] + 50)
    new_PS  = PS.transform.translate(direction='z', distance=-70)
    assert np.allclose(new_PS.ps_data[new_PS.columns['z']], PS.ps_data[PS.columns['z']] - 70)

    # test in place works
    old_x_mean = PS.ps_data[PS.columns['x']].mean()
    PS.transform.translate(direction='x', distance=100, in_place=True)
    assert np.allclose(old_x_mean+100, PS.ps_data[PS.columns['x']].mean())


def test_rotate():
    # test rotation in different directions for simple data
    units = ParticlePhaseSpaceUnits()('mm_MeV')
    all_allowed_columns = ps_cfg.get_all_column_names(units)
    demo_data = pd.DataFrame(
        {all_allowed_columns['x']: [0., 1., 2.],
         all_allowed_columns['y']: [2., 3., 4.],
         all_allowed_columns['z']: [5., 6., 7.],
         all_allowed_columns['px']: [1000, 1000, 2000],
         all_allowed_columns['py']: [1000, 1000, 2000],
         all_allowed_columns['pz']: [10000, 100000, 200000],
         all_allowed_columns['particle type']: [11, 11, 11],
         all_allowed_columns['weight']: [1, 1, 2],
         all_allowed_columns['particle id']: [0, 1, 2],
         all_allowed_columns['time']: [0, 0, 0]})
    data = DataLoaders.Load_PandasData(demo_data)
    PS = PhaseSpace(data)

    PS_rotate = PS.transform.rotate(rotation_axis='x', angle=-90)
    assert np.allclose(PS.ps_data['z [mm]'], PS_rotate.ps_data['y [mm]'])
    PS_rotate = PS.transform.rotate(rotation_axis='y', angle=-90)
    assert np.allclose(PS.ps_data['x [mm]'], PS_rotate.ps_data['z [mm]'])
    PS_rotate = PS.transform.rotate(rotation_axis='z', angle=90)
    assert np.allclose(PS.ps_data['x [mm]'], PS_rotate.ps_data['y [mm]'])

    # test in place
    old_y = PS.ps_data['y [mm]']
    PS.transform.rotate(rotation_axis='x', angle=90, in_place=True)
    assert np.allclose(old_y, PS.ps_data['z [mm]'])


def test_rotate_momentum():

    units = ParticlePhaseSpaceUnits()('mm_MeV')
    all_allowed_columns = ps_cfg.get_all_column_names(units)
    demo_data = pd.DataFrame(
        {all_allowed_columns['x']: [0., 1., 2.],
         all_allowed_columns['y']: [2., 3., 4.],
         all_allowed_columns['z']: [5., 6., 7.],
         all_allowed_columns['px']: [1000, 1000, 2000],
         all_allowed_columns['py']: [1000, 1000, 2000],
         all_allowed_columns['pz']: [10000, 100000, 200000],
         all_allowed_columns['particle type']: [11, 11, 11],
         all_allowed_columns['weight']: [1, 1, 2],
         all_allowed_columns['particle id']: [0, 1, 2],
         all_allowed_columns['time']: [0, 0, 0]})
    data = DataLoaders.Load_PandasData(demo_data)
    PS = PhaseSpace(data)

    PS_rotate = PS.transform.rotate(rotation_axis='x', angle=-90, rotate_momentum_vector=True)
    assert np.allclose(PS.ps_data['pz [MeV/c]'], PS_rotate.ps_data['py [MeV/c]'])
    assert np.allclose(PS.ps_data['z [mm]'], PS_rotate.ps_data['y [mm]'])
    PS_rotate = PS.transform.rotate(rotation_axis='y', angle=-90, rotate_momentum_vector=True)
    assert np.allclose(PS.ps_data['px [MeV/c]'], PS_rotate.ps_data['pz [MeV/c]'])
    assert np.allclose(PS.ps_data['x [mm]'], PS_rotate.ps_data['z [mm]'])
    PS_rotate = PS.transform.rotate(rotation_axis='z', angle=90, rotate_momentum_vector=True)
    assert np.allclose(PS.ps_data['px [MeV/c]'], PS_rotate.ps_data['py [MeV/c]'])
    assert np.allclose(PS.ps_data['x [mm]'], PS_rotate.ps_data['y [mm]'])


def test_zero_momentum_particle():
    """
    occasionally due to precision issues, particles are read in with
    zero absolute momnentum. I just want to make sure such cases still
    run
    :return:
    """

    units = ParticlePhaseSpaceUnits()('mm_MeV')
    all_allowed_columns = ps_cfg.get_all_column_names(units)
    demo_data = pd.DataFrame(
        {all_allowed_columns['x']: [0., 1., 2.],
         all_allowed_columns['y']: [2., 3., 4.],
         all_allowed_columns['z']: [5., 6., 7.],
         all_allowed_columns['px']: [1000, 1000, 0],
         all_allowed_columns['py']: [1000, 1000, 0],
         all_allowed_columns['pz']: [10000, 100000, 0],
         all_allowed_columns['particle type']: [11, 11, 11],
         all_allowed_columns['weight']: [1, 1, 2],
         all_allowed_columns['particle id']: [0, 1, 2],
         all_allowed_columns['time']: [0, 0, 0]})

    data = DataLoaders.Load_PandasData(demo_data)
    PS = PhaseSpace(data)
    PS.fill.absolute_momentum()
    PS.fill.beta_and_gamma()
    PS.fill.direction_cosines()
    PS.fill.kinetic_E()
    PS.fill.relativistic_mass()
    PS.fill.rest_mass()
    PS.fill.velocity()
import sys
from pathlib import Path
import numpy as np
this_file_loc = Path(__file__).parent
sys.path.insert(0, str(this_file_loc.parent))
from ParticlePhaseSpace import ParticlePhaseSpaceUnits
from ParticlePhaseSpace import DataLoaders
from ParticlePhaseSpace import PhaseSpace


available_units = ParticlePhaseSpaceUnits()

# general tests:

def test_consistency_of_unitless_quantities():
    """
    regardless of which unit system is in use, we should get the same answer for several quantities
    :return:
    """
    test_data_loc = this_file_loc / 'test_data'
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)
    PS_electrons = PS('electrons')
    PS_electrons.fill.beta_and_gamma()
    PS_electrons.fill.direction_cosines()
    gamma_base = PS_electrons.ps_data['gamma'].copy()
    beta_base = PS_electrons.ps_data['beta_abs'].copy()
    cos_x_base = PS_electrons.ps_data['Direction Cosine X'].copy()
    cos_y_base = PS_electrons.ps_data['Direction Cosine Y'].copy()
    cos_z_base = PS_electrons.ps_data['Direction Cosine Z'].copy()
    unit_sets = available_units.get_available_unit_strings()
    for unit_set in unit_sets:
        PS_electrons.set_units(available_units(unit_set))
        PS_electrons.fill.beta_and_gamma()
        PS_electrons.fill.direction_cosines()
        for quantity, base in zip(['gamma', 'beta_abs', 'Direction Cosine X', 'Direction Cosine Y', 'Direction Cosine Z'],
                                  [gamma_base, beta_base, cos_x_base, cos_y_base, cos_z_base]):
            if not np.allclose(base, PS_electrons.ps_data[quantity], atol=1e-3):
                raise Exception(f'Inconsistent calculation of dimensionsless quantity {quantity} for unit set {unit_set}')

def test_velocity_consistency():
    """
    velocity is consistent between a large number of unit sets so provides another useful test
    :return:
    """
    test_data_loc = this_file_loc / 'test_data'
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)
    PS.fill.velocity()
    vx_base = PS.ps_data['vx [m/s]'].copy()
    vy_base = PS.ps_data['vy [m/s]'].copy()
    vz_base = PS.ps_data['vz [m/s]'].copy()
    unit_sets = available_units.get_available_unit_strings()
    for unit_set in unit_sets:
        if not available_units(unit_set).velocity.label == 'm/s':
            continue
        PS.set_units(available_units(unit_set))
        PS.fill.velocity()
        for quantity, base in zip(
                ['vx [m/s]', 'vy [m/s]', 'vz [m/s]'],
                [vx_base, vy_base, vz_base]):
            if not np.allclose(base, PS.ps_data[quantity]):
                raise Exception(
                    f'Inconsistent calculation of velocity calculation for unit set {unit_set}')

def test_forward_back_unit_conversions():
    """
    for all unit sets, we should be able to go forward, back
    and end up with the same data
    :return:
    """
    test_data_loc = this_file_loc / 'test_data'
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    data2 = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)
    PS2 = PhaseSpace(data2)
    unit_sets = available_units.get_available_unit_strings()
    base_units = available_units('mm_MeV')
    assert np.allclose(PS.ps_data, PS2.ps_data)
    for unit_set in unit_sets:
        PS2.set_units(available_units(unit_set))
        PS2.set_units(base_units)
        assert np.allclose(PS.ps_data, PS2.ps_data)

# specific unit set tests:

def test_cm_MeV():
    test_data_loc = this_file_loc / 'test_data'
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    data2 = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)
    PS2 = PhaseSpace(data2)

    PS2.set_units(available_units('cm_MeV'))

    assert np.allclose(PS.ps_data['pz [MeV/c]'], PS2.ps_data['pz [MeV/c]'])
    assert np.allclose(PS.ps_data['x [mm]'], PS2.ps_data['x [cm]']*10)

def test_um_keV():
    test_data_loc = this_file_loc / 'test_data'
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    data2 = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)
    PS2 = PhaseSpace(data2)

    PS2.set_units(available_units('um_keV'))

    assert np.allclose(PS.ps_data['pz [MeV/c]']*1e3, PS2.ps_data['pz [keV/c]'])
    assert np.allclose(PS.ps_data['x [mm]']*1e3, PS2.ps_data['x [um]'])

def test_m_eV():
    test_data_loc = this_file_loc / 'test_data'
    data = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    data2 = DataLoaders.Load_TopasData(test_data_loc / 'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')
    PS = PhaseSpace(data)
    PS2 = PhaseSpace(data2)

    PS2.set_units(available_units('m_eV'))

    assert np.allclose(PS.ps_data['pz [MeV/c]']*1e6, PS2.ps_data['pz [eV/c]'])
    assert np.allclose(PS.ps_data['x [mm]'], PS2.ps_data['x [m]'] * 1e3)

import sys
from pathlib import Path
import numpy as np
this_file_loc = Path(__file__).parent
sys.path.insert(0, str(this_file_loc.parent))
from ParticlePhaseSpace import ParticlePhaseSpaceUnits
from ParticlePhaseSpace import DataLoaders
from ParticlePhaseSpace import PhaseSpace

available_units = ParticlePhaseSpaceUnits()


def test_SI():
    """
    velocity should be the same
    :return:
    """
    pass

def test_consistency_of_unitless_quantities():
    """
    regardless of which unit system is in use, we should get the same answer for several quantities
    :return:
    """
    ps_data = DataLoaders.Load_TopasData(Path(r'test_data/coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp'))
    PS = PhaseSpace(ps_data)
    PS_electrons = PS('electrons')
    PS_electrons.fill_beta_and_gamma()
    PS_electrons.fill_direction_cosines()
    gamma_base = PS_electrons.ps_data['gamma'].copy()
    beta_base = PS_electrons.ps_data['beta_abs'].copy()
    cos_x_base = PS_electrons.ps_data['Direction Cosine X'].copy()
    cos_y_base = PS_electrons.ps_data['Direction Cosine Y'].copy()
    cos_z_base = PS_electrons.ps_data['Direction Cosine Z'].copy()
    unit_sets = available_units.get_available_unit_strings()
    for unit_set in unit_sets:
        PS_electrons.set_units(available_units(unit_set))
        PS_electrons.fill_beta_and_gamma()
        PS_electrons.fill_direction_cosines()
        for quantity, base in zip(['gamma', 'beta_abs', 'Direction Cosine X', 'Direction Cosine Y', 'Direction Cosine Z'],
                                  [gamma_base, beta_base, cos_x_base, cos_y_base, cos_z_base]):
            if not np.allclose(base, PS_electrons.ps_data[quantity]):
                raise Exception(f'Inconsistent calculation of dimensionsless quantity {quantity} for unit set {unit_set}')

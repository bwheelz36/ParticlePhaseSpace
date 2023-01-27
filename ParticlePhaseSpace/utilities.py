import numpy as np
import ParticlePhaseSpace.__phase_space_config__ as ps_cfg
import ParticlePhaseSpace.__particle_config__ as particle_cfg
from ParticlePhaseSpace import ParticlePhaseSpaceUnits
from ParticlePhaseSpace.__unit_config__ import _UnitSet

def _check_particle_types(pdg_codes):
    """
    check that the particle types in the phase space are known particle types
    """
    for code in pdg_codes:
        try:
            test = particle_cfg.particle_properties[code]
        except:
            raise AttributeError(f'unknown particle type {particle_cfg.particle_properties[code]["name"]}')

def get_rest_masses_from_pdg_codes(pdg_codes):
    """
    get a list of rest masses associated with each particle in pdg_codes

    :param pdg_codes: particle codes defined in `pdg integers <https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf>`_
    :type pdg_codes: array_like
    :return: particle_rest_mass: numpy array of particle rest mass in MeV/c^2
    """
    _check_particle_types(np.unique(pdg_codes))
    particle_rest_mass = np.zeros(len(pdg_codes))
    for particle_type in particle_cfg.particle_properties:
        ind = particle_cfg.particle_properties[particle_type]['pdg_code'] == pdg_codes
        particle_rest_mass[ind] = particle_cfg.particle_properties[particle_type]['rest_mass']
    return particle_rest_mass

def get_all_column_names(units: _UnitSet):
    """
    return a dictionary of column names appropriate for the unit set defined in units

    :param units: instance of _UnitSet defining the units to generate column names for
    :type units: _UnitSet
    :return: column_names: dictionary of unit-appropriate columns names
    """

    # if not isinstance(units, ParticlePhaseSpaceUnits):
    #     raise TypeError('units must be an instance of articlePhaseSpace.ParticlePhaseSpaceUnits')

    column_names = {'x': f'x [{units.length.label}]',
                    'y': f'y [{units.length.label}]',
                    'z': f'z [{units.length.label}]',
                    'px': f'px [{units.momentum.label}]',
                    'py': f'py [{units.momentum.label}]',
                    'pz': f'pz [{units.momentum.label}]',
                    'vx': f'vx [{units.velocity.label}]',
                    'vy': f'vy [{units.velocity.label}]',
                    'vz': f'vz [{units.velocity.label}]',
                    'time': f'time [{units.time.label}]',
                    'rest_mass': f'pz [{units.mass.label}]',
                    'relativistic_mass': f'relativistic mass [{units.mass.label}]',
                    'Ek': f'Ek [{units.energy.label}]',
                    'beta_x': 'beta_x',
                    'beta_y': 'beta_y',
                    'beta_z': 'beta_z',
                    'beta_abs': 'beta_abs',
                    'gamma': 'gamma',
                    'DirCosX': 'Direction Cosine X',
                    'DirCosY': 'Direction Cosine Y',
                    'DirCosZ': 'Direction Cosine Z',
                    'particle type': 'particle type [pdg_code]',
                    'weight': 'weight',
                    'particle id': 'particle id'}
    return column_names

def get_required_column_names(units: _UnitSet):

    all_column_names = get_all_column_names(units)
    required_columns = ps_cfg.required_columns
    required_column_names = []
    for column in required_columns:
        required_column_names.append(all_column_names[column])
    return required_column_names

import numpy as np
import ParticlePhaseSpace.__phase_space_config__ as ps_cfg
import ParticlePhaseSpace.__particle_config__ as particle_cfg
# from ParticlePhaseSpace import ParticlePhaseSpaceUnits

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

def get_column_names(units):

    # if not isinstance(units, ParticlePhaseSpaceUnits):
    #     raise TypeError('units must be an instance of articlePhaseSpace.ParticlePhaseSpaceUnits')

    column_names = {'x': 0,
                    'y': 0,
                    'z': 0,
                    'px:': 0,
                    'py': 0,
                    'Ek': 0,
                    'vx': 0,
                    'vy': 0,
                    'vz': 0,
                    'time': 0,
                    'mass': 0}
    return column_names

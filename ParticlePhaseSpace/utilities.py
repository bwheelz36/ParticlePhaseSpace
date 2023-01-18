import numpy as np
import ParticlePhaseSpace.__phase_space_config__ as ps_cfg
import ParticlePhaseSpace.__particle_config__ as particle_cfg

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


import numpy as np
import ParticlePhaseSpace.__config as cf


def _check_particle_types(pdg_codes):
    """
    check that the particle types in the phase space are known particle types
    """
    print('particle type check not implemented')

def get_rest_masses_from_pdg_codes(pdg_codes):
    """
    get an array of rest masses based on the particle types in self.data
    """
    _check_particle_types(pdg_codes)
    particle_rest_mass = np.zeros(len(pdg_codes))
    for particle_type in cf.particle_properties:
        ind = cf.particle_properties[particle_type]['pdg_code'] == pdg_codes
        particle_rest_mass[ind] = cf.particle_properties[particle_type]['rest_mass']
    return particle_rest_mass


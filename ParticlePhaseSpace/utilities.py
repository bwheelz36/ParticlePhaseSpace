import numpy as np
import ParticlePhaseSpace.__phase_space_config__ as ps_cfg
import ParticlePhaseSpace.__particle_config__ as particle_cfg
from ParticlePhaseSpace import ParticlePhaseSpaceUnits
from ParticlePhaseSpace import UnitSet

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
    get a list of rest masses associated with each particle in pdg_codes in MeV/c^2

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

def get_unit_conversions(old_units: UnitSet, new_units: UnitSet):
    """
    get the conversion factors between the old units and new units

    :param old_units: The original units
    :type old_units: UnitSet
    :param new_units: the new units
    :type new_units: UnitSet
    :return: conversion_factors: a dictionary with keys [length, energy, momentum, mass, velocity]
    """
    if not isinstance(old_units, UnitSet) or not isinstance(new_units, UnitSet):
        raise TypeError('this function only works with instnances of ParticlePhaseSpace.UnitSet')
    length_conversion = new_units.length.conversion / old_units.length.conversion
    momentum_conversion = new_units.momentum.conversion / old_units.momentum.conversion
    velocity_conversion = new_units.velocity.conversion / old_units.velocity.conversion
    energy_conversion = new_units.energy.conversion / old_units.energy.conversion
    mass_conversion = new_units.mass.conversion / old_units.mass.conversion
    time_conversion = new_units.time.conversion / old_units.time.conversion

    conversion_factors = {'length': length_conversion,
                          'momentum': momentum_conversion,
                          'energy': energy_conversion,
                          'mass': mass_conversion,
                          'time': time_conversion,
                          'velocity': velocity_conversion}
    return conversion_factors



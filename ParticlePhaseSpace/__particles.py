from scipy import constants
"""
This module simply hods a dictionary object which defines
the allowed particle types which can be imported

definitions should match
https://pdg.lbl.gov/2012/mcdata/mc_particle_id_contents.html

particles are described in the following format
{particle_name: {
"""

q = constants.elementary_charge

# calculate rest masses in eV
electron_rest_mass = constants.electron_mass * constants.c**2 / constants.elementary_charge  # in eV
proton_rest_mass = constants.proton_mass * constants.c**2 / constants.elementary_charge  # in eV
neutron_rest_mass = constants.neutron_mass * constants.c**2 / constants.elementary_charge  # in eV

particles = {
             'electrons': {'rest_mass': electron_rest_mass, 'charge': -q, 'pdg_code': int(11)},
             'protons':   {'rest_mass': proton_rest_mass,   'charge': q,  'pdg_code': int(2212)},
             'gammas':    {'rest_mass': 0,                  'charge': 0,  'pdg_code': int(22)},
             'neutrons':  {'rest_mass': neutron_rest_mass,  'charge': 0,  'pdg_code': int(2112)}
             }



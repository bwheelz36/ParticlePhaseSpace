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
electron_rest_mass = constants.electron_mass * constants.c ** 2 / constants.elementary_charge  # in eV
proton_rest_mass = constants.proton_mass * constants.c ** 2 / constants.elementary_charge  # in eV
neutron_rest_mass = constants.neutron_mass * constants.c ** 2 / constants.elementary_charge  # in eV

particles = {
    'electrons':
        {'rest_mass': electron_rest_mass,
         'charge': -q,
         'pdg_code': int(11)},
    'positrons':
        {'rest_mass': electron_rest_mass,
         'charge': q,
         'pdg_code': int(-11)},
    'protons':
        {'rest_mass': proton_rest_mass,
         'charge': q,
         'pdg_code': int(2212)},
    'gammas':
        {'rest_mass': 0,
         'charge': 0,
         'pdg_code': int(22)},
    'neutrons':
        {'rest_mass': neutron_rest_mass,
         'charge': 0,
         'pdg_code': int(2112)}
}

# set up aliases with pdf codes
particles[11] = particles['electrons']
particles[-11] = particles['positrons']
particles[2212] = particles['protons']
particles[22] = particles['gammas']
particles[2112] = particles['neutrons']

# check that there are the same number of aliases as entries
keys = list(particles.keys())
n_integer_keys = len(list(i for i in keys if isinstance(i, int)))
n_string_keys = len(list(i for i in keys if isinstance(i, str)))
if not n_integer_keys == n_string_keys:
    raise Exception('particle aliasing is not properly set up;'
                    'there must be the same number of integer and string keys')

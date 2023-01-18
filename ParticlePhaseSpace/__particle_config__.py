"""
Defines the handled particles.
more particles can be added here as required.
"""

from scipy import constants

q = constants.elementary_charge
# calculate rest masses in MeV
electron_rest_mass = 1e-6 * constants.electron_mass * constants.c ** 2 / constants.elementary_charge  # in MeV
proton_rest_mass = 1e-6 * constants.proton_mass * constants.c ** 2 / constants.elementary_charge  # in MeV
neutron_rest_mass = 1e-6 * constants.neutron_mass * constants.c ** 2 / constants.elementary_charge  # in MeV

particle_properties = {
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
particle_properties[11] = particle_properties['electrons']
particle_properties[-11] = particle_properties['positrons']
particle_properties[2212] = particle_properties['protons']
particle_properties[22] = particle_properties['gammas']
particle_properties[2112] = particle_properties['neutrons']

# check that there are the same number of aliases as entries
keys = list(particle_properties.keys())
n_integer_keys = len(list(i for i in keys if isinstance(i, int)))
n_string_keys = len(list(i for i in keys if isinstance(i, str)))
if not n_integer_keys == n_string_keys:
    raise Exception('particle aliasing is not properly set up;'
                    'there must be the same number of integer and string keys')

# add each string key as a 'name' field for ease of use with pdg codes:
for key in particle_properties.keys():
    if isinstance(key, str):
        particle_properties[key]['name'] = key

# finally, add any further string aliases:
particle_properties['photons'] = particle_properties['gammas']

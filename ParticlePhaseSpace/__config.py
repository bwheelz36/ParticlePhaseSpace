from scipy import constants

"""
This module simply hods a dictionary object which defines
the allowed particle types which can be imported

definitions should match
https://pdg.lbl.gov/2012/mcdata/mc_particle_id_contents.html

particle_properties are described in the following format
{particle_name: {
"""

required_columns = ['x [mm]',
                    'y [mm]',
                    'z [mm]',
                    'px',
                    'py',
                    'pz',
                    'particle type',
                    'weight',
                    'particle id',
                    'time [ps]']

allowed_columns = ['E [MeV]']

q = constants.elementary_charge
# calculate rest masses in MeV
electron_rest_mass = 1e-6 * constants.electron_mass * constants.c ** 2 / constants.elementary_charge  # in eV
proton_rest_mass = 1e-6 * constants.proton_mass * constants.c ** 2 / constants.elementary_charge  # in eV
neutron_rest_mass = 1e-6 * constants.neutron_mass * constants.c ** 2 / constants.elementary_charge  # in eV

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

# add each string key as a 'name' field for ease of use with pdg codes:
for key in particle_properties.keys():
    if isinstance(key, str):
        particle_properties[key]['name'] = key


# check that there are the same number of aliases as entries
keys = list(particle_properties.keys())
n_integer_keys = len(list(i for i in keys if isinstance(i, int)))
n_string_keys = len(list(i for i in keys if isinstance(i, str)))
if not n_integer_keys == n_string_keys:
    raise Exception('particle aliasing is not properly set up;'
                    'there must be the same number of integer and string keys')

# finally, you can add any further string aliases:
particle_properties['photons'] = particle_properties['gammas']

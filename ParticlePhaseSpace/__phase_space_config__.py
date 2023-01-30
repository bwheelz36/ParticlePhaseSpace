from .__unit_config__ import UnitSet

"""
This file defined the format of the phase space data.

- required_columns: must be created by a valid DataLoader; the code will exit if these do not exist
- allowed_columns: columns which are allowed but not required in the phase space data. More columns can be
   added here if required, but an appropriate method to calculate these values must be added to _ParticlePhaseSpace
   (see other 'fill_' methods)
- The existence of columns not defined here is not allowed and will cause the code to exit.
"""

required_columns = ['x',  # x position of each particle
                    'y',  # y position of each particle
                    'z',  # z position of each particle
                    'px',  # x momentum of each particle
                    'py',  # y momentum of each particle
                    'pz',  # z momentum of each particle
                    'particle type',  # [pdg code](https://pdg.lbl.gov/2012/mcdata/mc_particle_id_contents.html) for each particle
                    'weight',  # statistical weight of each particle (defaults to 1)
                    'particle id',  # id of each particle
                    'time']  # time each particle was scored

'''
The below defines all other quantities that can be calculated.
each allowed method must have a defined method inside _ParticlePhaseSpace, the format of the below dict is:

allowed_columns = {name of column: name of method to calculate it in _ParticlePhaseSpace}
automatic testing is applied to ensure all these methods can be calculated
'''

allowed_columns = {'Ek': 'fill_kinetic_E',  # Kinetic energy
                   'rest mass': 'fill_rest_mass',  # rest mass
                   'relativistic mass': 'fill_relativistic_mass',  # relatavistic mass
                   'gamma': 'fill_beta_and_gamma',   # lorentz factor
                   'beta_x': 'fill_beta_and_gamma',    # relatavistic beta vx/c
                   'beta_y': 'fill_beta_and_gamma',    # relatavistic beta vy/c
                   'beta_z': 'fill_beta_and_gamma',    # relatavistic beta vz/c
                   'beta_abs': 'fill_beta_and_gamma',    # relatavistic beta v/c
                   'vx': 'fill_velocity',  # x velocity
                   'vy': 'fill_velocity',  # y velocity
                   'vz': 'fill_velocity',  # z velocity
                   'Direction Cosine X': 'fill_direction_cosines',  # x cosine of momentum
                   'Direction Cosine Y': 'fill_direction_cosines',  # y cosine of momentum
                   'Direction Cosine Z': 'fill_direction_cosines',  # z cosine of momentum
                   'p_abs': 'fill_absolute_momentum'}

def _check_all_column_names(column_names: dict):

    allowed_column_names = required_columns + list(allowed_columns.keys())
    # first check that all allowed column names are being filled
    for column in allowed_column_names:
        try:
            test = column_names[column]
        except KeyError:
            raise AttributeError(f'Column {column} '
                           f'is defined in __phase_space_config but does not defined in get_all_column_names')

    # check that all allowed columns and defined columns same length
    assert len(allowed_column_names) == len(column_names)

def get_all_column_names(units: UnitSet):
    """
    return a dictionary of column names appropriate for the unit set defined in units

    :param units: instance of _UnitSet defining the units to generate column names for
    :type units: UnitSet
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
                    'p_abs': f'p_abs [{units.momentum.label}]',
                    'vx': f'vx [{units.velocity.label}]',
                    'vy': f'vy [{units.velocity.label}]',
                    'vz': f'vz [{units.velocity.label}]',
                    'time': f'time [{units.time.label}]',
                    'rest mass': f'rest mass [{units.mass.label}]',
                    'relativistic mass': f'relativistic mass [{units.mass.label}]',
                    'Ek': f'Ek [{units.energy.label}]',
                    'beta_x': 'beta_x',
                    'beta_y': 'beta_y',
                    'beta_z': 'beta_z',
                    'beta_abs': 'beta_abs',
                    'gamma': 'gamma',
                    'Direction Cosine X': 'Direction Cosine X',
                    'Direction Cosine Y': 'Direction Cosine Y',
                    'Direction Cosine Z': 'Direction Cosine Z',
                    'particle type': 'particle type [pdg_code]',
                    'weight': 'weight',
                    'particle id': 'particle id'}
    _check_all_column_names(column_names)
    return column_names

def get_required_column_names(units: UnitSet):

    all_column_names = get_all_column_names(units)
    required_column_names = []
    for column in required_columns:
        required_column_names.append(all_column_names[column])
    return required_column_names




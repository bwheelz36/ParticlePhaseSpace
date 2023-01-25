"""
This file defined the format of the phase space data.

- required_columns: must be created by a valid DataLoader; the code will exit if these do not exist
- allowed_columns: columns which are allowed but not required in the phase space data. More columns can be
   added here if required, but an appropriate method to calculate these values must be added to _ParticlePhaseSpace
   (see other 'fill_' methods)
- The existence of columns not defined here is not allowed and will cause the code to exit.
"""

required_columns = ['x [mm]',  # x position of each particle
                    'y [mm]',  # y position of each particle
                    'z [mm]',  # z position of each particle
                    'px [MeV/c]',  # x momentum of each particle
                    'py [MeV/c]',  # y momentum of each particle
                    'pz [MeV/c]',  # z momentum of each particle
                    'particle type [pdg_code]',  # [pdg code](https://pdg.lbl.gov/2012/mcdata/mc_particle_id_contents.html) for each particle
                    'weight',  # statistical weight of each particle (defaults to 1)
                    'particle id',  # id of each particle
                    'time [ps]']  # time each particle was scored

'''
The below defines all other quantities that can be calculated.
each allowed method must have a defined method inside _ParticlePhaseSpace, the format of the below dict is:

allowed_columns = {name of column: name of method to calculate it in _ParticlePhaseSpace}
automatic testing is applied to ensure all these methods can be calculated
'''

allowed_columns = {'Ek [MeV]': 'fill_kinetic_E',  # Kinetic energy
                   'rest mass [MeV/c^2]': 'fill_rest_mass',  # rest mass
                   'relativistic mass [MeV/c^2]': 'fill_relativistic_mass',  # relatavistic mass
                   'gamma': 'fill_beta_and_gamma',   # lorentz factor
                   'beta': 'fill_beta_and_gamma',    # relatavistic beta v/c
                   'vx [m/s]': 'fill_velocity',  # x velocity
                   'vy [m/s]': 'fill_velocity',  # y velocity
                   'vz [m/s]': 'fill_velocity',  # z velocity
                   'Direction Cosine X': 'fill_direction_cosines',  # x cosine of momentum
                   'Direction Cosine Y': 'fill_direction_cosines',  # y cosine of momentum
                   'Direction Cosine Z': 'fill_direction_cosines'}  # z cosine of momentum





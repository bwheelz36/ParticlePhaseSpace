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

allowed_columns = ['Ek [MeV]',  # Kinetic energy
                   'rest mass [MeV/c^2]',  # rest mass
                   'gamma',   # lorentz factor
                   'beta',    # relatavistic beta v/c
                   'vx [m/s]',  # x velocity
                   'vy [m/s]',  # y velocity
                   'vz [m/s]',  # z velocity
                   'Direction Cosine X',  # x cosine of momentum
                   'Direction Cosine Y',  # y cosine of momentum
                   'Direction Cosine Z']  # z cosine of momentum





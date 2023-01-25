# Phase Space Format

The allowed columns in the phase space data are defined in the `__phase_space_config.py` file, and are split into
two categories:

1. **Required columns** are, as the name implies, required and present at all times. These are the core data used 
  to represent the phase space; all other data can be derived from these. Each DataLoader must fill in these and only these columns to generate a valid import object.
2. **Allowed columns** can optionally be filled in using a `fill` method on a PhaseSpace object. For instance, to fill the allowed column `Ek [MeV]` you would use the `fill_kinetic_E` method. You are free to update the `allowed_columns` at any time, however in order for the code to pass testing you must:
  1. Write an associated `fill` method
  2. Update the test `test_all_allowed_columns_can_be_filled` inside `test_ParticlePhaseSpace`
  3. Update the "Allowed Columns" documentation below


## Required Columns

| Column name                        | Description                                                  |
| ---------------------------------- | ------------------------------------------------------------ |
| x [mm], y [mm], z [mm]             | The position of each particle in mm                          |
| px [MeV/c], py [MeV/c], pz [MeV/c] | the relativistic momentum of each particle in MeV/c          |
| particle type [pdg_code]           | The particle type, encoded as an integer [pdg code](https://pdg.lbl.gov/2012/mcdata/mc_particle_id_contents.html) |
| weight                             | the statistical weight of each particle. This is used to calculate weighted statistics. In many cases this will be 1 for all particles |
| particle id                        | each particle in the phase space must have a unique integer particle ID |
| time [ps]                          | the time the particle was recorded (in many cases this will be 0 for all particles) |

## Allowed Columns

| Column Name                                                | Description                                                                              | fill method              |
|------------------------------------------------------------|------------------------------------------------------------------------------------------| ------------------------ |
| Ek [MeV]                                                   | Kinetic energy in MeV                                                                    | `fill_kinetic_E`         |
| rest mass [MeV/c^2]                                        | Particle rest mass in Mev/c^2                                                            | `fill_rest_mass`         |
| 'relativistic mass [MeV/c^2]'                              | Particle mass in Mev/c^2                                                            | `fill_relativistic_mass`         |
| beta                                                       | [Beta factor](https://en.wikipedia.org/wiki/Lorentz_factor) of particle                  | `fill_beta_and_gamma`    |
| gamma                                                      | [Lorentz factor](https://en.wikipedia.org/wiki/Lorentz_factor) of particle               | `fill_beta_and_gamma`    |
| vx [m/s], vy [m/s], vz [m/s]                               | relativistic particle velocity in m/s                                                    | `fill_velocity`          |
| Direction Cosine X, Direction Cosine Y, Direction Cosine Z | [Direction Cosines](https://en.wikipedia.org/wiki/Direction_cosine) of particle momentum | `fill_direction_cosines` |

## Notes on units

In this code, I chose to only a single unit framework. This is because units are a frequent source
of confusion and error, so the simplest and safest approach seems to be to just support one unit
framework.

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
| 'relativistic mass [MeV/c^2]'                              | Particle mass in Mev/c^2                                                                 | `fill_relativistic_mass` |
| beta_x, beta_y, beta_z, beta_abs                           | [x Beta factor](https://en.wikipedia.org/wiki/Lorentz_factor) of particle                | `fill_beta_and_gamma`    |
| gamma                                                      | [Lorentz factor](https://en.wikipedia.org/wiki/Lorentz_factor) of particle               | `fill_beta_and_gamma`    |
| vx [m/s], vy [m/s], vz [m/s]                               | relativistic particle velocity in m/s                                                    | `fill_velocity`          |
| Direction Cosine X, Direction Cosine Y, Direction Cosine Z | [Direction Cosines](https://en.wikipedia.org/wiki/Direction_cosine) of particle momentum | `fill_direction_cosines` |

## Notes on units

In this code, I chose to use only a single unit framework. This is because units are a frequent source
of confusion and error, so the simplest and safest approach seems to be to just support one unit
framework. 
If there is need, we could update the plots such that different units can be specified - however my thoughts/preference at the
momentum is that only one set of units are used internally. 
The units should be quite clear from the column names above, but to avoid ambiguity:


| quantity | units   |
| -------- | ------- |
| distance | mm      |
| momentum | MeV/c   |
| Energy   | MeV     |
| mass     | MeV/c^2 |
| time     | ps      |
| velocity | m/s     |

## Reading in momentum

Position-Momentum are the fundamental quantities read in by the DataLoaders. Position is (hopefully) trivial, 
but for momentum we make the following notes that may help in converting different properties into momentum:

### If energy/ direction cosines are specified:

```python
from math import sqrt
P = sqrt(E**2 + E_0**2)
```

where
```python
E = E_k + E_0
```
E should be in units of MeV.

To calculate px, py, and pz  from this, you must have something like the momentum cosines.
```python
p_x = P*DirectionCosine_x
```
etc.

### If beta/ gamma specified

For charged particles, data may be specified in terms of beta/gamma. In that case:
```python
P = beta*gamma*rest_energy
p_x = beta_x*gamma*rest_energy
```
etc.
rest_energy should be specified in MeV, e.g. for electrons the value is 0.511 MeV

### If momentum is specified in SI units

To convert from kg.m/s to MeV/c
```python
P_ev_c = P_SI * (c/q)
P_MeV_c = P_SI * (c/q) * 1e-6
```
# Supported particles

The following particles are currently supported:

| particle name   | pdg_code | rest_mass   | charge      |
|-----------------|----------|-------------| ----------- |
| electrons       | 11       | 0.511 MeV   | -1.602-19 C |
| positrons       | -11      | 0.511 MeV   | 1.602-19 C  |
| gammas          | 22       | 0 MeV       | 0 C         |
| protons         | 2212     | 938.272 MeV | 1.602-19 C  |
| neutrons        | 2112     | 939.565 MeV | 0 C         |
| optical_photons | 0        | 0 MeV       | 0 C         |

## Adding new particles

Adding new particles is simple:p

- update the `particle_properties` dictionary in `ParticlePhaseSpace.__particle_config__.py`:
  ```python
  particle_properties = {
      'some_new_particle':
      	{'rest_mass': rest_mass_in_MeV/C^2,
           'charge': charge_in_coulombs,
           'pdg_code': pdg_code}
  }
  ```
- Below the dictionary definition, add an alias for the particle. This enables various parts of the code to operate using either particle names or pdg_codes:
  ```python
  particle_properties[pdg_code_new_particle] = particle_properties['some_new_particle']
  ```
- Update the documents above with the new particle so the tests pass

  
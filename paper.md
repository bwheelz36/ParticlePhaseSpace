---
title: 'ParticlePhaseSpace: A python package for streamlined import, analysis, and export of particle phase space data'
tags:
  - Python
  - accelerator physics
  - topas-mc
  - phase space
authors:
  - name: Brendan Whelan
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: Image-X Institute, School of Health Sciences, University of Sydney
   index: 1
date: 22 January 2023
bibliography: paper.bib
---

# Summary

Simulations of particle transport in the presence of various forces are widely used for the design and fine tuning of particle accelerators ranging in scope from the large hadron collider to the humble medical x-ray tube. A description of the positions and directions of an ensemble of particles is commonly both the input and output for these simulations, and is termed a phase space. For a given particle at a given time, 6 properties are required; for example, position-momentum: [x y z px py pz].  To extend this description to arbitrary particles at an arbitrary time, the particle species (e.g. electron, x-ray. etc.), time at which its properties were recorded must also be included. Finally, it is common in simulations to represent many particles with one 'macro particle'; in this case, the statistical weight of each particle in the data set must also be recorded. 

Phase space data is widely used in the simulation of particle accelerators to store results; in particular phase space data is commonly both the input and output of accelerator simulations. Unfortunately, there is no widely accepted format for phase space data, nor are there common open source libraries for handling this data. 

# Statement of need

Although the use of phase space data is well established, there is no consistent implementation of phase space data between different programs. To appreciate why this is an issue, table 1 summarises the main simulation tasks involved in simulation a medical linear accelerator @st_aubin_integrated_2010. Note that this table is intended to be demonstrative rather than exhaustive; even so, 9 different simulation codes are listed. Each of these programs has their own format for saving and loading phase space data. This means that getting these programs to 'speak' to each other is generally a substantial amount of work - in some cases, the bulk of the work in running simulations occurs in the input/output stages!

| Simulation task                                              | Code type                                 | Examples of suitable programs |
| ------------------------------------------------------------ | ----------------------------------------- | ----------------------------- |
| Electron gun simulation                                      | Electrostatic FEM, Particle-in-cell (PIC) | Opera, CST, Comsol, Egun      |
| Track particles through time varying fields, including self-fields | Particle-in-cell (PIC)                    | CST, Parmela, GPT             |
| Calculate particle creation/ destruction and calculate dose to medium | Monte Carlo                               | Geant4, Topas, Fluka, EGS     |
| Calculate thermal effects of energy deposited by particle beam | Finite Element or similar                 | Comsol, CST, ANSYS            |

`ParticlePhaseSpace` is a python package to handle the import, analysis, and export of particle phase space data in a well documented and extensible fashion. `ParticlePhaseSpace` provides:

- A standardised format/ unit set for phase space data as  described in TABLE

- The ability to calculate additional beam properties such as energy, lorentz factor, direction cosines from this data. This functionality has been designed to be easily extensible to new quantities in terms of documentation and testing.
- Tests ensuring that all of the above quantities are well documented
- Various methods for analysis and visualisation of particle positions, energy
- Methods to project the particles to new positions based on their initial momentum/position
- Calculation and visualiastion of the twiss parameters
- Extensible mechanisms through which phase space data can be both imported and exported from different programs. 
- Multi particle support; currently gammas, electrons, neutrons, protons and positrons are supported and the code has been set up for easy extension to arbitrary particles. 
- Ability to generate new PhaseSpace objects by adding, subtracting, or filtering existing PhaseSpace objects

The underlying data is stored in a pandas data frame with clearly defined allowed columns and units. Both the supported particles and data format have been designed to be extensible, and documentation exists demosntrating how to do this. A PhaseSpace object is provided with a large library of methods for visualising, manipulating, characterising and analysing phase space data. Importantly,  base classes for data import and export are provided, and examples demonstrating how to write new data loaders/ data exporters. Another code providing similar functionality is @lesnat_particle_2021; however while this code includes many excellent features it lacks detailed documentation, a test framework, and extension mechanisms.



| Column name                        | Description                                                  |
| ---------------------------------- | ------------------------------------------------------------ |
| x [mm], y [mm], z [mm]             | The position of each particle in mm                          |
| px [MeV/c], py [MeV/c], pz [MeV/c] | the relativistic momentum of each particle in MeV/c          |
| particle type [pdg_code]           | The particle type, encoded as an integer [pdg code](https://pdg.lbl.gov/2012/mcdata/mc_particle_id_contents.html) |
| weight                             | the statistical weight of each particle. This is used to  calculate weighted statistics. In many cases this will be 1 for all  particles |
| particle id                        | each particle in the phase space must have a unique integer particle ID |
| time [ps]                          | the time the particle was recorded (in many cases this will be 0 for all particles) |

# Acknowledgements

Brendan Whelan acknowledges funding support from the NHMRC

# References

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

In accelerator particle physics, a description of the positions and directions of an ensemble of particles is termed a phase space. For a given particle at a given time, 6 properties are required; for example, position-momentum: [x y z px py pz].  To extend this description to arbitrary particles at an arbitrary time, the particle species (e.g. electron, x-ray. etc.), time at which its properties were recorded must also be included. Finally, it is common in simulations to represent many particles with one 'macro particle'; in this case, the statistical weight of each particle in the dataset must also be recorded. 

Simulation frameworks are widely used for the design and fine tuning of particle accelerators ranging in scope from the large hadron collider to the humble x-ray tube. In these simulations, 

Phase space data is widely used in the simulation of particle accelerators to store results; in particular phase space data is commonly both the input and output of accelerator simulations. Unfortunately, there is no widely accepted format for phase space data, nor are there common open source libraries for handling this data. 

# Statement of need

`ParticlePhaseSpace` is a python package to handle the import, analysis, and export of particle phase space data in a well documented and extensible fashion. Although the use of phase space data is well established, there is no consistent implementation of phase space data between different programs. To appreciate why this is an issue, table 1 summarises the main simulation tasks involved in simulation a medical linear accelerator `@st_aubin_integrated_2010`. Note that this table is intended to be demonstrative rather than exhaustive; even so, 9 different simulation codes are listed. Each of these programs has their own format for saving and loading phase space data. This means that getting these programs to 'speak' to each other is generally a substantial amount of work - in some cases, the bulk of the work in running simulations occurs in the input/output stages! As such, substantial efforts must be made to enable these programs to 'speak' to each other.

| Simulation task                                              | Code type                                 | Examples of suitable programs |
| ------------------------------------------------------------ | ----------------------------------------- | ----------------------------- |
| Electron gun simulation                                      | Electrostatic FEM, Particle-on-cell (PIC) | Opera, CST, Comsol, Egun      |
| Track particles through time varying fields, including self-fields | Particle-in-cell (PIC)                    | CST, Parmela, GPT             |
| Calculate particle creation/ destruction and calculate dose to medium | Monte Carlo                               | Geant4, Topas, Fluka, EGS     |
| Calculate thermal effects of energy deposited by particle beam | Finite Element or similar                 | Comsol, CST, ANSYS            |

There have been some attempts to define a universal phase space format through the IAEA, however there are still regular issues with non-compliance of IAEA phase and it is widely acknowledged that the format is poorly defined. In any case, the IAEA format has not been supported by any of the commercial vendors listed above who continue to rely on their own data formats.

`ParticlePhaseSpace` aims to solve these issues by providing extensible mechanisms through which phase space data can be both imported and exported from different programs., as well as a library of methods for visualising, manipulating, characterising and analysing phase space data. The underlying data is stored in a pandas data frame with clearly defined allowed columns and units. Both the supported particles and data format have been designed to be extensible, and documentation exists demosntrating how to do this. A PhaseSpace object is provided with a large library of methods for visualising, manipulating, characterising and analysing phase space data. Importantly,  base classes for data import and export are provided, and examples demonstrating how to write new data loaders/ data exporters. Another code providing similar functionality is @lesnat_particle_2021; however while this code includes many excellent features it lacks detailed documentation, test cases, and extension mechanisms.

- PhaseSpace objects can be added and subtracted from each other
- Derive other physical quantities such as the lorentz factor and kinetic energy

# Acknowledgements

Brendan Whelan acknowledges funding support from the NHMRC

# References
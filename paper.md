---
title: 'ParticlePhaseSpace: A python package for streamlined import, analysis, and export of particle phase space data'
tags:
  - Python
  - accelerator physics
  - topas-mc
  - phase space
authors:
  - name: Brendan Whelan
    orcid: 0000-0002-2326-0927
    affiliation: 1
  - name: Leo Esnault
    orcid: 0000-0002-0795-9957
    affiliation: 2
  
affiliations:
 - name: Image-X Institute, School of Health Sciences, University of Sydney
   index: 1
 - name: Univ. Bordeaux-CNRS-CEA, Centre Lasers Intenses et Applications, UMR 5107, 33405 Talence, France 
   index: 2
date: 22 January 2023
bibliography: paper.bib
---

# Summary

In accelerator particle physics, a description of the positions and directions of an ensemble of particles is termed a phase space @wiedemann_particle_2015. To describe arbitrary particles at a given time, 6 properties are required; for example, position-momentum: `[x y z px py pz]`. To extend this description to arbitrary particles at an arbitrary time, the particle species (e.g. electron, x-ray. etc.), time at which its properties were recorded, and statistical weight (importance) of the particle must also be included. Phase space data is commonly both the input and output of particle accelerator simulations. Unfortunately, there is no widely accepted format for phase space data. 

# Statement of need

Although the use of phase space data is well established, there is no consistent implementation of phase space data between different programs @tessier_proposal_2021. To appreciate why this is an issue, one must understand that in a typical accelerator workflow, it is common to utilise several different programs to simulate different phases of particle transport, and for any given simulation task, there are many different simulation programs one can use @noauthor_accelerator_2022.  Each of these programs will typically utilize their own unique format for saving and loading phase space data. This means that getting these programs to 'speak' to each other is generally a substantial amount of work. In addition, the fragmented data format has led to fragmented analysis frameworks, with different fields and research groups tending to develop there own, often in-house, code for handling phase space data.  

![Code structure overview\label{figure 1}](paper_figure.png)



`ParticlePhaseSpace` aims to solve these issues by providing well documented, extensible mechanisms for the import and export of data in different formats, as well as a library of methods for visualizing, manipulating, characterizing and analyzing phase space data. The underlying data is stored in a pandas data frame with clearly defined allowed columns and units. The supported particles, phase space data format, and import/export mechanisms have all been designed to be extensible, with documentation demonstrating how to do this. The basic code structure is shown in \autoref{figure 1}. 


The main `PhaseSpace` object provides large library of methods for visualizing, manipulating, characterizing, plotting, and calculations using phase space data. This code is unit-aware; users can select from a variety of pre-defined unit sets at the data read in stage; units can also be set within the `PhaseSpace` object or at the data export stage. If the users unit set is not already defined, documentation [is provided](https://bwheelz36.github.io/ParticlePhaseSpace/units.html) on defining new unit sets.



Examples of publications using code **like** this can be found in @st_aubin_integrated_2010, .




- The ability to calculate additional beam properties such as energy, lorentz factor, direction cosines from this data. This functionality has been designed to be easily extensible to new quantities in terms of documentation and testing.
- Tests ensuring that all of the above quantities are well documented (lack of documentation on a newly defined quantity will cause the tests to fail)
- Extensible support for different unit sets
- Various methods for analysis and visualisation of particle positions, energy
- Methods to project the particles to new positions based on their initial momentum/position
- Calculation and visualisation of the twiss parameters
- Extensible mechanisms through which phase space data can be both imported and exported from different programs. 
- Multi particle support; currently gammas, electrons, neutrons, protons and positrons are supported and the code has been set up for easy extension to arbitrary particles. 
- Ability to generate new PhaseSpace objects by adding, subtracting, or filtering existing PhaseSpace objects
- Multiple unit sets, and extension mechanisms for building more unit sets

Other codes providing similar functionality include the p2sat code @lesnat_particle_2021 and postpic @skuschel_postpic_nodate. ParticlePhaseSpace builds off these codes by enabling multi-particle support in the same PhaseSpace object, extension mechanisms, a testing framework, continuous integration, and automatic code documentation.

# Acknowledgements

Brendan Whelan acknowledges funding support from the NHMRC

- Julia if figure help?

# References

---
title: 'Gala: A Python package for galactic dynamics'
tags:
  - Python
  - accelerator physics
  - topas-mc
  - phase space
authors:
  - name: Brendan Whelan
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Image-X Institute, School of Health Sciences, University of Sydney
   index: 1
date: 22 January 2023
bibliography: paper.bib
---

# Summary

```
The forces on stars, galaxies, and dark matter under external gravitational fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration).
```

In accelerator particle physics, it is necessary to describe ensembles of particles. For a given particle at a given time, 6 properties are sufficient to describe where it is and where it going in the absence of any forces. For example, position-momentum can be used: [x y z px py pz].  To extend this description to arbitrary particles at an arbitrary time, one must also include the particle species (e.g. electron, x-ray. etc.), the time at which its properties were recorded, and the statistical weight (importance) of the particle. The description of all particles in a particle beam (or statistically valid equivalent) is termed a particle phase-space. Phase Space data is integral for storing the results of a given experiment or simulation. However, there is no single standard for storing phase space data, meaning 

# Statement of need

`ParticlePhaseSpace` is a python package to handle the import, analysis, and export of particle phase space data in a well documented and extensible fashion. Although the use of phase space data is well established, there is no consistent implementation of phase space data between different programs. To appreciate why this is an issue, table 1 summarises the main simulation tasks involved in simulation a medical linear accelerator. Note that this table is intended to be demonstrative rather than exhaustive; even so, 9 different simulation codes are listed; each of these programs has their own format for saving and loading phase space data. This means that getting these programs to 'speak' to each other is generally a substantiual amount of work - in some cases, the bulk of the work in running simulations occurs in the input/output stages! Substantial efforts must be made to enable these programs to 'speak' to each other - unfortunately such efforts all too frequently remain buried on the hard drive of PhD students! 

| Simulation task                                              |                              | Examples of suitable programs |
| ------------------------------------------------------------ | ---------------------------- | ----------------------------- |
| Electron gun simulation                                      | Electrostatic FEM or similar | Opera, CST, Comsol, Egun      |
| Track particles through time varying fields, including self-fields | Particle-in-cell (PIC)       | CST                           |
| Calculate particle creation/ destruction and calculate dose to medium | Monte Carlo                  | Geant4, Topas, Fluka, EGS     |
| Calculate thermal effects of enrergy deposited by particle beam | Finite Element or similar    | Comsol, CST, ANSYS            |

Unfortunately, each of the codes listed above have their own phase space formats, and do not 'speak' well to each other. There have been some attempts to define a universal phase space format through the IAEA, however there are still regular issues with non-compliance of IAEA phase and it is widely acknowledged that the format is poorly defined. In any case, the IAEA format has not been supported by any of the commercial vendors listed above who continue to rely on their own data formats. 

`ParticlePhaseSpace` aims to solve these issues by providing extensible mechanisms through which phase space data can be both imported and exported from different programs., as well as a library of methods for visualising, manipulating, characterising and analysing phase space data. The underlying data is stored in a pandas data frame with clearly defined allowed columns and units. Both the supported particles and data format have been designed to be extensible, and documetnation exists demosntrating how to do this. A PhaseSpace object is provided with a large library of methods for visualising, manipulating, characterising and analysing phase space data. Importantly,  base classes for data import and export are provided, and examples demonstrating how to write new data loaders/ data exporters. 

- PhaseSpace objects can be added and subtracted from each other
- Derive other physical quantities such as the lorentz factor and kinetic energy

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
# ParticlePhaseSpace

![tests](https://github.com/bwheelz36/ParticlePhaseSpace/actions/workflows/run_tests.yml/badge.svg)![tests](https://github.com/bwheelz36/ParticlePhaseSpace/actions/workflows/build_docs.yml/badge.svg)[![codecov](https://codecov.io/gh/bwheelz36/ParticlePhaseSpace/branch/main/graph/badge.svg?token=T44KBJ7INR)](https://codecov.io/gh/bwheelz36/ParticlePhaseSpace)[![PyPI version](https://badge.fury.io/py/ParticlePhaseSpace.svg)](https://badge.fury.io/py/ParticlePhaseSpace)

Common library for dealing with particle phase spaces, revolving around the simple workflow of `import`, `analyse`, `export`. If you have a data format that we can't already work with, extension mechanisms are provided for writing new `DataLoaders` and `DataExporters`.

## Install and Requirements

To install: ```pip install ParticlePhaseSpace```

If you want to develop the code, there are some additional requirements, listed in `dev_requirements.txt`

## Usage and Documentation

- Detailed documentation is provided [here](https://bwheelz36.github.io/ParticlePhaseSpace/).
- The source code for the [worked examples](https://bwheelz36.github.io/ParticlePhaseSpace/examples.html) is inside the examples folder.
- For a list of the current data loaders, see [here](https://bwheelz36.github.io/ParticlePhaseSpace/code_docs.html#module-ParticlePhaseSpace.DataLoaders)


## Directory Structure

- **ParticlePhaseSpaace:** source code
- **examples:** source code for the [worked examples](https://bwheelz36.github.io/ParticlePhaseSpace/examples.html) provided in the docs
- **docsrc:** markdown/rst documentation.
- **tests:** tests which are run through github actions

## Contributions

Contributions in the form of pull requests are very welcome! 
Please use the 'issues' tab in this repository to report problems 
or request support

## Other packages for analysis of particle phase space data

Below is a (almost certainly incomplete) list of other packages I have come accross for analysis of particle phase space data:

- [p2sat](https://github.com/lesnat/p2sat)
- [postpic](https://github.com/skuschel/postpic)
- [openPMD-beamphysics](https://christophermayes.github.io/openPMD-beamphysics)
- [openPMD-viewer](https://github.com/openPMD/openPMD-viewer)

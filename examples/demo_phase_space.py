from ParticlePhaseSpace import DataLoaders
from ParticlePhaseSpace import PhaseSpace
from ParticlePhaseSpace import DataExporters
from pathlib import Path
import matplotlib as mpl

# mpl.rcParams['figure.dpi'] = 200

# load topas data
test_data_loc = Path(r'coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp').absolute()
ps_data = DataLoaders.LoadTopasData(test_data_loc)
PS = PhaseSpace(ps_data)

DataExporters.Topas_Exporter(PS, output_location=Path('.').absolute(), output_name='test')

ps_data = DataLoaders.LoadTopasData(Path(r'test.phsp'))
PS2 = PhaseSpace(ps_data)



# # print report ro the screen
# PS.report()
# # generate new phase spaces based on different particle species
# electrons_only = PS('electrons')
# electrons_only.print_twiss_parameters()
# # electrons_only.plot_particle_positions(weight_position_plot=True)
# gamma_only, positrons_only = PS(['gammas', 'positrons'])
# # we can add these back together using the + operator:
# all_particles = electrons_only + gamma_only + positrons_only
# # we cannot add phase space objects together when they contain the same particles
# # you have to change the particle IDs if you really want to do this.
#
# # we can also create a phase space by subtracting one set of particles from the rest:
# electrons_and_gamma = all_particles - positrons_only
# # if our data is large, we can downsample it for quicker analysis
# downsampled = PS.get_downsampled_phase_space(downsample_factor=10)

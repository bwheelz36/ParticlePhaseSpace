import sys
from pathlib import Path
this_file_loc = Path(__file__).parent
sys.path.insert(0, str(this_file_loc.parent))
import ParticlePhaseSpace.__particle_config__ as particle_config

def test_aliases():
    particles = particle_config.particle_properties

    for particle in particles:
        pdg_code = particles[particle]['pdg_code']
        assert particles[particle] == particles[pdg_code]

def test_particles_documented():

    with open(this_file_loc.parent / 'docsrc' / 'supported_particles.md') as f:
        particle_format_txt = f.readlines()
    supported_ind = particle_format_txt.index('# Supported particles\n')
    new_ind = particle_format_txt.index('## Adding new particles\n')
    required_txt = particle_format_txt[supported_ind: new_ind]
    for particle in particle_config.particle_properties:
        if isinstance(particle, int):
            continue
        if particle == 'photons':
            continue
        found_entry = []
        try:
            for line in required_txt:
                found_entry.append(particle in line)
            assert any(found_entry)
        except:
            raise(f'{particle} is not documented')

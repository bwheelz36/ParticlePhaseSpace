from pathlib import Path
import sys
import numpy as np
'''
For testing/ example purposes it's safest to manually append the path variable to ensure our 
package will always be found. This isn't necessary once we actually install the package 
because installation in python essentially means "copying the package to a place where it can be found"
'''
this_file_loc = Path(__file__)
sys.path.insert(0, str(this_file_loc.parent.parent))
# note: insert(0) means that the path is above is the FIRST place that python will look for imports
# sys.path.append means that the path above is the LAST place
from ParticlePhaseSpace import sine_wave_utilities as swu

def test_amplitude():
    x, y = swu.generate_sine_data(Amplitude=10)
    assert y.max() > 9.9  # can't guarantee max=10 because we have finite samples
    assert y.min() < -9.9

def test_range():
    x, y = swu.generate_sine_data(x_max=10, x_min=-8)
    assert x.max() == 10
    assert x.min() == -8

def test_n_samples():
    x, y = swu.generate_sine_data(n_samples=192)
    assert y.shape[0] == 192

def test_phase():
    x_0, y_0 = swu.generate_sine_data(Phase=0)
    x_1, y_1 = swu.generate_sine_data(Phase=1)
    # if phase works, at a minimum we should see that these two data sets are different...
    assert not np.equal(y_0, y_1).all()



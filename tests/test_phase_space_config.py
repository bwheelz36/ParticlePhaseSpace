"""
test that config is appropriately documented
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
this_file_loc = Path(__file__).parent
sys.path.insert(0, str(this_file_loc.parent))
from ParticlePhaseSpace import DataLoaders
from ParticlePhaseSpace import PhaseSpace
import ParticlePhaseSpace.__phase_space_config__ as ps_cfg
import ParticlePhaseSpace.__particle_config__ as particle_cfg

def test_columns_documented():

    with open(this_file_loc.parent / 'docsrc' / 'phase_space_format.md') as f:
        pps_format_txt = f.readlines()
    assert '## Required Columns\n' in pps_format_txt
    assert '## Allowed Columns\n' in pps_format_txt
    required_heading_ind = pps_format_txt.index('## Required Columns\n')
    allowed_heading_ind = pps_format_txt.index('## Allowed Columns\n')
    required_txt = pps_format_txt[required_heading_ind: allowed_heading_ind]
    for column in ps_cfg.required_columns:
        found_entry = []
        for line in required_txt:
            found_entry.append(column in line)
        assert any(found_entry)

    allowed_txt = pps_format_txt[allowed_heading_ind:]
    for column in list(ps_cfg.allowed_columns.keys()):
        found_entry = []
        for line in allowed_txt:
            found_entry.append(column in line)
        if not any(found_entry):
            raise Exception(f'required column {column} does not appear to be documented')
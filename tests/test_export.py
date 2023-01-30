import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
this_file_loc = Path(__file__).parent
sys.path.insert(0, str(this_file_loc.parent))
from ParticlePhaseSpace import DataExporters
from ParticlePhaseSpace import DataLoaders
from ParticlePhaseSpace import PhaseSpace
import ParticlePhaseSpace.__phase_space_config__ as ps_cfg
from ParticlePhaseSpace import ParticlePhaseSpaceUnits
import pytest

def test_topas_export():

    units = ParticlePhaseSpaceUnits()('mm_MeV')
    all_allowed_columns = ps_cfg.get_all_column_names(units)
    demo_data = pd.DataFrame(
                    {all_allowed_columns['x']: [0, 1, 2],
                     all_allowed_columns['y']: [0, 1, 2],
                     all_allowed_columns['z']: [0, 1, 2],
                     all_allowed_columns['px']: [1, 1, 2],
                     all_allowed_columns['py']: [1, 1, 2],
                     all_allowed_columns['pz']: [1, 1, 2],
                     all_allowed_columns['particle type']: [11, 11, 11],
                     all_allowed_columns['weight']: [0, 1, 2],
                     all_allowed_columns['particle id']: [0, 1, 2],
                     all_allowed_columns['time']: [0, 0, 0]})

    data = DataLoaders.Load_PandasData(demo_data)
    PS = PhaseSpace(data)
    # ok: can we export this data:
    DataExporters.Topas_Exporter(PS, output_location='.', output_name='test.phsp')
    # now check we can read it back in:
    data = DataLoaders.Load_TopasData('test.phsp')
    PS2 = PhaseSpace(data)
    PS.reset_phase_space()
    gah = PS.ps_data - PS2.ps_data
    assert all(gah.max() < 1e-5)

def test_export_with_different_units():
    """
    The topas data exporter expects to get data with units mm_MeV;
    this tests that it is able to convert different units sets
    :return:
    """
    units = ParticlePhaseSpaceUnits()('m_eV')
    all_allowed_columns = ps_cfg.get_all_column_names(units)
    demo_data = pd.DataFrame(
        {all_allowed_columns['x']: [0, 1, 2],
         all_allowed_columns['y']: [0, 1, 2],
         all_allowed_columns['z']: [0, 1, 2],
         all_allowed_columns['px']: [1, 1, 2],
         all_allowed_columns['py']: [1, 1, 2],
         all_allowed_columns['pz']: [1, 1, 2],
         all_allowed_columns['particle type']: [11, 11, 11],
         all_allowed_columns['weight']: [0, 1, 2],
         all_allowed_columns['particle id']: [0, 1, 2],
         all_allowed_columns['time']: [0, 0, 0]})

    data = DataLoaders.Load_PandasData(demo_data, units=units)
    PS = PhaseSpace(data)
    # ok: can we export this data:
    DataExporters.Topas_Exporter(PS, output_location='.', output_name='test.phsp')
from abc import ABC, abstractmethod

import pandas as pd
import topas2numpy as tp
import numpy as np

from .utilities import get_rest_masses_from_pdg_codes
import ParticlePhaseSpace.__config as cf

class _DataImportersBase(ABC):

    def __init__(self, input_data):
        self.data = pd.DataFrame(columns=cf.required_columns)
        self._input_data = input_data
        self._check_input_data()
        self._import_data()
        self._check_loaded_data()

    @abstractmethod
    def _import_data(self):
        """
        this function loads the data into the PS object
        :return:
        """
        pass

    @ abstractmethod
    def _check_input_data(self):
        """
        check that the data is what you think it is (read in specific)
        :return:
        """
        pass

    def _check_loaded_data(self):
        """
        check that the phase space data
        1. contains the required columns
        2. doesn't contain any non-allowed columns
        3. doesn't contain NaN
        4. "particle id" should be unique
        """
        # required columns present?
        for col_name in cf.required_columns:
            if not col_name in self.data.columns:
                raise AttributeError(f'invalid data input; required column "{col_name}" is missing')

        # all columns allowed?
        for col_name in self.data.columns:
            if not col_name in cf.required_columns:
                raise AttributeError(f'non allowed column "{col_name}" in data.')

        # are NaNs present?
        if self.data.isnull().values.any():
            raise AttributeError(f'input data may not contain NaNs')

        # is every particle ID unique?
        if not len(self.data['particle id'].unique()) == len(self.data['particle id']):
            raise AttributeError('you have attempted to create a data set with non'
                                 'unique "particle id" fields, which is not allwoed')


class LoadTopasData(_DataImportersBase):

    def _import_data(self):
        """
        Read in topas  data
        assumption is that this is in cm and MeV

        this has to be tested for particle travelling in the x and y directions since topas seems to be quite confused
        about this...
        """
        topas_phase_space = tp.read_ntuple(self._input_data)
        ParticleTypes = topas_phase_space['Particle Type (in PDG Format)']
        self.data['particle type'] = ParticleTypes.astype(int)
        self.data['x [mm]'] = topas_phase_space['Position X [cm]'] * 1e1
        self.data['y [mm]'] = topas_phase_space['Position Y [cm]'] * 1e1
        self.data['z [mm]'] = topas_phase_space['Position Z [cm]'] * 1e1
        self.data['weight'] = topas_phase_space['Weight']
        self.data['particle id'] = np.arange(len(self.data))  # may want to replace with track ID if available?
        self.data['time [ps]'] = 0  # may want to replace with time feature if available?
        # figure out the momentums:
        ParticleDir = topas_phase_space['Flag to tell if Third Direction Cosine is Negative (1 means true)']
        DirCosineX = topas_phase_space['Direction Cosine X']
        DirCosineY = topas_phase_space['Direction Cosine Y']
        E = topas_phase_space['Energy [MeV]']
        rest_masses = get_rest_masses_from_pdg_codes(self.data['particle type'])
        P = np.sqrt(E ** 2 + rest_masses ** 2)
        self.TOT_E = np.sqrt(P ** 2 + rest_masses ** 2)
        self.data['px'] = np.multiply(P, DirCosineX)
        self.data['py'] = np.multiply(P, DirCosineY)
        temp = P ** 2 - self.data['px'] ** 2 - self.data['py'] ** 2
        ParticleDir = [1 if elem else -1 for elem in ParticleDir]
        self.data['pz'] = np.multiply(np.sqrt(temp), ParticleDir)

    def _check_input_data(self):
        """
        - is file
        - is valid extension
        - has valid header
        - what does topas2numpy already do?
        """
        print('no topas data check implemented')


class LoadPandasData(_DataImportersBase):
    """
    loads in pandas data; provides a general purpose interface for
    those who do not wish to write a specific data loader for their application
    """

    def _import_data(self):
        self.data = self._input_data

    def _check_input_data(self):
        """
        is pandas instance
        has required columns (probably checked elsewhere actually)
        """
        pass
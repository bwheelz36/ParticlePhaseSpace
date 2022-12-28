from abc import ABC, abstractmethod
import topas2numpy as tp
import numpy as np

class DataLoaders():

    def __init__(self):

        self.TopasDataLoader = _TopasImporter()

class _DataImportersBase(ABC):

    def __init__(self):

        # check that PhaseSpaceObject is correct
        # self.import_data()
        pass

    @abstractmethod
    def __call__(self, input_data):
        pass

    @abstractmethod
    def import_data(self):
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
        print('no data check implemented yet')

class _TopasImporter(_DataImportersBase):

    def __call__(self, input_data):

        self._input_data = input_data
        self._check_input_data()
        self.import_data()
        self._check_loaded_data()

    def import_data(self):
        """
        Read in topas  data
        assumption is that this is in cm and MeV
        """

        PhaseSpace = tp.read_ntuple(self._input_data)
        ParticleTypes = PhaseSpace['Particle Type (in PDG Format)']
        ParticleTypes = ParticleTypes.astype(int)
        ParticleDir = PhaseSpace['Flag to tell if Third Direction Cosine is Negative (1 means true)']

        self.x = PhaseSpace['Position X [cm]'][Ind] * 1e1
        self.y = PhaseSpace['Position Y [cm]'][Ind] * 1e1
        self.z = PhaseSpace['Position Z [cm]'][Ind] * 1e1
        self.DirCosineX = PhaseSpace['Direction Cosine X'][Ind]
        self.DirCosineY = PhaseSpace['Direction Cosine Y'][Ind]
        self.E = PhaseSpace['Energy [MeV]'][Ind]
        self.weight = PhaseSpace['Weight'][Ind]

        # figure out the momentums:
        self.__CosinesToMom()

        self.x = np.delete(self.x, ind)
        self.y = np.delete(self.y, ind)
        self.z = np.delete(self.z, ind)
        self.px = np.delete(self.px, ind)
        self.py = np.delete(self.py, ind)
        self.pz = np.delete(self.pz, ind)
        self.E = np.delete(self.E, ind)
        self.TOT_E = np.delete(self.TOT_E, ind)

    def _check_input_data(self):
        print('no data check implemented')

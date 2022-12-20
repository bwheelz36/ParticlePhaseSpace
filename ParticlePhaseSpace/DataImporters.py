from abc import ABC, abstractmethod
from ParticlePhaseSpace.ParticlePhaseSpace import ParticlePhaseSpace
import topas2numpy as tp


class _DataImportersBase(ABC):

    def __init__(self, PhaseSpaceObject):

        # check that PhaseSpaceObject is correct
        self._PhaseSpaceObject = PhaseSpaceObject
        self._check_phase_space_object()
        self.import_data()

    def _check_phase_space_object(self):
        """
        checks that the phase space object is what we think it is!
        :return:
        """

        if not isinstance(self._PhaseSpaceObject, ParticlePhaseSpace):
            raise AttributeError('Data importers must be instantiated with a valid ParticlePhaseSpace object')

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

class TopasImporter(_DataImportersBase):



    def import_data(self):
        """
        Read in topas  data
        assumption is that this is in cm and MeV
        """

        PhaseSpace = tp.read_ntuple(self.Data)
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

        # if any values of pz == 0 exist, remove them with warning:
        if np.any(self.pz == 0):
            ind = self.pz == 0
            logging.warning(
                f'\nIn read in of topas data, removing {np.count_nonzero(ind)} of {ind.shape[0]} values where pz ==0. '
                f'\nWhile this is not necesarily an error,it means electrons are going completely sideways.'
                f'\nIt also makes transverse emittance calcs difficult, so im just going to delete those entries.'
                f'\nIf this is happening a lot I need to find a better solution\n')

            self.x = np.delete(self.x, ind)
            self.y = np.delete(self.y, ind)
            self.z = np.delete(self.z, ind)
            self.px = np.delete(self.px, ind)
            self.py = np.delete(self.py, ind)
            self.pz = np.delete(self.pz, ind)
            self.E = np.delete(self.E, ind)
            self.TOT_E = np.delete(self.TOT_E, ind)



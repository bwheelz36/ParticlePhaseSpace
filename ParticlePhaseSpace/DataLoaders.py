from abc import ABC, abstractmethod
from .__particles import particles
import pandas as pd
import topas2numpy as tp
import numpy as np


class _DataImportersBase(ABC):

    def __init__(self, input_data):

        self._required_column_names = ['x [mm]',
                                       'y [mm]',
                                       'z [mm]',
                                       'px',
                                       'py',
                                       'pz',
                                       'particle type']
        self._allowed_column_names = ['E [MeV]',
                                      'gamma',
                                      'weight',
                                      'particle id']
        self._data = pd.DataFrame(columns=self._required_column_names)
        self._input_data = input_data
        self._check_input_data()
        self.import_data()
        self._check_loaded_data()

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

    def _return_data(self):
        return self._data

    def _check_loaded_data(self):
        print('no data check implemented yet')

    def _momentum_to_velocity(self):
        """
        I think that I may need to define the cosines in terms of velocity, and not in terms of momentum
        as I have been doing.
        I'm also not totally sure that i'm calculating these correctly....
        """
        self.vx = np.divide(self.px, (self.Gamma * self._me_MeV))
        self.vy = np.divide(self.py, (self.Gamma * self._me_MeV))
        self.vz = np.divide(self.pz, (self.Gamma * self._me_MeV))

    def _check_particle_types(self):
        """
        check that the particle types in the phase space are known particle types
        """
        print('not implemented')

    def _get_rest_masses_from_pdg_codes(self):
        """
        get an array of rest masses based on the particle types in self._data
        """
        self._check_particle_types()
        self._particle_rest_mass = np.zeros(len(self._data))
        for particle_type in particles:
            ind = particles[particle_type]['pdg_code'] == self._data['particle type']
            self._particle_rest_mass[ind] = particles[particle_type]['rest_mass']



    def _cosines_to_mom(self, DirCosineX, DirCosineY, ParticleDirection):
        """
        Internal function to convert direction cosines and energy back into momentum
        """
        # first calculte total momentum from total energy:

        P = np.sqrt(self._data['E [MeV]'] ** 2 + self._particle_rest_mass ** 2)
        self.TOT_E = np.sqrt(P ** 2 + self._particle_rest_mass ** 2)
        px = np.multiply(P, DirCosineX)
        py = np.multiply(P, DirCosineY)
        pz = P ** 2 - px ** 2 - py ** 2

        return px, py, pz

class LoadTopasData(_DataImportersBase):

    def import_data(self):
        """
        Read in topas  data
        assumption is that this is in cm and MeV
        """
        PhaseSpace = tp.read_ntuple(self._input_data)
        ParticleTypes = PhaseSpace['Particle Type (in PDG Format)']
        self._data['particle type'] = ParticleTypes.astype(int)

        self._data['x [mm]'] = PhaseSpace['Position X [cm]'] * 1e1
        self._data['y [mm]'] = PhaseSpace['Position Y [cm]'] * 1e1
        self._data['z [mm]'] = PhaseSpace['Position Z [cm]'] * 1e1
        ParticleDir = PhaseSpace['Flag to tell if Third Direction Cosine is Negative (1 means true)']
        DirCosineX = PhaseSpace['Direction Cosine X']
        DirCosineY = PhaseSpace['Direction Cosine Y']
        self._data['E [MeV]'] = PhaseSpace['Energy [MeV]']
        self._data['weight'] = PhaseSpace['Weight']

        # figure out the momentums:
        self._get_rest_masses_from_pdg_codes()
        self._data['px'],\
        self._data['py'],\
        self._data['pz'] = self._cosines_to_mom(DirCosineX, DirCosineY, ParticleDir)

    def _check_input_data(self):
        """
        - is file
        - is valid extension
        - has valid header
        - what does topas2numpy already do?
        """
        print('no data check implemented')

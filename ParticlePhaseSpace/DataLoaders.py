from abc import ABC, abstractmethod
import pandas as pd
import topas2numpy as tp
import numpy as np
from pathlib import Path
from .utilities import get_rest_masses_from_pdg_codes
import ParticlePhaseSpace.__phase_space_config__ as ps_cfg
import ParticlePhaseSpace.__particle_config__ as particle_cfg
from ParticlePhaseSpace import UnitSet
import warnings
from ParticlePhaseSpace import ParticlePhaseSpaceUnits

units=ParticlePhaseSpaceUnits()

class _DataLoadersBase(ABC):
    """
    DataLoader Abstract Base Class.
    Inherited by new instances of DataLoaders
    """

    def __init__(self, input_data, particle_type=None, units=units('mm_MeV')):
        self.data = pd.DataFrame()
        if not isinstance(units, UnitSet):
            raise TypeError('units must be an instance of articlePhaseSpace.__unit_config__._UnitSet.'
                            'UnitSets are accessed through the ParticlePhaseSpaceUnits class')
        self._units = units
        self._columns = ps_cfg.get_all_column_names(self._units)


        if particle_type:
            if not isinstance(particle_type, str):
                allowed_particles = [el for el in list(particle_cfg.particle_properties.keys()) if isinstance(el, str)]
                try:
                    particle_type = particle_cfg.particle_properties[particle_type]['name']
                except KeyError:
                    raise Exception(f'unknown particle type: {particle_type}.'
                                    f'allowed particles are {allowed_particles}')
                if not particle_type in allowed_particles:
                    raise Exception(f'unknown particle type: {particle_type}.'
                                f'allowed particles are {allowed_particles}')
        self._particle_type = particle_type

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
        required_columns = ps_cfg.get_required_column_names(self._units)
        for col_name in required_columns:
            if not col_name in self.data.columns:
                raise AttributeError(f'invalid data input; required column "{col_name}" is missing')

        # all columns allowed?
        for col_name in self.data.columns:
            if not col_name in required_columns:
                raise AttributeError(f'non allowed column "{col_name}" in data.')

        # are NaNs present?
        if self.data.isnull().values.any():
            raise AttributeError(f'input data may not contain NaNs')

        tot_mom = np.sqrt(self.data[self._columns['px']]**2 + self.data[self._columns['py']]**2 + self.data[self._columns['pz']]**2)
        if not np.min(tot_mom)>0:
            raise Exception('particles with zero absolute momentum make no sense')

        # is every particle ID unique?
        if not len(self.data[self._columns['particle id']].unique()) == len(self.data[self._columns['particle id']]):
            raise Exception('you have attempted to create a data set with non'
                                 'unique "particle id" fields, which is not allowed')

        #all pdg codes valid?
        get_rest_masses_from_pdg_codes(self.data['particle type [pdg_code]'])

    def _check_energy_consistency(self, Ek):
        """
        for data formats that specify kinetic energy, this can be called at the end
        of _import data to check that the momentums in self.data give rise to the same kinetic
        energy as specified in the input data

        :param Ek:
        :return:
        """
        if not hasattr(self,'_rest_masses'):
            self._rest_masses = get_rest_masses_from_pdg_codes(self.data['particle type [pdg_code]'])
        Totm = np.sqrt((self.data[self._columns['px']] ** 2 + self.data[self._columns['py']] ** 2 + self.data[self._columns['pz']] ** 2))
        self.TOT_E = np.sqrt(Totm ** 2 + self._rest_masses ** 2)
        Ek_internal = np.subtract(self.TOT_E, self._rest_masses)

        E_error = max(Ek - Ek_internal)
        if E_error > .01:  # .01 MeV is an aribitrary cut off
            raise Exception('Energy check failed: read in of data may be incorrect')


class Load_TopasData(_DataLoadersBase):
    """
    DataLoader for `Topas <https://topas.readthedocs.io/en/latest/>`_ data.
    This data loader will read in both ascii and binary topas phase space (phsp) files.
    At present, we do not handle time or particle-id fields which may or may not be present in topas data.
    Behind the scenes, it relies on `topas2numpy <https://github.com/davidchall/topas2numpy>`_::

        from ParticlePhaseSpace import DataLoaders
        from ParticlePhaseSpace import PhaseSpace
        from pathlib import Path

        data_loc = Path(r'../tests/test_data/coll_PhaseSpace_xAng_0.00_yAng_0.00_angular_error_0.0.phsp')

        data = DataLoaders.Load_TopasData(data_loc)
        PS = PhaseSpace(data)
    """

    def _import_data(self):
        """
        Read in topas  data
        This has been extensively tested for data travelling the z direction, but not so much in the x and y directions.
        since topas uses the direction cosines to define directions, I would be cautious about these other cases
        """
        topas_phase_space = tp.read_ntuple(self._input_data)
        ParticleTypes = topas_phase_space['Particle Type (in PDG Format)']
        self.data[self._columns['particle type']] = ParticleTypes.astype(int)
        self.data[self._columns['x']] = topas_phase_space['Position X [cm]'] * 1e1
        self.data[self._columns['y']] = topas_phase_space['Position Y [cm]'] * 1e1
        self.data[self._columns['z']] = topas_phase_space['Position Z [cm]'] * 1e1
        self.data[self._columns['weight']] = topas_phase_space['Weight']
        self.data[self._columns['particle id']] = np.arange(len(self.data))  # may want to replace with track ID if available?
        self.data[self._columns['time']] = 0  # may want to replace with time feature if available?
        # figure out the momentums:
        ParticleDir = topas_phase_space['Flag to tell if Third Direction Cosine is Negative (1 means true)']
        DirCosineX = topas_phase_space['Direction Cosine X']
        DirCosineY = topas_phase_space['Direction Cosine Y']
        E = topas_phase_space['Energy [MeV]']
        self._rest_masses = get_rest_masses_from_pdg_codes(self.data['particle type [pdg_code]'])
        P = np.sqrt((E + self._rest_masses) ** 2 - self._rest_masses ** 2)
        self.data[self._columns['px']] = np.multiply(P, DirCosineX)
        self.data[self._columns['py']] = np.multiply(P, DirCosineY)
        temp = P ** 2 - self.data[self._columns['px']] ** 2 - self.data[self._columns['py']] ** 2
        ParticleDir = [-1 if elem else 1 for elem in ParticleDir]
        self.data[self._columns['pz']] = np.multiply(np.sqrt(temp), ParticleDir)
        self._check_energy_consistency(Ek=E)

    def _check_input_data(self):
        """
        In this case, just check that the file exists.
        The rest of the checks are handles inside topas2nupy
        """
        if not Path(self._input_data).is_file():
            raise FileNotFoundError(f'input data file {self._import_data()} does not exist')
        if not Path(self._input_data).suffix == '.phsp':
            raise Exception('The topas data loader reads in files of extension *.phsp')
        if self._particle_type:
            warnings.warn('particle type is ignored in topas read in')


class Load_PandasData(_DataLoadersBase):
    """
    loads in pandas data of the format. This is used internally by ParticlePhaseSpace, and can also be used
    externally in cases where it is not desired to write a dedicated new data loader::

        from ParticlePhaseSpace import DataLoaders
        import pandas as pd

        demo_data = pd.DataFrame(
            {'x [mm]': [0, 1, 2],
             'y [mm]': [0, 1, 2],
             'z [mm]': [0, 1, 2],
             'px [MeV/c]': [0, 1, 2],
             'py [MeV/c]': [0, 1, 2],
             'pz [MeV/c]': [0, 1, 2],
             'particle type [pdg_code]': [11, 11, 11],
             'weight': [0, 1, 2],
             'particle id': [0, 1, 2],
             'time [ps]': [0, 1, 2]})

        data = DataLoaders.Load_PandasData(demo_data)
    """

    def _import_data(self):

        self.data = self._input_data
        #         Note that the format of the data is checked by the base class,
        #         so no additional checks are required here

    def _check_input_data(self):
        """
        is pandas instance
        """
        assert isinstance(self._input_data, pd.DataFrame)

        if self._particle_type:
            raise AttributeError('particle_type should not be specified for pandas import')


class Load_TibarayData(_DataLoadersBase):
    """
    Load ASCII data from tibaray of format
    `x y z rxy Bx By Bz G t m q nmacro rmacro ID`::

        data_loc = Path(r'../tests/test_data/tibaray_test.dat')
        data = DataLoaders.Load_TibarayData(data_loc, particle_type=11)
        PS = PhaseSpace(data)
    """

    def _check_input_data(self):
        if not Path(self._input_data).is_file():
            raise FileNotFoundError(f'input data file {self._import_data()} does not exist')
        if not self._particle_type:
            raise Exception('particle_type must be specified when readin tibaray data')
        with open(self._input_data) as f:
            first_line = f.readline()
            if not first_line == 'x y z rxy Bx By Bz G t m q nmacro rmacro ID \n':
                warnings.warn('first line of tibaray data does not look as expected, proceed with caution')

    def _import_data(self):
        Data = np.loadtxt(self._input_data, skiprows=1)
        self.data[self._columns['x']] = Data[:, 0] * 1e3  # mm to m
        self.data[self._columns['y']] = Data[:, 1] * 1e3
        self.data[self._columns['z']] = Data[:, 2] * 1e3
        Bx = Data[:, 4]
        By = Data[:, 5]
        Bz = Data[:, 6]
        Gamma = Data[:, 7]
        self.data[self._columns['time']] = Data[:, 8] * 1e9
        m = Data[:, 9]
        q = Data[:, 10]
        self.data[self._columns['weight']] = Data[:, 11]
        rmacro = Data[:, 12]
        self.data[self._columns['particle id']] = Data[:, 13]
        self.data[self._columns['particle type']] = particle_cfg.particle_properties[self._particle_type]['pdg_code']

        self.data[self._columns['px']] = np.multiply(Bx, Gamma) * particle_cfg.particle_properties[self._particle_type]['rest_mass']
        self.data[self._columns['py']] = np.multiply(By, Gamma) * particle_cfg.particle_properties[self._particle_type]['rest_mass']
        self.data[self._columns['pz']] = np.multiply(Bz, Gamma) * particle_cfg.particle_properties[self._particle_type]['rest_mass']


class NewDataLoader(_DataLoadersBase):

    def _import_data(self):
        Data = np.loadtxt(self._input_data, skiprows=1)
        self.data['x [mm]'] = Data[:, 0]
        self.data['y [mm]'] = Data[:, 1]
        self.data['z [mm]'] = Data[:, 2]
        self.data['px [MeV/c]'] = Data[:, 3]
        self.data['py [MeV/c]'] = Data[:, 4]
        self.data['pz [MeV/c]'] = Data[:, 5]
        self.data['particle type [pdg_code]'] = particle_cfg.particle_properties[self._particle_type]['name']
        # we also need to fill in weight, particle id, and time; since none of these are specified we just use all
        # ones for weight, 1,2,3... for particle id, and all zeros for time:
        self.data['weight'] = np.ones(Data.shape[0])
        self.data['particle id'] = np.arange(len(self.data))

        self.data['time [ps]'] = 0  # may want to replace with time feature if available?

        # because we have momentum and energy, we can double check that our momentum to energy conversion is
        # consisten with the values in the phase space:
        E = Data[:, 6]
        self._check_energy_consistency(Ek=E)

    def _check_input_data(self):
        # is the input a file?
        if not Path(self._input_data).is_file():
            raise FileNotFoundError(f'input data file {self._import_data()} does not exist')
        # does it have the right extension?
        if not Path(self._input_data).suffix == '.dat':
            raise Exception('This data loaders requires a *.dat file')
        # the header is on the first line; does it look correct?
        with open(self._input_data) as f:
            first_line = f.readline()
            if not first_line == 'x (mm)\ty (mm)\tz (mm)\tpx (MeV/c)\tpy (MeV/c)\tpz (MeV/c)\tE (MeV)\n':
                raise Exception('file header does not look correct')
        if not self._particle_type:
            raise Exception('this data loader requires particle_type to be specified')
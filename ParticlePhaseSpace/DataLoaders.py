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
import re

units=ParticlePhaseSpaceUnits()

class _DataLoadersBase(ABC):
    """
    DataLoader Abstract Base Class.
    Inherited by new instances of DataLoaders

    :param input_data: location of file to read, or data to read
    :param particle_type: optional parameter if phase space format does not specify particle.
        particle type is a string matching a particle name from particle config
    :param units:  optionally specify units by passing a unit set
    """

    def __init__(self, input_data: (str, Path), particle_type:(str, None)=None, units:UnitSet=units('mm_MeV')):

        self.data = pd.DataFrame()
        if not isinstance(units, UnitSet):
            raise TypeError('units must be an instance of articlePhaseSpace.__unit_config__._UnitSet.'
                            'UnitSets are accessed through the ParticlePhaseSpaceUnits class')
        self._units = units
        self._columns = ps_cfg.get_all_column_names(self._units)
        self._required_columns = ps_cfg.get_required_column_names(self._units)
        self._energy_consistency_check_cutoff = .001 * self._units.energy.conversion # in cases where it is possible to check energy/momentum consistency,
        # discrepencies greater than this will raise an error
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

        for col_name in self._required_columns:
            if not col_name in self.data.columns:
                raise AttributeError(f'invalid data input; required column "{col_name}" is missing')

        # all columns allowed?
        for col_name in self.data.columns:
            if not col_name in self._required_columns:
                raise AttributeError(f'non allowed column "{col_name}" in data.')

        # are NaNs present?
        if self.data.isnull().values.any():
            NaN_cols = self.data.columns[self.data.isna().any()].tolist()
            raise AttributeError(f'input data may not contain NaNs; the following columns contain NaN:'
                                 f'\n{NaN_cols}')

        tot_mom = np.sqrt(self.data[self._columns['px']]**2 + self.data[self._columns['py']]**2 + self.data[self._columns['pz']]**2)
        if not np.min(tot_mom) > 0:
            ind = tot_mom <= 0
            warnings.warn(f'{np.count_nonzero(ind)} particles have zero absolute momentum; this makes no sense and they will be removed')

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

        :param Ek: existing value to check against
        :return:
        """
        if not hasattr(self,'_rest_masses'):
            self._rest_masses = get_rest_masses_from_pdg_codes(self.data['particle type [pdg_code]'])
        Totm = np.sqrt((self.data[self._columns['px']] ** 2 + self.data[self._columns['py']] ** 2 + self.data[self._columns['pz']] ** 2))
        self.TOT_E = np.sqrt(Totm ** 2 + self._rest_masses ** 2)
        Ek_internal = np.subtract(self.TOT_E, self._rest_masses)

        E_error = max(Ek - Ek_internal)
        if E_error > self._energy_consistency_check_cutoff:  # .01 MeV is an aribitrary cut off
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
        _negative_temp_ind = temp < 0
        if any(_negative_temp_ind):
            # this should never happen, but does occur when pz is essentially 0. we will attempt to resolve it here.
            negative_locations = np.where(_negative_temp_ind)[0]
            n_negative_locations = np.count_nonzero(_negative_temp_ind)
            momentum_precision_factor = 1e-3
            for location in negative_locations:
                relative_difference = np.divide(np.sqrt(abs(temp[location])), P[location])
                if relative_difference < momentum_precision_factor:
                    temp[location] = 0
                else:
                    raise Exception(f'failed to calculate momentums from topas data. Possible solution is to increase'
                                    f'the value of momentum_precision_factor, currently set to {momentum_precision_factor: 1.2e}'
                                    f'and failed data has value {relative_difference: 1.2e}')
            warnings.warn(f'{n_negative_locations: d} entries returned invalid pz values and were set to zero.')

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
        # make sure column names match the input units

        column_names = ps_cfg.get_required_column_names(self._units)
        for existing_col in self.data.columns:
            if not existing_col in column_names:
                raise Exception(f'the column names in the input pandas data are not consistent with the defined unit set:'
                                f'{self._units.label}')

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
            
            if first_line == 'x y z rxy Bx By Bz G t m q nmacro rmacro ID \n':
                self._skiprows = 1  # update below if different formay
            elif first_line == '@logo            B&M-General Particle Tracer\n':
                self._skiprows = 5
            else:
                warnings.warn('first line of tibaray data does not look as expected, proceed with caution')
                self._skiprows = 1  # take a guess

    def _import_data(self):
        Data = np.loadtxt(self._input_data, skiprows=self._skiprows)
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


class Load_p2sat_txt(_DataLoadersBase):
    """
    Adapted from the `p2sat <https://github.com/lesnat/p2sat/blob/master/p2sat/datasets/_LoadPhaseSpace.py>`_
    'txt' loader; loads csv data of format
    `# weight          x (um)          y (um)          z (um)          px (MeV/c)      py (MeV/c)      pz (MeV/c)      t (fs)`
    Note that we use a hard coded seperator value ",".
    ::

        available_units = ParticlePhaseSpaceUnits()
        data_url = 'https://raw.githubusercontent.com/lesnat/p2sat/master/examples/ExamplePhaseSpace.csv'
        file_name = 'p2sat_txt_test.csv'
        request.urlretrieve(data_url, file_name)
        # read in
        ps_data = DataLoaders.Load_p2sat_txt(file_name, particle_type='electrons', units=available_units('p2_sat_UHI'))
        PS = PhaseSpace(ps_data)
    """
    def _check_input_data(self):
        if not Path(self._input_data).is_file():
            raise FileNotFoundError(f'input data file {self._import_data()} does not exist')
        if not self._particle_type:
            raise Exception('particle_type must be specified when readin p2sat_txt data')

    def _import_data(self):
        # Initialize data lists
        w = []
        x, y, z = [], [], []
        px, py, pz = [], [], []
        t = []

        # Open file
        with open(self._input_data, 'r') as f:
            # Loop over lines
            for line in f.readlines():
                # If current line is not a comment, save data
                if line[0] != "#":
                    data = line.split(",")
                    w.append(float(data[0]))
                    x.append(float(data[1]))
                    y.append(float(data[2]))
                    z.append(float(data[3]))
                    px.append(float(data[4]))
                    py.append(float(data[5]))
                    pz.append(float(data[6]))
                    t.append(float(data[7]))


        self.data[self._columns['x']] = x
        self.data[self._columns['y']] = y
        self.data[self._columns['z']] = z
        self.data[self._columns['time']] = t
        self.data[self._columns['weight']] = w

        self.data[self._columns['particle id']] = np.arange(self.data.shape[0])
        self.data[self._columns['particle type']] = particle_cfg.particle_properties[self._particle_type]['pdg_code']

        self.data[self._columns['px']] = px
        self.data[self._columns['py']] = py
        self.data[self._columns['pz']] = pz


class Load_IAEA(_DataLoadersBase):
    """
    this loads a binary varian IAEA sent through the topas forums.
    Because this format is so arbitrary, uses are required to pass a data_schema variable indicating the order
    and types of the data in the phase space, because there is no general way for us to figure this out.
    Please see `here <https://bwheelz36.github.io/ParticlePhaseSpace/IAEA.html>`_ for examples of how to
    use this data loader.

    :param data_schema: the types and order of data, specified as an np.dtype
    :type data_schema: np.dtype
    :param constants: any constants in the phase space
    :type constatnts: dict
    :param input_data: path to the a .phsp or .IAEAphsp file
    :param n_records: specify how many rows of data to read in. By default, will read all rows.
    :param offset: which row to start at. defaults to 0 (first row). Can be used in conjunction with n_records
        to read a large file in a series of small chunks

    Example::

        from ParticlePhaseSpace import PhaseSpace, DataLoaders
        from pathlib import Path
        import numpy as np

        file_name = Path(r'/home/brendan/Downloads/Varian_TrueBeam6MV_01.phsp')
        data_schema = np.dtype([
                               ('particle type', 'i1'),
                               ('Ek', 'f4'),
                               ('x', 'f4'),
                               ('y', 'f4'),
                               ('z', 'f4'),
                               ('Cosine X', 'f4'),
                               ('Cosine Y', 'f4')
                               ])
        constants = {'weight': np.int8(1)}
        ps_data = DataLoaders.Load_IAEA(data_schema=data_schema, constants=constants, input_data=file_name, n_records=int(1e5))
        PS = PhaseSpace(ps_data)
    """

    def __init__(self,data_schema: np.dtype, constants: dict, input_data: (str, Path),  n_records:int=-1, offset=0, **kwargs):
        self._data_schema = data_schema
        self._constants = constants
        self._n_records = n_records
        self._offset = offset
        super().__init__(input_data, **kwargs)

    def _check_input_data(self):
        if not Path(self._input_data).is_file():
            raise FileNotFoundError(f'input data file {self._import_data()} does not exist')
        if not Path(self._input_data).suffix == '.phsp' or Path(self._input_data).suffix == '.IAEAphsp':
            raise Exception('This data loader reads in files of extension *.phsp or *.IAEAphsp')
        if self._particle_type:
            warnings.warn('particle type is ignored in IAEA read in')
        if not self._input_data.with_suffix('.header').is_file():
            raise FileNotFoundError(f'expected header file at {self._input_data.with_suffix(".header")} but'
                                    f'was not found')

    def _import_data(self):
        self._header_to_dict()
        self._check_required_info_present()
        self._check_record_length_versus_data_schema()
        for quantity in ['Cosine X', 'Cosine Y', 'Ek']:
            if not quantity in self._data_schema.fields:
                raise AttributeError('need at least Cosine X, Cosine Y, and Ek to read data')
        data = np.fromfile(self._input_data, dtype=self._data_schema, count=self._n_records, offset=self._offset)
        for field in self._data_schema.fields:
            if field in self._columns.keys():
                if self._columns[field] in self._required_columns:
                    self.data[self._columns[field]] = data[field]
        for constant in self._constants:
            if constant in self._columns.keys():
                if self._columns[constant] in self._required_columns:
                    self.data[self._columns[constant]] = pd.Series(self._constants[constant] *
                                                                   np.ones(self.data.shape[0]), dtype="category")

        # OK; add the fields I assume is not in the data
        particle_types_pdg = self._iaea_types_to_pdg(data['particle type'])
        self._check_n_particles_in_header(particle_types_pdg)
        self.data[self._columns['particle type']] = particle_types_pdg
        self.data[self._columns['particle id']] = np.arange(
            len(self.data))  # may want to replace with track ID if available?
        self.data[self._columns['time']] = pd.Series(0 * np.ones(self.data.shape[0]), dtype="category")  # may want to replace with time feature if available?
        # figure out the momentums:
        DirCosineX = data['Cosine X']
        DirCosineY = data['Cosine Y']
        E = data['Ek']
        if E.min() < 0:
            warnings.warn('this data has negative energy in it, what does that even mean. forcing all energy to positive')
            E = np.abs(E)
        self._rest_masses = get_rest_masses_from_pdg_codes(particle_types_pdg)
        P = np.sqrt((E + self._rest_masses) ** 2 - self._rest_masses ** 2)
        self.data[self._columns['px']] = pd.Series(np.multiply(P, DirCosineX), dtype=np.float32)
        self.data[self._columns['py']] = pd.Series(np.multiply(P, DirCosineY), dtype=np.float32)
        temp = P ** 2 - self.data[self._columns['px']] ** 2 - self.data[self._columns['py']] ** 2
        _negative_temp_ind = temp < 0
        if any(_negative_temp_ind):
            # this should never happen, but does occur when pz is essentially 0. we will attempt to resolve it here.
            negative_locations = np.where(_negative_temp_ind)[0]
            n_negative_locations = np.count_nonzero(_negative_temp_ind)
            momentum_precision_factor = 2e-3
            for location in negative_locations:
                relative_difference = np.divide(np.sqrt(abs(temp[location])), P[location])
                if relative_difference < momentum_precision_factor:
                    temp[location] = 0
                else:
                    raise Exception(f'failed to calculate momentums from topas data. Possible solution is to increase'
                                    f'the value of momentum_precision_factor, currently set to {momentum_precision_factor: 1.2e}'
                                    f'and failed data has value {relative_difference: 1.2e}')
            warnings.warn(f'{n_negative_locations: d} entries returned invalid pz values and were set to zero.'
                          f'\nWe will now check that momentum and energy are consistent to within '
                          f'{self._energy_consistency_check_cutoff: 1.4f} {self._units.energy.label}')

        self.data[self._columns['pz']] = pd.Series(np.sqrt(temp), dtype=np.float32)
        self._check_energy_consistency(Ek=E)

    def _iaea_types_to_pdg(self, varian_types):
        """
        convert varian integer type code to pdg integer type code
        :param varian_types:
        :return:
        """
        pdg_types = np.zeros(varian_types.shape, dtype=np.int32)
        for type in np.unique(varian_types):
            if not type in [1, 2, 3, 4, 5]:
                raise TypeError(f'unknown particle code {type}')
        pdg_types[varian_types==1] = 22  # gammas
        pdg_types[varian_types == 2] = 11  # electrons
        pdg_types[varian_types == 3] = -11  # positrons
        pdg_types[varian_types == 4] = 2112  # neutrons
        pdg_types[varian_types == 5] = 2212  # protons
        for type in np.unique(pdg_types):
            if not type in [22, 11, -11, 2112, 2212]:
                raise TypeError(f'unknown particle code {type}')

        return pdg_types

    def _header_to_dict(self):
        """
        try to extract quantities from the IAEA header
        for now will just get what I need, may be updated later
        :return:
        """
        self._header_dict = {}
        # read the header file
        with open(self._input_data.with_suffix('.header')) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if '$' in line:  # then we have a variable definition
                var_name = line[1:-2]
                if var_name == 'RECORD_CONSTANT':
                    self._header_dict[var_name] = {}
                    for var_line in lines[i+1:]:
                        if var_line == '\n':
                            break
                        try:
                            value = float(re.split('//', var_line)[0])
                            gah = re.split('//', var_line)[1]
                            # name = re.search(r' \S+\s', gah).group()
                            name = re.split(' ', gah[:-1])[2]
                            self._header_dict[var_name][name] = value
                        except Exception as e:
                            print('failed to extract data from IAEA header during RECORD_CONSTANT stage')
                            raise e
                elif var_name == 'RECORD_CONTENTS':
                    self._header_dict[var_name] = {}
                    for var_line in lines[i+1:]:
                        if var_line == '\n':
                            break
                        # extract var_name
                        try:
                            sub_var_name = re.search(' \S+ ',re.split('//', var_line)[1])[0]
                            sub_var_name = sub_var_name.replace(' ','')
                            exists = bool(int(re.split('//', var_line)[0]))
                            self._header_dict[var_name][sub_var_name] = exists
                        except Exception as e:
                            print('failed to extract data from IAEA header during RECORD_CONTENTS stage')
                            raise e
                elif var_name in ['RECORD_LENGTH', 'PARTICLES', 'PHOTONS', 'ELECTRONS','POSITRONS', 'PROTONS']:
                    self._header_dict[var_name] = int(lines[i+1])

    def _check_required_info_present(self):
        """
        check that all quantities required for read in are either present in data
        or defined as constants
        :return:
        """
        defined_quantities = list(self._data_schema.fields.keys()) + list(self._constants.keys())
        required_quantities = ['x', 'y', 'z', 'Ek', 'Cosine X', 'Cosine Y', 'weight']
        for quantity in required_quantities:
            if not quantity in defined_quantities:
                raise Exception(f'could not find required quantity {quantity}')

    def _check_record_length_versus_data_schema(self):
        """
        there is often a field called RECORD_LENGTH which describest the total byte length on each row.
        if this exists, make sure the
        input data_schema is consistent
        :return:
        """
        if 'RECORD_LENGTH' in self._header_dict.keys():
            if not self._data_schema.itemsize == self._header_dict['RECORD_LENGTH']:
                raise Exception(f'specified data schema has differeny byte length to that indicated in header.'
                                f'\nheader specifies {self._header_dict["RECORD_LENGTH"]},'
                                f'\nschema specifies {self._data_schema.itemsize}')

    def _check_n_particles_in_header(self, pdg_codes):
        """
        check that the number of particles we extracted from the record matches what is in the header

        we won't stop reading in the data if it doesn't seem to match the header but we will warn
        """
        if not self._n_records == -1:
            # if the user is only reading chunks at a time this check makes no sense
            return
        unique_particle_codes = np.unique(pdg_codes)
        unique_particle_names = []
        for code in unique_particle_codes:
            unique_particle_names.append(particle_cfg.particle_properties[code]['name'])
        for particle_name, particle_code in zip(unique_particle_names, unique_particle_codes):
            # attempt to extract the number from the header dict
            if particle_name == 'gammas':
                particle_name = 'photons'
            try:
                n_particles_header = self._header_dict[particle_name.upper()]
            except KeyError:
                warnings.warn(f'\nunable to check number of particles is correct for species {particle_name}')
                n_particles_header = np.nan
            n_particles_code = np.count_nonzero(pdg_codes == particle_code)
            if not n_particles_header == n_particles_code:
                warnings.warn(f'this code read in {n_particles_code} {particle_name}, but the header specifies'
                              f'{n_particles_header}')
            try:
                if not len(pdg_codes) == self._header_dict['PARTICLES']:
                    warnings.warn(f'\nheader file specifies {self._header_dict["PARTICLES"]}, but read in'
                                  f'{len(pdg_codes)}')
            except KeyError:
                pass

import platform
import numpy as np
from ParticlePhaseSpace import PhaseSpace
from pathlib import Path
from abc import ABC, abstractmethod
from ParticlePhaseSpace import __phase_space_config__ as ps_cfg
from ParticlePhaseSpace import __particle_config__ as particle_cfg
import warnings

class _DataExportersBase(ABC):
    """
    Abstract base class to be inherited by other DataExporters
    """

    def __init__(self, PhaseSpaceInstance: PhaseSpace, output_location: (str, Path), output_name: str):

        if not isinstance(PhaseSpaceInstance, PhaseSpace):
            raise TypeError(f'PhaseSpaceInstance must be an instance of ParticlePhaseSpace.PhaseSpace,'
                            f'not {type(PhaseSpaceInstance)}')
        self._PS = PhaseSpaceInstance
        self._output_location = Path(output_location)
        self._check_output_location_exists()
        self._output_name = output_name
        self._required_columns = []  # filled in _define_required_columns
        self._define_required_columns()
        self._check_required_columns_allowed()
        self._fill_required_columns()
        self._export_data()

    def _check_output_location_exists(self):
        if not self._output_location.is_dir():
            raise FileNotFoundError(f'output_location should be an existing path;'
                                    f'\n{self._output_location}\n does not exist')

    def _check_required_columns_allowed(self):
        """
        check that the columns that are required for data export are actually allowed
        :return:
        """
        allowed_columns = ps_cfg.required_columns + ps_cfg.allowed_columns
        for col in self._required_columns:
            if not col in allowed_columns:
                raise AttributeError(f'column: "{col}" is required for export, but is not an allowed column name.')

    def _fill_required_columns(self):

        for col in self._required_columns:
            if not col in self._PS.ps_data.columns:
                if col == 'Ek [MeV]':
                    self._PS.fill_kinetic_E()
                elif col == 'rest mass [MeV/c^2]':
                    self._PS.fill_rest_mass()
                elif col in 'gamma, beta':
                    self._PS.fill_beta_and_gamma()
                elif col in 'vx [m/s], vy [m/s], vz [m/s]':
                    self._PS.fill_velocity()
                elif col in 'Direction Cosine X, Direction Cosine Y, Direction Cosine Z':
                    self._PS.fill_direction_cosines()
                else:
                    raise Exception(f'unable to fill required column {col}')

    @abstractmethod
    def _define_required_columns(self):
        """
        user should fill in the required columns here
        :return:
        """
        pass

    @abstractmethod
    def _export_data(self):
        """
        this is the method which should actually perform the data export
        :return:
        """
        pass


class Topas_Exporter(_DataExportersBase):
    """
    output the phase space to topas ascii format:
    https://topas.readthedocs.io/en/latest/parameters/source/phasespace.html

    Note:
        - we do not handle any time features
        - every particle in the phase space is flagged as being a new history.
    """

    def _define_required_columns(self):
        self._required_columns = ['x [mm]', 'y [mm]', 'z [mm]', 'Direction Cosine X', 'Direction Cosine Y',
                                  'Direction Cosine Z', 'Ek [MeV]', 'weight', 'particle id']

    def _export_data(self):
        """
        Convert Phase space into format appropriate for topas.

        You can read more about the required format
        `Here <https://topas.readthedocs.io/en/latest/parameters/scoring/phasespace.html>`_
        """

        if 'windows' in platform.system().lower():
            raise Exception('to generate a valid file, please use a unix-based system')
        print('generating topas data file')

        self._generate_topas_header_file()
        # make sure output is in correct format
        if not Path(self._output_name).suffix == '.phsp':
            _output_name = str(self._output_name) + '.phsp'
        else:
            _output_name = self._output_name
        WritefilePath = Path(self._output_location) / _output_name

        first_particle_flag = np.ones(self._PS.ps_data['x [mm]'].shape[0])
        third_direction_flag = np.int8(self._PS.ps_data['Direction Cosine Z'] < 0)

        # Nb: topas requires units of cm
        Data = [self._PS.ps_data['x [mm]'].to_numpy() * 0.1, self._PS.ps_data['y [mm]'].to_numpy() * 0.1,
                self._PS.ps_data['z [mm]'].to_numpy() * 0.1,
                self._PS.ps_data['Direction Cosine X'].to_numpy(), self._PS.ps_data['Direction Cosine Y'].to_numpy(),
                self._PS.ps_data['Ek [MeV]'].to_numpy(), self._PS.ps_data['weight'].to_numpy(),
                self._PS.ps_data['particle type [pdg_code]'].to_numpy(),
                third_direction_flag, first_particle_flag]

        # write the data to a text file
        Data = np.transpose(Data)
        FormatSpec = ['%11.5f', '%11.5f', '%11.5f', '%11.5f', '%11.5f', '%11.5f', '%11.5f', '%2d', '%2d', '%2d']
        np.savetxt(WritefilePath, Data, fmt=FormatSpec, delimiter='      ')
        print('success')

    def _generate_topas_header_file(self):
        """
        Generate the header file required for a topas phase space source.
        This is only intended to be used from within the class (private method)
        """

        if Path(self._output_name).suffix == '.phsp':
            _output_name = Path(self._output_name).stem
        else:
            _output_name = self._output_name
        _output_name = str(_output_name) + '.header'
        WritefilePath = self._output_location / _output_name

        ParticlesInPhaseSpace = str(len(self._PS.ps_data['x [mm]'] ))
        TopasHeader = []

        TopasHeader.append('TOPAS ASCII Phase Space\n')
        TopasHeader.append('Number of Original Histories: ' + ParticlesInPhaseSpace)
        TopasHeader.append('Number of Original Histories that Reached Phase Space: ' + ParticlesInPhaseSpace)
        TopasHeader.append('Number of Scored Particles: ' + ParticlesInPhaseSpace + '\n')
        TopasHeader.append('Columns of data are as follows:')
        TopasHeader.append(' 1: Position X [cm]')
        TopasHeader.append(' 2: Position Y [cm]')
        TopasHeader.append(' 3: Position Z [cm]')
        TopasHeader.append(' 4: Direction Cosine X')
        TopasHeader.append(' 5: Direction Cosine Y')
        TopasHeader.append(' 6: Energy [MeV]')
        TopasHeader.append(' 7: Weight')
        TopasHeader.append(' 8: Particle Type (in PDG Format)')
        TopasHeader.append(' 9: Flag to tell if Third Direction Cosine is Negative (1 means true)')
        TopasHeader.append(' 10: Flag to tell if this is the First Scored Particle from this History (1 means true)\n')
        particle_number_string = []
        minimum_Ek_string = []
        maximum_Ek_string = []
        for particle in self._PS._unique_particles:
            if particle_cfg.particle_properties[particle]['name'] == 'electrons':
                electron_PS = self._PS('electrons')
                electron_PS.fill_kinetic_E()
                particle_number_string.append('Number of e-: ' + str(len(electron_PS.ps_data['x [mm]'])) )
                minimum_Ek_string.append('Minimum Kinetic Energy of e-: ' + str(min(electron_PS.ps_data['Ek [MeV]'])) + ' MeV')
                maximum_Ek_string.append('Maximum Kinetic Energy of e-: ' + str(max(electron_PS.ps_data['Ek [MeV]'])) + ' MeV')
            elif particle_cfg.particle_properties[particle]['name'] == 'positrons':
                positron_PS = self._PS('positrons')
                positron_PS.fill_kinetic_E()
                particle_number_string.append('Number of e+: ' + str(len(positron_PS.ps_data['x [mm]'])))
                minimum_Ek_string.append('Minimum Kinetic Energy of e+: ' + str(min(positron_PS.ps_data['Ek [MeV]'])) + ' MeV')
                maximum_Ek_string.append('Maximum Kinetic Energy of e+: ' + str(max(positron_PS.ps_data['Ek [MeV]'])) + ' MeV')
            elif particle_cfg.particle_properties[particle]['name'] == 'gammas':
                gamma_PS = self._PS('gammas')
                gamma_PS.fill_kinetic_E()
                particle_number_string.append('Number of gamma: ' + str(len(gamma_PS.ps_data['x [mm]'])))
                minimum_Ek_string.append('Minimum Kinetic Energy of gamma: ' + str(min(gamma_PS.ps_data['Ek [MeV]'])) + ' MeV')
                maximum_Ek_string.append('Maximum Kinetic Energy of gamma: ' + str(max(gamma_PS.ps_data['Ek [MeV]'])) + ' MeV')
            else:
                raise NotImplementedError(f'cannot currently export particle type {particle_cfg.particle_properties[particle]["name"]}.'
                                          f'\nneed to update header writer')
        for line in particle_number_string:
            TopasHeader.append(line)
        TopasHeader.append('')
        for line in minimum_Ek_string:
            TopasHeader.append(line)
        TopasHeader.append('')
        for line in maximum_Ek_string:
            TopasHeader.append(line)

        # open file:
        f = open(WritefilePath, 'w')
        # Write file line by line:
        for Line in TopasHeader:
            f.write(Line)
            f.write('\n')
        f.close()




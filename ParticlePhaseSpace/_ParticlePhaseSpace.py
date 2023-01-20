import numpy as np
from matplotlib import pyplot as plt
import logging
import pandas as pd
logging.basicConfig(level=logging.WARNING)
import warnings
from scipy.stats import gaussian_kde
from scipy import constants
import json
from pathlib import Path
import ParticlePhaseSpace.__phase_space_config__ as ps_cfg
import ParticlePhaseSpace.__particle_config__ as particle_cfg
from ParticlePhaseSpace.DataLoaders import _DataLoadersBase
from ParticlePhaseSpace import utilities as ps_util
from ParticlePhaseSpace import DataLoaders

class _FigureSpecs:
    """
    Thought this might be the easiest way to ensure universal parameters accross all figures
    """
    LabelFontSize = 14
    TitleFontSize = 16
    Font = 'serif'
    AxisFontSize = 14
    TickFontSize = 14


class PhaseSpace:
    """
    This class holds phase space data in  a pandas dataframe, and allowed users to utilise common libraries for
    plotting and analysis. It accepts data from any `DataLoader <https://bwheelz36.github.io/ParticlePhaseSpace/code_docs.html#module-ParticlePhaseSpace.DataLoaders>`_
    Basic use is documented `here <https://bwheelz36.github.io/ParticlePhaseSpace/basic_example.html>`_.

    :param data_loader: an instance of ParticlePhaseSpace.DataLoaders._DataLoadersBase
    :type data_loader: _DataLoadersBase
    """

    def __init__(self, data_loader):

        if not isinstance(data_loader, _DataLoadersBase):
            raise TypeError(f'ParticlePhaseSpace must be instantiated with a valid object'
                            f'from DataLoaders, not {type(data_loader)}')
        self._ps_data = data_loader.data
        self._unique_particles = self._ps_data['particle type [pdg_code]'].unique()
        self.twiss_parameters = {}
        self.energy_stats = {}

    def __call__(self, particle_list):
        """
        this function allows users to seperate the phase space based on particle types

        users should be able to specify strings or pdg codes.
        if string, then we should convert to pdg code.
        """
        if not isinstance(particle_list, list):
            particle_list = [particle_list]
        # check all same type:
        _types = [type(particle) for particle in particle_list]
        _types = set(_types)
        if len(_types) > 1:
            raise TypeError(f'particle_list must contain all strings or all integers')
        # if pdg_codes, convery to names:
        if not str in _types:
            # nb: check for str instead of int as lots of different int types, so will assume
            # we have ints here and raise error if failure
            try:
                new_particle_list = [particle_cfg.particle_properties[particle]['name'] for particle in particle_list]
            except KeyError:
                raise Exception('unable to convert input particle_list to valid data, please check')
            particle_list = new_particle_list
        allowed_particles = list(particle_cfg.particle_properties.keys())
        for particle in particle_list:
            if not particle in allowed_particles:
                raise Exception(f'particle type {particle} is unknown')
        particle_data_sets = []
        for particle in particle_list:
            pdg_code = particle_cfg.particle_properties[particle]['pdg_code']
            particle_data = self._ps_data.loc[self._ps_data['particle type [pdg_code]'] == pdg_code].copy(deep=True)
            particle_data.reset_index(inplace=True, drop=True)
            # delete any non required columns
            for col_name in particle_data.columns:
                if not col_name in ps_cfg.required_columns:
                    particle_data.drop(columns=col_name, inplace=True)
            # create a new instance of _DataImportersBase based on particle_data
            particle_data_loader = DataLoaders.LoadPandasData(particle_data)
            particle_instance = PhaseSpace(particle_data_loader)
            particle_data_sets.append(particle_instance)

        if len(particle_data_sets) == 1:
            return particle_data_sets[0]
        else:
            return tuple(particle_data_sets)

    def __add__(self, other):
        """
        add two phase spaces together. requires that each phase space has
        unique particle IDs
        """
        new_data = pd.concat([self._ps_data, other.ps_data])

        for col_name in new_data.columns:
            if not col_name in ps_cfg.required_columns:
                new_data.drop(columns=col_name, inplace=True)
        new_data_loader = DataLoaders.LoadPandasData(new_data)
        new_instance = PhaseSpace(new_data_loader)
        return new_instance

    def __sub__(self, other):
        """
        subtract phase space (remove every particle in other from self)
        """
        new_data = pd.merge(self._ps_data, other.ps_data, how='outer', indicator=True)\
            .query("_merge != 'both'")\
            .drop('_merge', axis=1)\
            .reset_index(drop=True)
        for col_name in new_data.columns:
            if not col_name in ps_cfg.required_columns:
                new_data.drop(columns=col_name, inplace=True)
        new_data_loader = DataLoaders.LoadPandasData(new_data)
        new_instance = PhaseSpace(new_data_loader)
        return new_instance

    def __len__(self):
        return self.ps_data.shape[0]

    @property
    def ps_data(self):
        return self._ps_data

    @ps_data.setter
    def ps_data(self, new_data_frame):
        """
        gets run whenever ps_data gets changed
        :param new_data_frame:
        :return:
        """
        self._ps_data = new_data_frame
        self.reset_phase_space()
        self._assert_unique_particle_ids()
        self._check_ps_data_format()
        self._unique_particles = self._ps_data['particle type [pdg_code]'].unique()

    def _weighted_median(self, data, weights):
        """
        calculate a weighted median
        @author Jack Peterson (jack@tinybike.net)
        credit: https://gist.github.com/tinybike/d9ff1dad515b66cc0d87

        Args:
          data (list or numpy.array): data
          weights (list or numpy.array): weights
        """
        data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
        s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
        midpoint = 0.5 * sum(s_weights)
        if any(weights > midpoint):
            w_median = (data[weights == np.max(weights)])[0]
        else:
            cs_weights = np.cumsum(s_weights)
            idx = np.where(cs_weights <= midpoint)[0][-1]
            if cs_weights[idx] == midpoint:
                w_median = np.mean(s_data[idx:idx + 2])
            else:
                w_median = s_data[idx + 1]
        return w_median

    def _weighted_avg_and_std(self, values, weights):
        """
        credit: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values - average) ** 2, weights=weights)
        return (average, np.sqrt(variance))

    def _weighted_quantile(self, values, quantiles, sample_weight=None):
        """
        credit: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy

        Very close to numpy.percentile, but supports weights.
        NOTE: quantiles should be in [0, 1]!
        :param values: numpy.array with data
        :param quantiles: array-like with many quantiles needed
        :param sample_weight: array-like of the same length as `array`
        :param values_sorted: bool, if True, then will avoid sorting of
            initial array
        :param old_style: if True, will correct output to be consistent
            with numpy.percentile.
        :return: numpy.array with computed quantiles.
        """
        values = np.array(values)
        quantiles = np.array(quantiles)
        if sample_weight is None:
            sample_weight = np.ones(len(values))
        sample_weight = np.array(sample_weight)
        assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]
        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
        weighted_quantiles /= np.sum(sample_weight)
        return np.interp(quantiles, weighted_quantiles, values)

    def _check_ps_data_format(self):
        """
        check that the phase space remains consistent with what is mandated in __config
        :return:
        """

        all_allowed_columns = ps_cfg.required_columns + ps_cfg.allowed_columns
        for col_name in self._ps_data.columns:
            if not col_name in all_allowed_columns:
                raise AttributeError(f'non allowed column name {col_name} in ps_data')

    def _get_ellipse_xy_points(self, ellipse_parameters, x_search_min,  # pragma: no cover
                               x_search_max, xpq_search_min, xpq_search_max):
        """
        given the parameters of an ellipse, return a set of points in XY which meet those parameters
        :return:
        """
        gamma = ellipse_parameters['gamma']
        alpha = ellipse_parameters['alpha']
        beta = ellipse_parameters['beta']
        epsilon = ellipse_parameters['epsilon']

        # set up search grid:
        xq = np.linspace(x_search_min, x_search_max, 1000)
        xpq = np.linspace(xpq_search_min, xpq_search_max, 1000)
        [ElipseGridx, ElipseGridy] = np.meshgrid(xq, xpq)
        # find matching points
        EmittanceGrid = (gamma * np.square(ElipseGridx)) + \
                        (2 * alpha * np.multiply(ElipseGridx, ElipseGridy)) + \
                        (beta * np.square(ElipseGridy))
        tol = .01 * epsilon
        Elipse = (EmittanceGrid >= epsilon - tol) & (EmittanceGrid <= epsilon + tol)
        ElipseIndex = np.where(Elipse == True)
        elipseX = ElipseGridx[ElipseIndex]
        elipseY = ElipseGridy[ElipseIndex]
        return elipseX, elipseY

    # public methods

    def plot_energy_histogram(self, n_bins=100, title=None):  # pragma: no cover
        """
        generate a histogram plot of paritcle energies.
        Each particle spcies present in the phase space is overlaid  on the same plot.

        :param n_bins: number of bins in histogram
        :param title: title of histogram
        :return: None
        """
        Efig, axs = plt.subplots()
        if not 'Ek [MeV]' in self._ps_data.columns:
            self.fill_kinetic_E()
        legend = []
        for particle in self._unique_particles:
            legend.append(particle_cfg.particle_properties[particle]['name'])
            ind = self._ps_data['particle type [pdg_code]'] == particle
            Eplot = self._ps_data['Ek [MeV]'][ind]
            n, bins, patches = axs.hist(Eplot, bins=n_bins, weights=self._ps_data['weight'][ind], alpha=.5)

        axs.set_xlabel('Energy [MeV]', fontsize=_FigureSpecs.LabelFontSize)
        axs.set_ylabel('N counts', fontsize=_FigureSpecs.LabelFontSize)
        if title:
            axs.set_title(title, fontsize=_FigureSpecs.TitleFontSize)
        axs.tick_params(axis="y", labelsize=_FigureSpecs.TickFontSize)
        axs.tick_params(axis="x", labelsize=_FigureSpecs.TickFontSize)
        axs.legend(legend)
        plt.tight_layout()
        plt.show()  # pragma: no cover

    def plot_position_histogram(self, n_bins=100, alpha=0.5):  # pragma: no cover
        """
        plot a histogram of particle positions in x, y, z.
        a new histogram is generated for each particle species.
        histograms are overlaid.

        :param n_bins:  number of bins in histogram
        :param alpha: controls transparency of each histogram.
        :return:
        """
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(15, 5)
        legend = []
        for particle in self._unique_particles:
            legend.append(particle_cfg.particle_properties[particle]['name'])
            ind = self._ps_data['particle type [pdg_code]'] == particle
            x_plot = self._ps_data['x [mm]'][ind]
            y_plot = self._ps_data['y [mm]'][ind]
            z_plot = self._ps_data['z [mm]'][ind]
            axs[0].hist(x_plot, bins=n_bins, weights=self._ps_data['weight'][ind], alpha=alpha)
            axs[1].hist(y_plot, bins=n_bins, weights=self._ps_data['weight'][ind], alpha=alpha)
            axs[2].hist(z_plot, bins=n_bins, weights=self._ps_data['weight'][ind], alpha=alpha)

        axs[0].set_xlabel('x [mm]')
        axs[0].set_ylabel('counts')
        axs[0].set_title('x [mm]')
        axs[0].legend(legend)

        axs[1].set_xlabel('y [mm]')
        axs[1].set_ylabel('counts')
        axs[1].set_title('y [mm]')

        axs[2].set_xlabel('Z position [mm]')
        axs[2].set_ylabel('counts')
        axs[2].set_title('z [mm]')

        plt.tight_layout()
        plt.show()

    def plot_particle_positions(self, beam_direction='z', weight_position_plot=False,
                                xlim=None, ylim=None):  # pragma: no cover
        """
        produce a scatter plot of particle positions.
        one plot is produced for each unique species.

        :param beam_direction: the direction the beam is travelling in. "x", "y", or "z" (default)
        :type beam_direction: str, optional
        :param weight_position_plot: if True, a gaussian kde is used to weight the particle
            positions. This can produce very informative and useful plots, but also be very slow.
            If it is slow, you could try downsampling the phase space first using get_downsampled_phase_space
        :type weight_position_plot: bool
        :param xlim: set the xlim for all plots, e.g. [-2,2]
        :type xlim: list or None, optional
        :param ylim: set the ylim for all plots, e.g. [-2,2]
        :type ylim: list or None, optional
        :return: None
        """


        fig, axs = plt.subplots(1, len(self._unique_particles), squeeze=False)
        fig.set_size_inches(5*len(self._unique_particles), 5)
        n_axs = 0
        for particle in self._unique_particles:
            ind = self._ps_data['particle type [pdg_code]'] == particle
            ps_data = self._ps_data.loc[ind]
            axs_title = particle_cfg.particle_properties[particle]['name']
            if beam_direction == 'x':
                x_data = ps_data['y [mm]']
                y_data = ps_data['z [mm]']
                x_label = 'y [mm]'
                y_label = 'z [mm]'
            elif beam_direction == 'y':
                x_data = ps_data['x [mm]']
                y_data = ps_data['z [mm]']
                x_label = 'x [mm]'
                y_label = 'z [mm]'
            elif beam_direction == 'z':
                x_data = ps_data['x [mm]']
                y_data = ps_data['y [mm]']
                x_label = 'x [mm]'
                y_label = 'y [mm]'
            else:
                raise NotImplementedError('beam_direction must be "x", "y", or "z"')

            if weight_position_plot:
                _kde_data_grid = 150 ** 2
                print('generating weighted scatter plot...can be slow...')
                xy = np.vstack([x_data, y_data])
                k = gaussian_kde(xy, weights=self._ps_data['weight'][ind])
                if x_data.shape[0] > _kde_data_grid:
                    # in this case we can downsample for display
                    down_sample_factor = np.round(x_data.shape[0] / _kde_data_grid)
                    print(f'down sampling scatter plot data by factor of {down_sample_factor}')
                    rng = np.random.default_rng()
                    rng.shuffle(xy)  # operates in place for some confusing reason
                    xy = rng.choice(xy, int(x_data.shape[0] / down_sample_factor), replace=False, axis=1, shuffle=False)
                z = k(xy)
                z = z / max(z)
                axs[0, n_axs].scatter(xy[0], xy[1], c=z, s=1)

            else:
                axs[0, n_axs].scatter(x_data, y_data, s=1, c=self._ps_data['weight'][ind])
            axs[0, n_axs].set_aspect(1)
            axs[0, n_axs].set_title(axs_title, fontsize=_FigureSpecs.TitleFontSize)
            axs[0, n_axs].set_xlabel(x_label, fontsize=_FigureSpecs.LabelFontSize)
            axs[0, n_axs].set_ylabel(y_label, fontsize=_FigureSpecs.LabelFontSize)
            if xlim:
                axs[0, n_axs].set_xlim(xlim)
            if ylim:
                axs[0, n_axs].set_ylim(ylim)
            n_axs = n_axs+1

        plt.tight_layout()
        plt.show()

    def plot_transverse_trace_space(self, beam_direction='z', plot_twiss_ellipse=True):  # pragma: no cover
        """
        Generate a scatter plot of x versus x'=px/pz and y versus y'=py/pz (these definitions are for
        beam_direction='z')

        :param beam_direction: main direction in which beam is travelling. 'x', 'y', or 'z' (default)
        :type beam_direction: str, optional
        :param plot_twiss_ellipse: if True, will overlay the RMS twiss ellipse onto the trace space
        :type plot_twiss_ellipse: bool, optional
        :return: None
        """

        self.calculate_twiss_parameters(beam_direction=beam_direction)
        fig, axs = plt.subplots(nrows=len(self._unique_particles), ncols=2, squeeze=False)
        row = 0
        for particle in self._unique_particles:
            particle_name = particle_cfg.particle_properties[particle]['name']
            ind = self._ps_data['particle type [pdg_code]'] == particle
            ps_data = self._ps_data.loc[ind]
            if beam_direction == 'z':
                x_data_1 = ps_data['x [mm]']
                div_data_1 = np.divide(ps_data['px [MeV/c]'], ps_data['pz [MeV/c]'])
                x_label_1 = 'x [mm]'
                y_label_1 = "x' [mrad]"
                title_1 = particle_name + ': x'
                weight = ps_data['weight']
                elipse_parameters_1 = self.twiss_parameters[particle_name]['x']

                x_data_2 = ps_data['y [mm]']
                div_data_2 = np.divide(ps_data['py [MeV/c]'], ps_data['pz [MeV/c]'])
                x_label_2 = 'y [mm]'
                y_label_2 = "y' [mrad]"
                title_2 = particle_name  + ': y'
                elipse_parameters_2 = self.twiss_parameters[particle_name ]['y']
            elif beam_direction == 'x':
                x_data_1 = ps_data['y [mm]']
                div_data_1 = np.divide(ps_data['py [MeV/c]'], ps_data['px [MeV/c]'])
                x_label_1 = 'y [mm]'
                y_label_1 = "y' [mrad]"
                title_1 = particle_name + ': x'
                weight = ps_data['weight']
                elipse_parameters_1 = self.twiss_parameters[particle_name]['y']

                x_data_2 = ps_data['z [mm]']
                div_data_2 = np.divide(ps_data['pz [MeV/c]'], ps_data['px [MeV/c]'])
                x_label_2 = 'z [mm]'
                y_label_2 = "z' [mrad]"
                title_2 = particle_name + ': y'
                elipse_parameters_2 = self.twiss_parameters[particle_name]['z']
            elif beam_direction == 'y':
                x_data_1 = ps_data['x [mm]']
                div_data_1 = np.divide(ps_data['px [MeV/c]'], ps_data['py [MeV/c]'])
                x_label_1 = 'x [mm]'
                y_label_1 = "x' [mrad]"
                title_1 = particle_name + ': x'
                weight = ps_data['weight']
                elipse_parameters_1 = self.twiss_parameters[particle_name]['x']

                x_data_2 = ps_data['z [mm]']
                div_data_2 = np.divide(ps_data['pz [MeV/c]'], ps_data['py [MeV/c]'])
                x_label_2 = 'z [mm]'
                y_label_2 = "z' [mrad]"
                title_2 = particle_name + ': y'
                elipse_parameters_2 = self.twiss_parameters[particle_name]['z']
            else:
                raise NotImplementedError(f'beam_direction must be "x", "y", or "z", not {beam_direction}')


            axs[row, 0].scatter(x_data_1, div_data_1, s=1, marker='.', c=weight)
            axs[row, 0].set_xlabel(x_label_1)
            axs[row, 0].set_ylabel(y_label_1)
            axs[row, 0].set_title(title_1)
            if plot_twiss_ellipse:
                twiss_X, twiss_Y = self._get_ellipse_xy_points(elipse_parameters_1, x_data_1.min(), x_data_1.max(),
                                                               div_data_1.min(), div_data_1.max())
                axs[row, 0].scatter(twiss_X, twiss_Y, c='r')
                # axs[row, 0].set_xlim([3*np.min(twiss_X), 3*np.max(twiss_X)])
                # axs[row, 0].set_ylim([3 * np.min(twiss_Y), 3 * np.max(twiss_Y)])
            axs[row, 0].grid()

            if plot_twiss_ellipse:
                twiss_X, twiss_Y = self._get_ellipse_xy_points(elipse_parameters_2, x_data_2.min(), x_data_2.max(),
                                                               div_data_2.min(), div_data_2.max())
                axs[row, 1].scatter(twiss_X, twiss_Y, c='r')
            axs[row, 1].scatter(x_data_2, div_data_2, s=1, marker='.', c=weight)
            axs[row, 1].set_xlabel(x_label_2)
            axs[row, 1].set_ylabel(y_label_2)
            axs[row, 1].set_title(title_2)
            axs[row, 1].grid()
            row = row + 1

        plt.tight_layout()
        plt.show()

    def print_energy_stats(self, file_name=None):  # pragma: no cover
        """
        Prints a sumary of the phase space to the screen.
        """
        if not self.energy_stats:
            self.calculate_energy_statistics()
        if file_name:
            file_name = Path(file_name)
            if not file_name.parent.is_dir():
                raise NotADirectoryError(f'{file_name.parent} is not a directory')
            if not file_name.suffix == '.json':
                file_name = file_name.parent / (file_name.name + '.json')
            with open(file_name, 'w') as fp:
                json.dump(self.energy_stats, fp)
        print('===================================================')
        print('                 ENERGY STATS                  ')
        print('===================================================')
        print(f'total number of particles in phase space: {len(self): d}')

        print(f'number of unique particle species: {len(self._unique_particles): d}')
        for particle in self.energy_stats:
            print(f'    {self.energy_stats[particle]["number"]: d} {particle_cfg.particle_properties[particle]["name"]}'
                  f'\n        mean energy: {self.energy_stats[particle]["mean energy"]: 1.2f} MeV'
                  f'\n        median energy: {self.energy_stats[particle]["median energy"]: 1.2f} MeV'
                  f'\n        Energy spread IQR: {self.energy_stats[particle]["energy spread IQR"]: 1.2f} MeV'
                  f'\n        min energy {self.energy_stats[particle]["min energy"]: 1.2f} MeV'
                  f'\n        max energy {self.energy_stats[particle]["max energy"]: 1.2f} MeV')

    def print_twiss_parameters(self, file_name=None, beam_direction='z'):  #pragma: no cover
        """
        prints the twiss parameters if they exist
        they are always printed to the screen.
        if filename is specified, they are also saved to file as json

        :param file_name: filename to write twiss data to. should be absolute
            path to an existing directory
        :return: None
        """
        if not self.twiss_parameters:
            self.calculate_twiss_parameters(beam_direction=beam_direction)
        if file_name:
            file_name = Path(file_name)
            if not file_name.parent.is_dir():
                raise NotADirectoryError(f'{file_name.parent} is not a directory')
            if not file_name.suffix == '.json':
                file_name = file_name.parent / (file_name.name + '.json')
            with open(file_name, 'w') as fp:
                json.dump(self.twiss_parameters, fp)

        print('===================================================')
        print('                 TWISS PARAMETERS                  ')
        print('===================================================')
        for particle in self._unique_particles:
            particle_name = particle_cfg.particle_properties[particle]['name']
            print(f'\n{particle_name}:')
            data = pd.DataFrame(self.twiss_parameters[particle_name])
            print(data)

    def fill_kinetic_E(self):
        """
        adds kinetic energy into self._ps_data
        """
        if not hasattr(self,'_rest_masses'):
            self.fill_rest_mass()
        Totm = np.sqrt(self._ps_data['px [MeV/c]'] ** 2 + self._ps_data['py [MeV/c]'] ** 2 + self._ps_data['pz [MeV/c]'] ** 2)
        TOT_E = np.sqrt(Totm ** 2 + self._ps_data['rest mass [MeV/c^2]'] ** 2)
        Kin_E = np.subtract(TOT_E, self._ps_data['rest mass [MeV/c^2]'])
        self._ps_data['Ek [MeV]'] = Kin_E
        self._check_ps_data_format()

    def fill_rest_mass(self):
        """
        add rest mass data to self._ps_data
        :return: 
        """
        self._ps_data['rest mass [MeV/c^2]'] = ps_util.get_rest_masses_from_pdg_codes(self._ps_data['particle type [pdg_code]'])
        self._check_ps_data_format()

    def fill_velocity(self):
        """
        add velocities into self._ps_data
        """
        if not 'rest mass [MeV/c^2]' in self._ps_data.columns:
            self.fill_rest_mass()
        if not 'gamma' in self._ps_data.columns:
            self.fill_beta_and_gamma()
        self._ps_data['vx [m/s]'] = np.divide(self._ps_data['px [MeV/c]'], (self._ps_data['gamma'] * self._ps_data['rest mass [MeV/c^2]']))
        self._ps_data['vy [m/s]'] = np.divide(self._ps_data['py [MeV/c]'], (self._ps_data['gamma'] * self._ps_data['rest mass [MeV/c^2]']))
        self._ps_data['vz [m/s]'] = np.divide(self._ps_data['pz [MeV/c]'], (self._ps_data['gamma'] * self._ps_data['rest mass [MeV/c^2]']))
        self._check_ps_data_format()

    def fill_beta_and_gamma(self):
        """
        add the relatavistic beta and gamma factors into self._ps_data
        """
        if not 'Ek [MeV]' in self._ps_data.columns:
            self.fill_kinetic_E()
        if not 'rest mass [MeV/c^2]' in self._ps_data.columns:
            self.fill_rest_mass()
        TOT_P = np.sqrt(self._ps_data['px [MeV/c]'] ** 2 + self._ps_data['py [MeV/c]'] ** 2 + self._ps_data['pz [MeV/c]'] ** 2)
        self._ps_data['beta'] = np.divide(TOT_P, self._ps_data['Ek [MeV]'] + self._ps_data['rest mass [MeV/c^2]'])
        self._ps_data['gamma'] = 1 / np.sqrt(1 - np.square(self._ps_data['beta'] ))
        self._check_ps_data_format()

    def fill_direction_cosines(self):
        """
        Calculate direction cosines, which are required for topas import:
        U (direction cosine of momentum with respect to X)
        V (direction cosine of momentum with respect to Y)
        :return:
        """

        V = np.sqrt(self._ps_data['px [MeV/c]'] ** 2 + self._ps_data['py [MeV/c]'] ** 2 + self._ps_data['pz [MeV/c]'] ** 2)
        self._ps_data['Direction Cosine X'] = self._ps_data['px [MeV/c]'] / V
        self._ps_data['Direction Cosine Y'] = self._ps_data['py [MeV/c]'] / V
        self._ps_data['Direction Cosine Z'] = self._ps_data['pz [MeV/c]'] / V
        self._check_ps_data_format()

    def get_downsampled_phase_space(self, downsample_factor=10):
        """
        return a new phase space object which randomlt samples from the larger phase space.
        the new phase space has size 'original data/downsample_factor'. the data is shuffled
        before being randomly sampled.

        :param downsample_factor: the factor to downsample the phase space by
        :type downsample_factor: int
        """
        new_data = self._ps_data.sample(frac=1).reset_index(drop=True)  # this shuffles the data
        new_data = new_data.sample(frac=1/downsample_factor, ignore_index=True)
        for col_name in new_data.columns:
            if not col_name in ps_cfg.required_columns:
                new_data.drop(columns=col_name, inplace=True)
        new_data_loader = DataLoaders.LoadPandasData(new_data)
        new_instance = PhaseSpace(new_data_loader)
        return new_instance

    def calculate_twiss_parameters(self, beam_direction='z'):
        """
        Calculate the RMS `twiss parameters <https://en.wikipedia.org/wiki/Courant%E2%80%93Snyder_parameters>`_

        :param beam_direction: main direction in which beam is travelling. 'x', 'y', or 'z' (default)
        :type beam_direction: str, optional
        :return: None
        """
        if beam_direction == 'x':
            intersection_columns = ['x [mm]', 'px [MeV/c]']
            direction_columns = [['z [mm]', 'pz [MeV/c]'], ['y [mm]', 'py [MeV/c]']]
        elif beam_direction == 'y':
            intersection_columns = ['y [mm]', 'py [MeV/c]']
            direction_columns = [['x [mm]', 'px [MeV/c]'], ['z [mm]', 'pz [MeV/c]']]
        elif beam_direction == 'z':
            intersection_columns = ['z [mm]', 'pz [MeV/c]']
            direction_columns = [['x [mm]', 'px [MeV/c]'], ['y [mm]', 'py [MeV/c]']]
        else:
            raise NotImplementedError('beam direction must be "x", "y", or "z"')
        for particle in self._unique_particles:
            particle_name = particle_cfg.particle_properties[particle]['name']
            ind = self._ps_data['particle type [pdg_code]'] == particle
            particle_data = self._ps_data[ind]
            self.twiss_parameters[particle_name] = {}
            for calc_dir in direction_columns:
                x2 = np.average(np.square(particle_data[calc_dir[0]]), weights=particle_data['weight'])
                xp = np.divide(particle_data[calc_dir[1]], particle_data[intersection_columns[1]])
                xp2 = np.average(np.square(xp), weights=particle_data['weight'])
                x_xp = np.average(np.multiply(particle_data[calc_dir[0]], xp), weights=particle_data['weight'])

                epsilon = np.sqrt((x2 * xp2) - (x_xp ** 2))
                alpha = -x_xp / epsilon
                beta = x2 / epsilon
                gamma = xp2 / epsilon

                self.twiss_parameters[particle_name][calc_dir[0][0]] = {'epsilon': epsilon,
                                                         'alpha': alpha,
                                                         'beta': beta,
                                                         'gamma': gamma}

    def calculate_energy_statistics(self):
        if not 'Ek [MeV]' in self._ps_data.columns:
            self.fill_kinetic_E()
        for particle in self._unique_particles:
            particle_name = particle_cfg.particle_properties[particle]['name']
            self.energy_stats[particle_name] = {}
            ind = self._ps_data['particle type [pdg_code]'] == particle
            ps_data = self._ps_data[ind]

            self.energy_stats[particle_name]['number'] = np.count_nonzero(ind)
            meanEnergy, stdEnergy = self._weighted_avg_and_std(ps_data['Ek [MeV]'], ps_data['weight'])
            self.energy_stats[particle_name]['min energy'] = ps_data['Ek [MeV]'].min()
            self.energy_stats[particle_name]['max energy'] = ps_data['Ek [MeV]'].max()
            self.energy_stats[particle_name]['mean energy'] = meanEnergy
            self.energy_stats[particle_name]['std mean'] = stdEnergy
            self.energy_stats[particle_name]['median energy'] = self._weighted_median(ps_data['Ek [MeV]'], ps_data['weight'])
            q75, q25 = self._weighted_quantile(ps_data['Ek [MeV]'], [0.25, 0.75], sample_weight=ps_data['weight'])
            self.energy_stats[particle_name]['energy spread IQR'] = q25 - q75



    def project_particles(self, beam_direction='z', distance=100):
        """
        Update the positions of each particle by projecting it forward/back by distance.

        This serves as a crude approximation to more advanced particle transport codes, but gives
        an indication of where the particles would end up in the absence of other forces.
        When this function is recalled, any dervied quantities will be deleted as it is too hard to check
        that they all remain correct.

        :param direction: the direction to project in. 'x', 'y', or 'z'
        :param distance: how far to project in mm
        :return: None
        """
        if not 'vx [m/s]' in self._ps_data.columns:
            self.fill_velocity()

        if beam_direction == 'z':
            self._ps_data['x [mm]'] = self._ps_data['x [mm]'] + np.divide(self._ps_data['vx [m/s]'], self._ps_data['vz [m/s]']) * distance
            self._ps_data['y [mm]'] = self._ps_data['y [mm]'] + np.divide(self._ps_data['vy [m/s]'], self._ps_data['vz [m/s]']) * distance
            self._ps_data['z [mm]'] = self._ps_data['z [mm]'] + distance
        else:
            raise NotImplementedError('havent coded the other directions yet')

        self.reset_phase_space()  # safest to get rid of any derived quantities

    def reset_phase_space(self):
        """
        reduce self._ps_data to only the required columns
        delete any other dervied quantities such as twiss parameters
        this can be called whenever you want to reduce the memory footprint
        It is also called internally whenever the user changes the data in ps_data
        """
        for col_name in self._ps_data.columns:
            if not col_name in ps_cfg.required_columns:
                self._ps_data.drop(columns=col_name, inplace=True)

        self.twiss_parameters = {}
        self.energy_stats = {}

    def assess_density_versus_r(self, Rvals=None, verbose=True, beam_direction='z'):
        """
        Assess how many particles are in a given radius

        :param Rvals: list of rvals to assess in mm, e.g. np.linspace(0, 2, 21)
        :param verbose: prints data to screen if True
        :param beam_direction: main direction in which beam is travelling. 'x', 'y', or 'z' (default)
        :type beam_direction: str, optional
        :return density_data: a dataframe containing the roi vals and the proportion of particles inside
        """

        if self.ps_data['weight'].max() > 1:
            warnings.warn('AssessDensityVersusR function ignores particle weights')

        if Rvals is None:
            # pick a default
            Rvals = np.linspace(0, 2, 21)
        if not isinstance(Rvals, (list, np.ndarray)):
            Rvals = list(Rvals)
        if beam_direction == 'x':
            r = np.sqrt(self.ps_data['z [mm]']**2 + self.ps_data['y [mm]']**2)
        elif beam_direction == 'y':
            r = np.sqrt(self.ps_data['x [mm]']**2 + self.ps_data['z [mm]']**2)
        elif beam_direction == 'z':
            r = np.sqrt(self.ps_data['x [mm]']**2 + self.ps_data['y [mm]']**2)


        numparticles = self.ps_data['x [mm]'].shape[0]
        rad_prop = []

        for rcheck in Rvals:
            Rind = r < rcheck
            rad_prop.append(np.count_nonzero(Rind) * 100 / numparticles)

        density_data = pd.DataFrame({'roi [mm]': Rvals, '% particles inside': rad_prop})
        if verbose:
            print(density_data)
        return density_data




# PhaseSpace.py
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import os, sys
import logging
import inspect
import pandas as pd

logging.basicConfig(level=logging.WARNING)
import warnings
from scipy.stats import gaussian_kde
from scipy import constants
from time import perf_counter
import json
from pathlib import Path
import ParticlePhaseSpace.__config as cf
from ParticlePhaseSpace.DataLoaders import _DataImportersBase
from ParticlePhaseSpace import utilities as ps_util
from ParticlePhaseSpace import DataLoaders

class FigureSpecs:
    """
    Thought this might be the easiest way to ensure universal parameters accross all figures
    """
    LabelFontSize = 14
    TitleFontSize = 16
    Font = 'serif'
    AxisFontSize = 14
    TickFontSize = 14


class ParticlePhaseSpace:
    """
    """

    def __init__(self, ps_data):

        if not isinstance(ps_data, _DataImportersBase):
            raise TypeError(f'ParticlePhaseSpace must be instantiated with a valid object'
                            f'from DataLoaders, not {type(ps_data)}')
        self.ps_data = ps_data.data
        self.twiss_parameters = {}

    def __call__(self, particle_list):
        """
        this function allows users to seperate the phase space based on particle types

        users should be able to specify strings or pdg codes.
        if string, then we should convert to pdg code.
        """
        if not isinstance(particle_list, list):
            particle_list = [particle_list]
        allowed_particles = list(cf.particle_properties.keys())
        for particle in particle_list:
            if not particle in allowed_particles:
                raise Exception(f'particle type {particle} is unknown')
        particle_data_sets = []
        for particle in particle_list:
            pdg_code = cf.particle_properties[particle]['pdg_code']
            particle_data = self.ps_data.loc[self.ps_data['particle type [pdg_code]'] == pdg_code].copy(deep=True)
            particle_data.reset_index(inplace=True, drop=True)
            # delete any non required columns
            for col_name in particle_data.columns:
                if not col_name in cf.required_columns:
                    particle_data.drop(columns=col_name, inplace=True)
            # create a new instance of _DataImportersBase based on particle_data
            particle_data_loader = DataLoaders.LoadPandasData(particle_data)
            particle_instance = ParticlePhaseSpace(particle_data_loader)
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
        new_data = pd.concat([self.ps_data, other.ps_data])

        for col_name in new_data.columns:
            if not col_name in cf.required_columns:
                new_data.drop(columns=col_name, inplace=True)
        new_data_loader = DataLoaders.LoadPandasData(new_data)
        new_instance = ParticlePhaseSpace(new_data_loader)
        return new_instance

    def __sub__(self, other):
        """
        subtract phase space (remove every particle in other from self)
        """
        new_data = pd.merge(self.ps_data, other.ps_data, how='outer', indicator=True)\
            .query("_merge != 'both'")\
            .drop('_merge', axis=1)\
            .reset_index(drop=True)
        for col_name in new_data.columns:
            if not col_name in cf.required_columns:
                new_data.drop(columns=col_name, inplace=True)
        new_data_loader = DataLoaders.LoadPandasData(new_data)
        new_instance = ParticlePhaseSpace(new_data_loader)
        return new_instance

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

    def _assert_single_species_status(self):
        """
        raises an error if phase space is not single specied
        """
        n_unique_particle_species = len(np.unique(self.ps_data['particle type [pdg_code]']))
        if n_unique_particle_species > 1:
            raise AttributeError(f'{inspect.stack()[1][3]} can only be used on single species phase spaces;'
                                 f'the current phase space contains {n_unique_particle_species} different particle species')

    def _check_ps_data_format(self):
        """
        check that the phase space remains consisten with what is mandated in __config
        :return:
        """

        all_allowed_columns = cf.required_columns + cf.allowed_columns
        for col_name in self.ps_data.columns:
            if not col_name in all_allowed_columns:
                raise AttributeError(f'non allowed column name {col_name} in ps_data')

    def _calculate_energy_statistics(self, ps_data):
        meanEnergy, stdEnergy = self._weighted_avg_and_std(ps_data['Ek [MeV]'], ps_data['weight'])
        medianEnergy = self._weighted_median(ps_data['Ek [MeV]'], ps_data['weight'])
        EnergySpreadSTD = np.std(np.multiply(ps_data['Ek [MeV]'], ps_data['weight']))
        q75, q25 = self._weighted_quantile(ps_data['Ek [MeV]'], [0.25, 0.75], sample_weight=ps_data['weight'])
        EnergySpreadIQR = q25 - q75
        return meanEnergy, medianEnergy, EnergySpreadSTD, EnergySpreadIQR

    # public methods

    def plot_energy_histogram(self, n_bins=100, title=None):
        """
        generate a histogram plot of paritcle energies.
        Each particle spcies present in the phase space is overlaid
        on the same plot.

        :param n_bins: number of bins in histogram
        :param title: title of histogram
        :return: None
        """
        Efig, axs = plt.subplots()
        if not 'Ek [MeV]' in self.ps_data.columns:
            self.fill_kinetic_E()
        unique_particles = self.ps_data['particle type [pdg_code]'].unique()
        legend = []
        for particle in unique_particles:
            legend.append(cf.particle_properties[particle]['name'])
            ind = self.ps_data['particle type [pdg_code]'] == particle
            Eplot = self.ps_data['Ek [MeV]'][ind]
            n, bins, patches = axs.hist(Eplot, bins=n_bins, weights=self.ps_data['weight'][ind], alpha=.5)

        axs.set_xlabel('Energy [MeV]', fontsize=FigureSpecs.LabelFontSize)
        axs.set_ylabel('N counts', fontsize=FigureSpecs.LabelFontSize)
        if title:
            axs.set_title(title, fontsize=FigureSpecs.TitleFontSize)
        axs.tick_params(axis="y", labelsize=FigureSpecs.TickFontSize)
        axs.tick_params(axis="x", labelsize=FigureSpecs.TickFontSize)
        axs.legend(legend)
        plt.tight_layout()
        plt.show()

    def plot_position_histogram(self, n_bins=100, alpha=0.5):
        """
        plot a histogram of particle positions in x, y, z.
        a new histogram is generated for each particle species.
        histograms are overlaid.

        :return: None
        """
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(15, 5)
        unique_particles = self.ps_data['particle type [pdg_code]'].unique()
        legend = []
        for particle in unique_particles:
            legend.append(cf.particle_properties[particle]['name'])
            ind = self.ps_data['particle type [pdg_code]'] == particle
            x_plot = self.ps_data['x [mm]'][ind]
            y_plot = self.ps_data['y [mm]'][ind]
            z_plot = self.ps_data['z [mm]'][ind]
            axs[0].hist(x_plot, bins=n_bins, weights=self.ps_data['weight'][ind], alpha=alpha)
            axs[1].hist(y_plot, bins=n_bins, weights=self.ps_data['weight'][ind], alpha=alpha)
            axs[2].hist(z_plot, bins=n_bins, weights=self.ps_data['weight'][ind], alpha=alpha)

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
                                xlim=None, ylim=None):
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
        :return:
        """

        unique_particles = self.ps_data['particle type [pdg_code]'].unique()
        fig, axs = plt.subplots(1, len(unique_particles), squeeze=False)
        fig.set_size_inches(5*len(unique_particles), 5)
        n_axs = 0
        for particle in unique_particles:
            ind = self.ps_data['particle type [pdg_code]'] == particle
            ps_data = self.ps_data.loc[ind]
            axs_title = cf.particle_properties[particle]['name']
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
                k = gaussian_kde(xy, weights=self.ps_data['weight'][ind])
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
                axs[0, n_axs].scatter(x_data, y_data, s=1, c=self.ps_data['weight'][ind])
            axs[0, n_axs].set_aspect(1)
            axs[0, n_axs].set_title(axs_title, fontsize=FigureSpecs.TitleFontSize)
            axs[0, n_axs].set_xlabel(x_label, fontsize=FigureSpecs.LabelFontSize)
            axs[0, n_axs].set_ylabel(y_label, fontsize=FigureSpecs.LabelFontSize)
            if xlim:
                axs[0, n_axs].set_xlim(xlim)
            if ylim:
                axs[0, n_axs].set_ylim(ylim)
            n_axs = n_axs+1

        plt.tight_layout()
        plt.show()

    def report(self):
        """
        print a sumary of the phase space to the screen.
        """
        if not 'Ek [MeV]' in self.ps_data.columns:
            self.fill_kinetic_E()
        print(f'total number of particles in phase space: {self.ps_data.shape[0]}')
        unique_particles = self.ps_data['particle type [pdg_code]'].unique()
        print(f'number of unique particle species: {len(unique_particles): d}')
        for particle in unique_particles:
            ind = self.ps_data['particle type [pdg_code]'] == particle
            ps_data = self.ps_data.loc[ind]
            meanEnergy, medianEnergy, EnergySpreadSTD, EnergySpreadIQR = self._calculate_energy_statistics(ps_data)
            print(f'    {np.count_nonzero(ind): d} {cf.particle_properties[particle]["name"]}'
                  f'\n        mean energy: {meanEnergy: 1.2f} MeV'
                  f'\n        median energy: {medianEnergy: 1.2f} MeV'
                  f'\n        Energy spread IQR: {EnergySpreadIQR: 1.2f} MeV'
                  f'\n        min energy {self.ps_data.loc[ind]["Ek [MeV]"].min()} MeV'
                  f'\n        max energy {self.ps_data.loc[ind]["Ek [MeV]"].max()} MeV')

    def fill_kinetic_E(self):
        """
        adds kinetic energy into self.ps_data
        """
        if not hasattr(self,'_rest_masses'):
            self.fill_rest_mass()
        Totm = np.sqrt(self.ps_data['px [MeV/c]'] ** 2 + self.ps_data['py [MeV/c]'] ** 2 + self.ps_data['pz [MeV/c]'] ** 2)
        TOT_E = np.sqrt(Totm ** 2 + self.ps_data['rest mass [MeV/c^2]'] ** 2)
        Kin_E = np.subtract(TOT_E, self.ps_data['rest mass [MeV/c^2]'])
        self.ps_data['Ek [MeV]'] = Kin_E
        self._check_ps_data_format()

    def fill_rest_mass(self):
        """
        add rest mass data to self.ps_data
        :return: 
        """
        self.ps_data['rest mass [MeV/c^2]'] = ps_util.get_rest_masses_from_pdg_codes(self.ps_data['particle type [pdg_code]'])
        self._check_ps_data_format()

    def fill_velocity(self):
        """
        add velocities into self.ps_data
        """
        if not 'rest mass [MeV/c^2]' in self.ps_data.columns:
            self.fill_rest_mass()
        if not 'gamma' in self.ps_data.columns:
            self.fill_beta_and_gamma()
        self.ps_data['vx [m/s]'] = np.divide(self.ps_data['px [MeV/c]'], (self.ps_data['gamma'] * self.ps_data['rest mass [MeV/c^2]']))
        self.ps_data['vy [m/s]'] = np.divide(self.ps_data['py [MeV/c]'], (self.ps_data['gamma'] * self.ps_data['rest mass [MeV/c^2]']))
        self.ps_data['vz [m/s]'] = np.divide(self.ps_data['pz [MeV/c]'], (self.ps_data['gamma'] * self.ps_data['rest mass [MeV/c^2]']))
        self._check_ps_data_format()
    
    def fill_beta_and_gamma(self):
        """
        add the relatavistic beta and gamma factors into self.ps_data
        """
        if not 'Ek [MeV]' in self.ps_data.columns:
            self.fill_kinetic_E()
        if not 'rest mass [MeV/c^2]' in self.ps_data.columns:
            self.fill_rest_mass()
        TOT_P = np.sqrt(self.ps_data['px [MeV/c]'] ** 2 + self.ps_data['py [MeV/c]'] ** 2 + self.ps_data['pz [MeV/c]'] ** 2)
        self.ps_data['beta'] = np.divide(TOT_P, self.ps_data['Ek [MeV]'] + self.ps_data['rest mass [MeV/c^2]'])
        self.ps_data['gamma'] = 1 / np.sqrt(1 - np.square(self.ps_data['beta']))
        self._check_ps_data_format()

    def get_downsampled_phase_space(self, downsample_factor=10):
        """
        return a new phase space object which randomlt samples from the larger phase space.
        the new phase space has size 'original data/downsample_factor'. the data is shuffled
        before being randomly sampled.

        :param downsample_factor: the factor to downsample the phase space by
        """
        new_data = self.ps_data.sample(frac=1).reset_index(drop=True)  # this shuffles the data
        new_data = new_data.sample(frac=1/downsample_factor, ignore_index=True)
        for col_name in new_data.columns:
            if not col_name in cf.required_columns:
                new_data.drop(columns=col_name, inplace=True)
        new_data_loader = DataLoaders.LoadPandasData(new_data)
        new_instance = ParticlePhaseSpace(new_data_loader)
        return new_instance

    def calculate_twiss_parameters(self, beam_direction='z'):
        """
        Calculate the twiss parameters
        """
        self._assert_single_species_status()
        if beam_direction == 'x':
            intersection_columns = ['x [mm]', 'px [MeV/c]']
            direction_columns = [['x [mm]', 'px [MeV/c]'], ['y [mm]', 'py [MeV/c]']]
        elif beam_direction == 'y':
            intersection_columns = ['y [mm]', 'py [MeV/c]']
            direction_columns = [['x [mm]', 'px [MeV/c]'], ['z [mm]', 'pz [MeV/c]']]
        elif beam_direction == 'z':
            intersection_columns = ['z [mm]', 'pz [MeV/c]']
            direction_columns = [['x [mm]', 'px [MeV/c]'], ['y [mm]', 'py [MeV/c]']]
        else:
            raise NotImplementedError('beam direction must be "x", "y", or "z"')

        for calc_dir in direction_columns:
            x2 = np.average(np.square(self.ps_data[calc_dir[0]]), weights=self.ps_data['weight'])
            xp = np.divide(self.ps_data[calc_dir[0]], self.ps_data[intersection_columns[1]]) * 1e3
            xp2 = np.average(np.square(xp), weights=self.ps_data['weight'])
            x_xp = np.average(np.multiply(self.ps_data[calc_dir[0]], xp), weights=self.ps_data['weight'])

            epsilon = np.sqrt((x2 * xp2) - (x_xp ** 2)) * np.pi
            alpha = -x_xp / epsilon
            beta = x2 / epsilon
            gamma = xp2 / epsilon

            self.twiss_parameters[calc_dir[0][0]] = {'epsilon': epsilon,
                                                     'alpha': alpha,
                                                     'beta': beta,
                                                     'gamma': gamma}

    def print_twiss_parameters(self, file_name=None, beam_direction='z'):
        """
        prints the twiss parameters if they exist
        they are always printed to the screen.
        if filename is specified, they are also saved to file as json

        :param file_name: filename to write twiss data to. should be absolute
            path to an existing directory
        :return:
        """
        self.calculate_twiss_parameters(beam_direction=beam_direction)
        twiss_data = pd.DataFrame(self.twiss_parameters)
        print(twiss_data)
        if file_name:
            file_name = Path(file_name)
            if not file_name.parent.is_dir():
                raise NotADirectoryError(f'{file_name.parent} is not a directory')
            if not file_name.suffix == '.json':
                file_name = file_name.parent / (file_name.name + '.json')
            with open('data.json', 'w') as fp:
                json.dump(self.twiss_parameters, fp)

    def project_particles(self, beam_direction='z', distance=100):
        """
        Update the positions of each particle by projecting it forward/back by distance.

        This serves as a crude approximation to more advanced particle transport codes, but gives
        an indication of where the particles would end up in the absence of other forces.
        When this function is recalled, any dervied quantities will be deleted as it is too hard to check
        that they all remain correct.

        :param direction: the direction to project in. 'x', 'y', or 'z'
        :param distance: how far to project
        :return: None
        """
        if not 'vx [m/s]' in self.ps_data.columns:
            self.fill_velocity()

        if beam_direction == 'z':
            self.ps_data['x [mm]'] = self.ps_data['x [mm]'] + np.divide(self.ps_data['vx [m/s]'], self.ps_data['vz [m/s']) * distance
            self.ps_data['y [mm]'] = self.ps_data['y [mm]'] + np.divide(self.ps_data['vy [m/s]'], self.ps_data['vz [m/s']) * distance
            self.ps_data['z [mm]'] = self.ps_data['z [mm]'] + distance
        else:
            raise NotImplementedError('havent coded the other directions yet')

        self.reset_phase_space()  # safest to get rid of any derived quantities

    def reset_phase_space(self):
        """
        reduce self.ps_data to only the required columns
        delete any other dervied quantities such as twiss parameters
        this can be called whenever you want to reduce the memory footprint, or whenever
        you have perfored some operation which may have invalidiated derived metrics
        """
        for col_name in self.ps_data.columns:
            if not col_name in cf.required_columns:
                self.ps_data.drop(columns=col_name, inplace=True)

        self.twiss_parameters = {}

    def PlotPhaseSpaceX(self, beam_direction='z'):

        self.calculate_twiss_parameters(beam_direction=beam_direction)
        if self.weight.max() > 1:
            warnings.warn('this plot does not take into account particle weights')
        fig, axs = plt.subplots(nrows=1, ncols=2)

        #plot phase elipse
        xq = np.linspace(min(self.x), max(self.x), 1000)
        xpq = np.linspace(min(self.xp), max(self.xp), 1000)
        [ElipseGridx, ElipseGridy] = np.meshgrid(xq, xpq)
        EmittanceGrid = (self.twiss_gamma * np.square(ElipseGridx)) + \
                        (2 * self.twiss_alpha * np.multiply(ElipseGridx, ElipseGridy)) + \
                        (self.twiss_beta * np.square(ElipseGridy))
        tol = .01 * self.twiss_epsilon
        Elipse = (EmittanceGrid >= self.twiss_epsilon - tol) & (EmittanceGrid <= self.twiss_epsilon + tol)
        ElipseIndex = np.where(Elipse == True)
        elipseX = ElipseGridx[ElipseIndex]
        elipseY = ElipseGridy[ElipseIndex]
        plt.scatter(elipseX, elipseY, s=1, c='r')
        xmin, xmax, ymin, ymax = plt.axis()

        plt.scatter(self.x, self.xp, s=1, marker='.')
        plt.xlabel('X [mm]')
        plt.ylabel("X' [mrad]")
        plt.ylim([ymin, ymax])
        plt.xlim([xmin, xmax])
        TitleString = "\u03C0\u03B5: %1.1f mm mrad, \u03B1: %1.1f, \u03B2: %1.1f, \u03B3: %1.1f" % \
                      (self.twiss_epsilon, self.twiss_alpha, self.twiss_beta, self.twiss_gamma)
        plt.title(TitleString)

        plt.grid(True)
        plt.show()


    ###############################


    def AssessDensityVersusR(self, Rvals=None):
        """
        Crude code to assess how many particles are in a certain radius

        If ROI = None,  then all particles are assessed.
        Otherwise, use ROI = [zval, radius] to only include particles that would be within radius r at distance z from
        the read in location
        """

        if self.weight.max() > 1:
            warnings.warn('The AssessDensityVersusR function has not been updated for weighted particles')
        if Rvals is None:
            # pick a default
            Rvals = np.linspace(0, 2, 21)

        r = np.sqrt(self.x ** 2 + self.y ** 2)
        numparticles = self.x.shape[0]
        rad_prop = []

        if self.verbose:
            if self.ROI == None:
                print(f'Assessing particle density versus R for all particles')
            else:
                print(f'Assessing particle density versus R for particles projected to be within a radius of'
                      f' {self.ROI[1]} at a distance of {self.ROI[0]}')

        for rcheck in Rvals:
            if self.ROI == None:
                Rind = r <= rcheck
                rad_prop.append(np.count_nonzero(Rind) * 100 / numparticles)
            else:
                # apply the additional ROI filter by projecting x,y to the relevant z position
                Xproj = np.multiply(self.ROI[0], np.divide(self.px, self.pz)) + self.x
                Yproj = np.multiply(self.ROI[0], np.divide(self.py, self.pz)) + self.y
                Rproj = np.sqrt(Xproj ** 2 + Yproj ** 2)
                ROIind = Rproj <= self.ROI[1]

                Rind = r <= rcheck
                ind = np.multiply(ROIind, Rind)
                rad_prop.append(np.count_nonzero(ind) * 100 / numparticles)

        self.rad_prop = rad_prop
import numpy as np
from matplotlib import pyplot as plt
import logging
import pandas as pd

logging.basicConfig(level=logging.WARNING)
import warnings
from scipy.stats import gaussian_kde
from scipy import constants
from scipy.spatial.transform import Rotation as R
import json
from pathlib import Path
from time import perf_counter
import ParticlePhaseSpace.__phase_space_config__ as ps_cfg
import ParticlePhaseSpace.__particle_config__ as particle_cfg
from ParticlePhaseSpace.DataLoaders import _DataLoadersBase
from ParticlePhaseSpace import utilities as ps_util
from ParticlePhaseSpace import DataLoaders
from ParticlePhaseSpace import UnitSet, ParticlePhaseSpaceUnits
from abc import ABC


class _FigureSpecs:
    """
    Thought this might be the easiest way to ensure universal parameters accross all figures
    """
    LabelFontSize = 14
    TitleFontSize = 16
    Font = 'serif'
    AxisFontSize = 14
    TickFontSize = 14


class _PhaseSpace_MethodHolder(ABC):
    """
    base class for sub-classes of PhaseSpace
    """

    def __init__(self, PS):
        self._PS = PS

    def get_methods(self):
        """
        print all public methods (except this one)
        :return:
        """
        all_attributes = filter(lambda a: not a.startswith('__'), dir(self))

        self._attributes = []
        for attribute in all_attributes:
            if attribute[0] == '_':
                continue
            if attribute == 'get_methods':
                continue
            print(attribute)


class _Plots(_PhaseSpace_MethodHolder):

    def energy_hist_1D(self, n_bins: int = 100, grid: bool = False):  # pragma: no cover
        """
        generate a histogram plot of paritcle energies.
        Each particle spcies present in the phase space is overlaid  on the same plot.

        :param n_bins: number of bins in histogram
        :type n_bins: int, optional
        :param grid: turns grid on/off
        :type grid: bool, optional
        :return: None
        """
        Efig, axs = plt.subplots()
        if not self._PS.columns['Ek'] in self._PS._ps_data.columns:
            self._PS.fill.kinetic_E()
        legend = []
        for particle in self._PS._unique_particles:
            legend.append(particle_cfg.particle_properties[particle]['name'])
            ind = self._PS._ps_data['particle type [pdg_code]'] == particle
            Eplot = self._PS._ps_data[self._PS.columns['Ek']][ind]
            n, bins, patches = axs.hist(Eplot, bins=n_bins, weights=self._PS._ps_data['weight'][ind], alpha=.5)

        axs.set_xlabel(self._PS.columns['Ek'], fontsize=_FigureSpecs.LabelFontSize)
        axs.set_ylabel('N counts', fontsize=_FigureSpecs.LabelFontSize)
        axs.tick_params(axis="y", labelsize=_FigureSpecs.TickFontSize)
        axs.tick_params(axis="x", labelsize=_FigureSpecs.TickFontSize)
        if grid:
            axs.grid()
        axs.legend(legend)

        plt.tight_layout()
        plt.show()

    def position_hist_1D(self, n_bins: int = 100, alpha: float = 0.5, grid: bool = False):  # pragma: no cover
        """
        plot a histogram of particle positions in x, y, z.
        a new histogram is generated for each particle species.
        histograms are overlaid.

        :param n_bins:  number of bins in histogram
        :param alpha: controls transparency of each histogram.
        :param grid: turns grid on/off
        :type grid: bool, optional
        """
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(15, 5)
        legend = []
        for particle in self._PS._unique_particles:
            legend.append(particle_cfg.particle_properties[particle]['name'])
            ind = self._PS._ps_data['particle type [pdg_code]'] == particle
            x_plot = self._PS._ps_data[self._PS.columns['x']][ind]
            y_plot = self._PS._ps_data[self._PS.columns['y']][ind]
            z_plot = self._PS._ps_data[self._PS.columns['z']][ind]
            axs[0].hist(x_plot, bins=n_bins, weights=self._PS._ps_data['weight'][ind], alpha=alpha)
            axs[1].hist(y_plot, bins=n_bins, weights=self._PS._ps_data['weight'][ind], alpha=alpha)
            axs[2].hist(z_plot, bins=n_bins, weights=self._PS._ps_data['weight'][ind], alpha=alpha)

        axs[0].set_xlabel(self._PS.columns['x'])
        axs[0].set_ylabel('counts')
        axs[0].set_title(self._PS.columns['x'])
        axs[0].legend(legend)

        axs[1].set_xlabel(self._PS.columns['y'])
        axs[1].set_ylabel('counts')
        axs[1].set_title(self._PS.columns['y'])

        axs[2].set_xlabel(self._PS.columns['z'])
        axs[2].set_ylabel('counts')
        axs[2].set_title(self._PS.columns['z'])

        if grid:
            axs[0].grid()
            axs[1].grid()
            axs[2].grid()

        plt.tight_layout()
        plt.show()

    def momentum_hist_1D(self, n_bins: int = 100, alpha: float = 0.5, grid: bool = False):  # pragma: no cover
        """
        plot a histogram of particle momentum in x, y, z.
        a new histogram is generated for each particle species.
        histograms are overlaid.

        :param n_bins:  number of bins in histogram
        :param alpha: controls transparency of each histogram.
        :param grid: turns grid on/off
        :type grid: bool, optional
        """
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(15, 5)
        legend = []
        for particle in self._PS._unique_particles:
            legend.append(particle_cfg.particle_properties[particle]['name'])
            ind = self._PS._ps_data['particle type [pdg_code]'] == particle
            x_plot = self._PS._ps_data[self._PS.columns['px']][ind]
            y_plot = self._PS._ps_data[self._PS.columns['py']][ind]
            z_plot = self._PS._ps_data[self._PS.columns['pz']][ind]
            axs[0].hist(x_plot, bins=n_bins, weights=self._PS._ps_data['weight'][ind], alpha=alpha)
            axs[1].hist(y_plot, bins=n_bins, weights=self._PS._ps_data['weight'][ind], alpha=alpha)
            axs[2].hist(z_plot, bins=n_bins, weights=self._PS._ps_data['weight'][ind], alpha=alpha)

        axs[0].set_xlabel(self._PS.columns['px'])
        axs[0].set_ylabel('counts')
        axs[0].set_title(self._PS.columns['px'])
        axs[0].legend(legend)

        axs[1].set_xlabel(self._PS.columns['py'])
        axs[1].set_ylabel('counts')
        axs[1].set_title(self._PS.columns['py'])

        axs[2].set_xlabel(self._PS.columns['pz'])
        axs[2].set_ylabel('counts')
        axs[2].set_title(self._PS.columns['pz'])

        if grid:
            axs[0].grid()
            axs[1].grid()
            axs[2].grid()

        plt.tight_layout()
        plt.show()

    def particle_positions_scatter_2D(self, beam_direction: str = 'z', weight_position_plot: bool = False,
                                      grid: bool = True, xlim=None, ylim=None):  # pragma: no cover
        """
        produce a scatter plot of particle positions.
        one plot is produced for each unique species.

        :param beam_direction: the direction the beam is travelling in. "x", "y", or "z" (default)
        :type beam_direction: str, optional
        :param weight_position_plot: if True, a gaussian kde is used to weight the particle
            positions. This can produce very informative and useful plots, but also be very slow.
            If it is slow, you could try downsampling the phase space first using get_downsampled_phase_space
        :type weight_position_plot: bool
        :param grid: turns grid on/off
        :type grid: bool, optional
        :param xlim: set the xlim for all plots, e.g. [-2,2]
        :type xlim: list or None, optional
        :param ylim: set the ylim for all plots, e.g. [-2,2]
        :type ylim: list or None, optional
        :return: None
        """
        fig, axs = plt.subplots(1, len(self._PS._unique_particles), squeeze=False)
        fig.set_size_inches(5 * len(self._PS._unique_particles), 5)
        n_axs = 0
        for particle in self._PS._unique_particles:
            ind = self._PS._ps_data['particle type [pdg_code]'] == particle
            ps_data = self._PS._ps_data.loc[ind]
            axs_title = particle_cfg.particle_properties[particle]['name']
            if beam_direction == 'x':
                x_data = ps_data[self._PS.columns['y']]
                y_data = ps_data[self._PS.columns['z']]
                x_label = self._PS.columns['y']
                y_label = self._PS.columns['z']
            elif beam_direction == 'y':
                x_data = ps_data[self._PS.columns['x']]
                y_data = ps_data[self._PS.columns['z']]
                x_label = self._PS.columns['x']
                y_label = self._PS.columns['z']
            elif beam_direction == 'z':
                x_data = ps_data[self._PS.columns['x']]
                y_data = ps_data[self._PS.columns['y']]
                x_label = self._PS.columns['x']
                y_label = self._PS.columns['y']
            else:
                raise NotImplementedError('beam_direction must be "x", "y", or "z"')

            if weight_position_plot:
                _kde_data_grid = 150 ** 2
                print('generating weighted scatter plot...can be slow...')
                xy = np.vstack([x_data, y_data])
                k = gaussian_kde(xy, weights=self._PS._ps_data['weight'][ind])
                down_sample_factor = np.round(x_data.shape[0] / _kde_data_grid)
                if down_sample_factor > 1:
                    # in this case we can downsample for display
                    print(f'down sampling kde data by factor of {down_sample_factor}')
                    rng = np.random.default_rng()
                    rng.shuffle(xy)  # operates in place for some confusing reason
                    xy = rng.choice(xy, int(x_data.shape[0] / down_sample_factor), replace=False, axis=1, shuffle=False)
                z = k(xy)
                z = z / max(z)
                axs[0, n_axs].scatter(xy[0], xy[1], c=z, s=1)
            else:
                axs[0, n_axs].scatter(x_data, y_data, s=1, c=self._PS._ps_data['weight'][ind])
            axs[0, n_axs].set_aspect('equal')
            axs[0, n_axs].set_aspect(1)
            axs[0, n_axs].set_title(axs_title, fontsize=_FigureSpecs.TitleFontSize)
            axs[0, n_axs].set_xlabel(x_label, fontsize=_FigureSpecs.LabelFontSize)
            axs[0, n_axs].set_ylabel(y_label, fontsize=_FigureSpecs.LabelFontSize)
            if grid:
                axs[0, n_axs].grid()
            if xlim:
                axs[0, n_axs].set_xlim(xlim)
            if ylim:
                axs[0, n_axs].set_ylim(ylim)
            n_axs = n_axs + 1

        plt.tight_layout()
        plt.show()

    def particle_positions_hist_2D(self, beam_direction: str = 'z', quantity: str = 'intensity',
                                   grid: bool = True, log_scale: bool = False, bins: int = 100,
                                   normalize: bool = True, xlim=None, ylim=None, vmin=None,
                                   vmax=None, ):  # pragma: no cover
        """
        plot a 2D histogram of data, either of accumulated number of particules or accumulated energy

        :param beam_direction: the direction the beam is travelling in. "x", "y", or "z" (default)
        :type beam_direction: str, optional
        :param xlim: set the xlim for all plots, e.g. [-2,2]
        :type xlim: list, optional
        :param ylim: set the ylim for all plots, e.g. [-2,2]
        :type ylim: list, optional
        :param quantity: quantity to accumulate; either 'intensity' or 'energy
        :type quantity: str
        :param grid: turns grid on/off
        :type grid: bool, optional
        :param bins: number of bins in X/Y direction. n_pixels = bins ** 2
        :type bins: int, optional
        :param vmin: minimum color range
        :type vmin: float, optional
        :param vmax: maximum color range
        :type vmax: float, optional
        :param log_scale: if True, log scale is used
        :type log_scale: bool, optional
        :param normalize: if True, data is normalized to 0-100 - otherwise raw values are used
        :type normalize: bool, optional
        :return: None
        """
        if log_scale:
            _scale = 'log'
        else:
            _scale = None
        fig, axs = plt.subplots(1, len(self._PS._unique_particles), squeeze=False)
        fig.set_size_inches(5 * len(self._PS._unique_particles), 5)
        n_axs = 0
        if not beam_direction in ['x', 'y', 'z']:
            raise NotImplementedError('beam_direction must be "x", "y", or  "z"')
        if not quantity in ['intensity', 'energy']:
            raise NotImplementedError('quantity must be "intensity" or "energy"')

        if (not self._PS.columns['Ek'] in self._PS._ps_data.columns):
            self._PS.fill.kinetic_E()
        for particle in self._PS._unique_particles:
            ind = self._PS._ps_data['particle type [pdg_code]'] == particle
            ps_data = self._PS._ps_data.loc[ind]
            if beam_direction == 'x':
                x_data = ps_data[self._PS.columns['y']]
                y_data = ps_data[self._PS.columns['z']]
                _x_label = self._PS.columns['y']
                _y_label = self._PS.columns['z']
            elif beam_direction == 'y':
                x_data = ps_data[self._PS.columns['x']]
                y_data = ps_data[self._PS.columns['z']]
                _x_label = self._PS.columns['x']
                _y_label = self._PS.columns['z']
            elif beam_direction == 'z':
                x_data = ps_data[self._PS.columns['x']]
                y_data = ps_data[self._PS.columns['y']]
                _x_label = self._PS.columns['x']
                _y_label = self._PS.columns['y']
            else:
                raise NotImplementedError('beam direction must be one of "x", "y" or "z"')
            if xlim is None:
                if beam_direction == 'x':
                    xlim = [ps_data[self._PS.columns['y']].min(), ps_data[self._PS.columns['y']].max()]
                elif beam_direction == 'y' or beam_direction == 'z':
                    xlim = [ps_data[self._PS.columns['x']].min(), ps_data[self._PS.columns['x']].max()]
            if ylim is None:
                if beam_direction == 'z':
                    ylim = [ps_data[self._PS.columns['y']].min(), ps_data[self._PS.columns['y']].max()]
                elif beam_direction == 'x' or beam_direction == 'y':
                    ylim = [ps_data[self._PS.columns['z']].min(), ps_data[self._PS.columns['z']].max()]
            if quantity == 'intensity':
                _title = f"n_particles intensity;\n{particle_cfg.particle_properties[particle]['name']}"
                _weight = ps_data['weight']
            elif quantity == 'energy':
                _title = f"energy intensity;\n{particle_cfg.particle_properties[particle]['name']}"
                _weight = np.multiply(ps_data['weight'], ps_data[self._PS.columns['Ek']])
            X = np.linspace(xlim[0], xlim[1], bins)
            Y = np.linspace(ylim[0], ylim[1], bins)
            h, xedges, yedges = np.histogram2d(x_data,
                                               y_data,
                                               bins=[X, Y], weights=_weight)

            if normalize:
                try:
                    h = h * 100 / h.max()
                except ValueError:
                    pass
            _im1 = axs[0, n_axs].pcolormesh(xedges, yedges, h.T, cmap='inferno',
                                            norm=_scale, rasterized=False, vmin=vmin, vmax=vmax)

            fig.colorbar(_im1, ax=axs[0, n_axs])

            axs[0, n_axs].set_title(_title)
            axs[0, n_axs].set_xlabel(_x_label, fontsize=_FigureSpecs.LabelFontSize)
            axs[0, n_axs].set_ylabel(_y_label, fontsize=_FigureSpecs.LabelFontSize)
            axs[0, n_axs].set_aspect('equal')
            if grid:
                axs[0, n_axs].grid()
            n_axs = n_axs + 1
        plt.tight_layout()
        plt.show()

    def transverse_trace_space_scatter_2D(self, beam_direction: str = 'z', plot_twiss_ellipse: bool = True,
                                          grid: bool = True, xlim=None, ylim=None, ):  # pragma: no cover
        """
        Generate a scatter plot of x versus x'=px/pz and y versus y'=py/pz (these definitions are for
        beam_direction='z')

        :param beam_direction: main direction in which beam is travelling. 'x', 'y', or 'z' (default)
        :type beam_direction: str, optional
        :param plot_twiss_ellipse: if True, will overlay the RMS twiss ellipse onto the trace space
        :type plot_twiss_ellipse: bool, optional
        :param xlim: set xlim, e.g. [-2,2]
        :type xlim: list, optional
        :param ylim: set ylim, e.g. [-2,2]
        :type ylim: list, optional
        :param grid: turns grid on/off
        :type grid: bool, optional
        """

        self._PS.calculate_twiss_parameters(beam_direction=beam_direction)
        fig, axs = plt.subplots(nrows=len(self._PS._unique_particles), ncols=2, squeeze=False)
        row = 0
        for particle in self._PS._unique_particles:
            particle_name = particle_cfg.particle_properties[particle]['name']
            ind = self._PS._ps_data['particle type [pdg_code]'] == particle
            ps_data = self._PS._ps_data.loc[ind]
            x_data_1, div_data_1, x_label_1, y_label_1, title_1, weight, elipse_parameters_1, \
            x_data_2, div_data_2, x_label_2, y_label_2, title_2, elipse_parameters_2 = \
                self._PS._get_data_for_trace_space_plots(beam_direction, ps_data, particle_name)

            axs[row, 0].scatter(x_data_1, div_data_1, s=1, marker='.', c=weight)
            axs[row, 0].set_xlabel(x_label_1)
            axs[row, 0].set_ylabel(y_label_1)
            axs[row, 0].set_title(title_1)
            if plot_twiss_ellipse:
                twiss_X, twiss_Y = self._PS._get_ellipse_xy_points(elipse_parameters_1, x_data_1.min(), x_data_1.max(),
                                                                   div_data_1.min(), div_data_1.max())
                axs[row, 0].scatter(twiss_X, twiss_Y, c='r', s=2)
                # axs[row, 0].set_xlim([3*np.min(twiss_X), 3*np.max(twiss_X)])
                # axs[row, 0].set_ylim([3 * np.min(twiss_Y), 3 * np.max(twiss_Y)])

            if xlim:
                axs[row, 0].set_xlim(xlim)
            if ylim:
                axs[row, 0].set_ylim(ylim)
            if plot_twiss_ellipse:
                twiss_X, twiss_Y = self._PS._get_ellipse_xy_points(elipse_parameters_2, x_data_2.min(), x_data_2.max(),
                                                                   div_data_2.min(), div_data_2.max())
                axs[row, 1].scatter(twiss_X, twiss_Y, c='r', s=2)
            axs[row, 1].scatter(x_data_2, div_data_2, s=1, marker='.', c=weight)
            axs[row, 1].set_xlabel(x_label_2)
            axs[row, 1].set_ylabel(y_label_2)
            axs[row, 1].set_title(title_2)
            if xlim:
                axs[row, 1].set_xlim(xlim)
            if ylim:
                axs[row, 1].set_ylim(ylim)
            if grid:
                axs[row, 0].grid()
                axs[row, 1].grid()
            row = row + 1

        plt.tight_layout()
        plt.show()

    def transverse_trace_space_hist_2D(self, beam_direction: str = 'z', plot_twiss_ellipse: bool = True,
                                       grid: bool = True, bins: int = 100, log_scale: bool = True,
                                       normalize: bool = True,
                                       xlim=None, ylim=None, vmin=None, vmax=None, ):  # pragma: no cover
        """
        plot the intensity of the beam in trace space

        :param beam_direction: the direction the beam is travelling in. "x", "y", or "z" (default)
        :type beam_direction: str, optional
        :param xlim: set the xlim for all plots, e.g. [-2,2]
        :type xlim: list, optional
        :param ylim: set the ylim for all plots, e.g. [-2,2]
        :type ylim: list, optional
        :param plot_twiss_ellipse: if True, RMS ellipse from twiss parameters is overlaid.
        :type plot_twiss_ellipse: bool, optional
        :param grid: turns grid on/off
        :type grid: bool, optional
        :param log_scale: if True, log scale is used
        :type log_scale: bool, optional
        :param bins: number of bins in X/Y direction. n_pixels = bins ** 2
        :type bins: int, optional
        :param vmin: minimum color range
        :type vmin: float, optional
        :param vmax: maximum color range
        :type vmax: float, optional
        """
        if log_scale:
            _scale = 'log'
        else:
            _scale = None
        self._PS.calculate_twiss_parameters(beam_direction=beam_direction)
        fig, axs = plt.subplots(nrows=len(self._PS._unique_particles), ncols=2, squeeze=False)
        row = 0
        for particle in self._PS._unique_particles:
            particle_name = particle_cfg.particle_properties[particle]['name']
            ind = self._PS._ps_data['particle type [pdg_code]'] == particle
            ps_data = self._PS._ps_data.loc[ind]
            x_data_1, div_data_1, x_label_1, y_label_1, title_1, weight, elipse_parameters_1, \
            x_data_2, div_data_2, x_label_2, y_label_2, title_2, elipse_parameters_2 = \
                self._PS._get_data_for_trace_space_plots(beam_direction, ps_data, particle_name)
            # accumulate data
            if not xlim:
                xlim = [np.min([x_data_1, x_data_2]), np.max([x_data_1, x_data_2])]
            if not ylim:
                ylim = [np.min([div_data_1, div_data_2]), np.max([div_data_1, div_data_2])]

            X = np.linspace(xlim[0], xlim[1], bins)
            Y = np.linspace(ylim[0], ylim[1], bins)
            _extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
            h, xedges, yedges = np.histogram2d(x_data_1, div_data_1, bins=[X, Y],
                                               weights=ps_data[self._PS.columns['weight']])
            if normalize:
                h = h * 100 / h.max()
            _im1 = axs[row, 0].pcolormesh(xedges, yedges, h.T, cmap='inferno',
                                          norm=_scale, rasterized=False, vmin=vmin, vmax=vmax)
            fig.colorbar(_im1, ax=axs[row, 0])
            axs[row, 0].set_xlabel(x_label_1)
            axs[row, 0].set_ylabel(y_label_1)
            axs[row, 0].set_title(title_1)
            if plot_twiss_ellipse:
                twiss_X, twiss_Y = self._PS._get_ellipse_xy_points(elipse_parameters_1, x_data_1.min(), x_data_1.max(),
                                                                   div_data_1.min(), div_data_1.max())
                axs[row, 0].scatter(twiss_X, twiss_Y, c='r', s=2)
            axs[row, 0].set_xlim(xlim)
            axs[row, 0].set_ylim(ylim)
            axs[row, 0].set_aspect('auto')
            h, xedges, yedges = np.histogram2d(x_data_2, div_data_2, bins=[X, Y],
                                               weights=ps_data[self._PS.columns['weight']])
            if normalize:
                h = h * 100 / h.max()
            _im2 = axs[row, 1].pcolormesh(xedges, yedges, h.T, cmap='inferno',
                                          norm=_scale, rasterized=False, vmin=vmin, vmax=vmax)
            fig.colorbar(_im2, ax=axs[row, 1])
            if plot_twiss_ellipse:
                twiss_X, twiss_Y = self._PS._get_ellipse_xy_points(elipse_parameters_2, x_data_2.min(), x_data_2.max(),
                                                                   div_data_2.min(), div_data_2.max())
                axs[row, 1].scatter(twiss_X, twiss_Y, c='r', s=2)
            if grid:
                axs[row, 0].grid()
                axs[row, 1].grid()
            axs[row, 1].set_xlabel(x_label_2)
            axs[row, 1].set_ylabel(y_label_2)
            axs[row, 1].set_title(title_2)
            axs[row, 1].set_xlim(xlim)
            axs[row, 1].set_ylim(ylim)
            axs[row, 1].set_aspect('auto')
            row = row + 1

        plt.tight_layout()
        plt.show()

    def n_particles_v_time(self, n_bins: int = 100, grid: bool = False):  # pragma: no cover
        """
        basic plot of number of particles versus time; useful for quickly seperating out different bunches
        of electrons such that you can apply the 'filter_by_time' method

        :param n_bins: number of bins for histogram
        :type n_bins: int
        :param grid: turns grid on/off
        :type grid: bool, optional
        """
        plt.figure()
        plt.hist(self._PS._ps_data[self._PS.columns['time']], n_bins)
        plt.xlabel(f'time {self._PS._units.time.label}')
        plt.ylabel('N particles')
        if grid:
            plt.grid()
        plt.tight_layout()


class _Transform(_PhaseSpace_MethodHolder):
    """
    sub-class of PhaseSpace; methods for the transformation of the particle position data
    """

    def _return_position_update(self, new_x, new_y, new_z, in_place, new_px=None, new_py=None, new_pz=None):
        self._PS.reset_phase_space()
        if in_place:
            if new_x is not None:
                self._PS._ps_data[self._PS.columns['x']] = new_x
            if new_y is not None:
                self._PS._ps_data[self._PS.columns['y']] = new_y
            if new_z is not None:
                self._PS._ps_data[self._PS.columns['z']] = new_z
            if new_px is not None:
                self._PS._ps_data[self._PS.columns['px']] = new_px
            if new_py is not None:
                self._PS._ps_data[self._PS.columns['py']] = new_py
            if new_pz is not None:
                self._PS._ps_data[self._PS.columns['pz']] = new_pz
        else:
            ps_data = self._PS._ps_data.copy(deep=True)
            if new_x is not None:
                ps_data[self._PS.columns['x']] = new_x
            if new_y is not None:
                ps_data[self._PS.columns['y']] = new_y
            if new_z is not None:
                ps_data[self._PS.columns['z']] = new_z
            if new_px is not None:
                ps_data[self._PS.columns['px']] = new_px
            if new_py is not None:
                ps_data[self._PS.columns['py']] = new_py
            if new_pz is not None:
                ps_data[self._PS.columns['pz']] = new_pz

            new_data = DataLoaders.Load_PandasData(ps_data, units=self._PS._units)
            new_instance = PhaseSpace(new_data)
            return new_instance

    def translate(self, direction: str = 'z', distance: float = 100.0, in_place: bool = False):
        """
        :param beam_direction:
        :param distance:
        :param in_place:
        :return:
        """
        if direction == 'x':
            new_x = self._PS._ps_data[self._PS.columns['x']] + distance
            new_y = None
            new_z = None
        elif direction == 'y':
            new_x = None
            new_y = self._PS._ps_data[self._PS.columns['y']] + distance
            new_z = None
        elif direction == 'z':
            new_x = None
            new_y = None
            new_z = self._PS._ps_data[self._PS.columns['z']] + distance
        else:
            raise NotImplementedError('beam_direction must be "x", "y", or "z"')
        new_instance = self._return_position_update(new_x, new_y, new_z, in_place)
        return new_instance  # nb: None if in_place = True

    def rotate(self, rotation_axis: str = 'x', angle: float = 90.0, in_place: bool = False,
               rotate_momentum_vector=False):
        """
        rotate the phase space about an axis

        :param rotation_axis: axis to rotate; one of 'x','y','z'
        :type rotation_axis: str
        :param angle: angle to rotate in degrees
        :type angle: float
        :param in_place: operate in place or not
        :type in_place: bool
        :param rotate_momentum_vector: if True, will rotate the momentum data as well as the position data
        :type rotate_momentum_vector: bool
        :return: None or PhaseSpace
        """
        self._PS.reset_phase_space()
        r = R.from_euler(rotation_axis, angle, degrees=True)
        cols = [self._PS.columns['x'], self._PS.columns['y'], self._PS.columns['z']]
        rotated_points = r.apply(self._PS._ps_data[cols])

        if rotate_momentum_vector:
            cols = [self._PS.columns['px'], self._PS.columns['py'], self._PS.columns['pz']]
            rotated_mom = r.apply(self._PS._ps_data[cols])
            new_instance = self._return_position_update(rotated_points[:, 0],
                                                        rotated_points[:, 1],
                                                        rotated_points[:, 2],
                                                        in_place,
                                                        new_px=rotated_mom[:, 0],
                                                        new_py=rotated_mom[:, 1],
                                                        new_pz=rotated_mom[:, 2])
        else:
            new_instance = self._return_position_update(rotated_points[:, 0],
                                                        rotated_points[:, 1],
                                                        rotated_points[:, 2],
                                                        in_place)

        return new_instance  # nb: None if in_place = True

    def project(self, direction: str = 'z', distance: float = 100, in_place: bool = False):
        """
        Update the positions of each particle by projecting it forward/back by distance.

        This is valid in cases where the particles would end up in the absence of any forces or interactions.

        :param beam_direction: the direction to project in. 'x', 'y', or 'z'
        :type beam_direction: str, optional
        :param distance: how far to project in mm
        :type distance: float, optional
        :param in_place: if True, the existing PhaseSpace object has its data updated. if False,
            a new PhaseSpace object is returned
        :type in_place: bool, optional
        :return: new_instance: if in_place = False, a new PhaseSpace object is returned
        """
        if direction == 'x':
            new_x = self._PS._ps_data[self._PS.columns['x']] + distance
            new_y = self._PS._ps_data[self._PS.columns['y']] + np.divide(self._PS._ps_data[self._PS.columns['py']],
                                                                         self._PS._ps_data[
                                                                             self._PS.columns['px']]) * distance
            new_z = self._PS._ps_data[self._PS.columns['z']] + np.divide(self._PS._ps_data[self._PS.columns['pz']],
                                                                         self._PS._ps_data[
                                                                             self._PS.columns['px']]) * distance
        elif direction == 'y':
            new_x = self._PS._ps_data[self._PS.columns['x']] + np.divide(self._PS._ps_data[self._PS.columns['px']],
                                                                         self._PS._ps_data[
                                                                             self._PS.columns['py']]) * distance
            new_y = self._PS._ps_data[self._PS.columns['y']] + distance
            new_z = self._PS._ps_data[self._PS.columns['z']] + np.divide(self._PS._ps_data[self._PS.columns['pz']],
                                                                         self._PS._ps_data[
                                                                             self._PS.columns['py']]) * distance
        elif direction == 'z':
            new_x = self._PS._ps_data[self._PS.columns['x']] + np.divide(self._PS._ps_data[self._PS.columns['px']],
                                                                         self._PS._ps_data[
                                                                             self._PS.columns['pz']]) * distance
            new_y = self._PS._ps_data[self._PS.columns['y']] + np.divide(self._PS._ps_data[self._PS.columns['py']],
                                                                         self._PS._ps_data[
                                                                             self._PS.columns['pz']]) * distance
            new_z = self._PS._ps_data[self._PS.columns['z']] + distance
        else:
            raise NotImplementedError('direction must be "x", "y", or "z"')

        new_instance = self._return_position_update(new_x, new_y, new_z, in_place)
        return new_instance  # nb: None if in_place = True

    def regrid(self, quantities: (list, None) = None, n_bins: (int, list) = 10, in_place=False):
        """
        this re-grids each quantity in quantities onto a new grid. The new grid is defined by
         np.linspace(min(quantity, max(quantity), n_bins). Regridding following by merging is a good way of combining particles
         which are very close together. The underlying algorithm was developed by Leo Esnault for the
         `p2sat <https://github.com/lesnat/p2sat>`_ code::

            PS.regrid()
            PS.regrid(quantities=['x', 'y'], n_bins=50)

        :param quantities: Quantities to regrid; if None defaults of ['x', 'y', 'z', 'px', 'py', 'pz', 'time'] are used.
            quantities can be anything in PhaseSpace.columns
        :param n_bins: number of bins to rebin into. Can be a single number, in which case this is applied to all quantities,
            or a list of integers, one per quantity
        """

        def _rounder(values):
            """
            function to round the values of an array to closest point in second array.
            Full credit here:
            https://stackoverflow.com/questions/33450235/rounding-a-list-of-values-to-the-nearest-value-from-another-list-in-python
            """

            def f(x):
                idx = np.argmin(np.abs(values - x))
                return values[idx]

            return np.frompyfunc(f, 1, 1)


        for col in self._PS._ps_data.columns:
            if isinstance(self._PS._ps_data[col].dtype, pd.CategoricalDtype):
                warnings.warn(f'forcing {col} from categorical to explicit for regrid')
                self._PS._ps_data[col] = self._PS.ps_data[col].cat.categories[self._PS.ps_data[col].cat.codes]
        # set up quantities to regrid:
        quantities = self._PS._get_quantities(quantities)

        # set up bins:
        if isinstance(n_bins, int):
            # all quantities have same number of bins
            n_bins = [n_bins] * len(quantities)
        elif isinstance(n_bins, list):
            # different number of bins per quantity
            if not len(n_bins) == len(quantities):
                raise Exception(f'length of bins must equal length of quantities; '
                                f'\nyou have len(n_bins)={len(n_bins)} and len(quantities)={len(quantities)}')
        bin_array = {}
        start_time = perf_counter()
        for quantity, bin_length in zip(quantities, n_bins):
            bin_min = self._PS._ps_data[self._PS.columns[quantity]].min()
            bin_max = self._PS._ps_data[self._PS.columns[quantity]].max()
            bin_array[quantity] = np.linspace(bin_min, bin_max, bin_length)
        new_data = self._PS._ps_data.copy(True)
        for quantity in quantities:
            q_bins = bin_array[quantity]
            if len(np.unique(q_bins)) == 1:
                print(f'not regriding {quantity} as it is already single valued')
                # skip quantities that are already single valued
                continue
            new_data[self._PS.columns[quantity]] = list(_rounder(q_bins)(self._PS._ps_data[self._PS.columns[quantity]]))
            # new_data[self._PS.columns[quantity]] = rounded_new_quantity
        print(f'regrid operation took {perf_counter() - start_time: 1.1f} s')
        if in_place:
            self._PS.ps_data = new_data
            self._PS.reset_phase_space()
        else:
            ps_data = DataLoaders.Load_PandasData(new_data)
            new_PS = PhaseSpace(ps_data)
            return new_PS


class _Fill_Methods(_PhaseSpace_MethodHolder):
    """
    Methods for calculating secondary quantities and adding them to ps_data
    """

    def kinetic_E(self):
        """
        Uses `energy-momementum relation <https://en.wikipedia.org/wiki/Energy%E2%80%93momentum_relation>`_ to add
         kinetic energy into self._ps_data
        """
        if not self._PS.columns['rest mass'] in self._PS._ps_data.columns:
            self._PS.fill.rest_mass()
        if not self._PS.columns['p_abs'] in self._PS._ps_data.columns:
            self._PS.fill.absolute_momentum()

        TOT_E = np.sqrt(
            self._PS._ps_data[self._PS.columns['p_abs']] ** 2 + self._PS._ps_data[self._PS.columns['rest mass']] ** 2)
        Kin_E = np.subtract(TOT_E, self._PS._ps_data[self._PS.columns['rest mass']])
        self._PS._ps_data[self._PS.columns['Ek']] = Kin_E
        self._PS._check_ps_data_format()

    def rest_mass(self):
        """
        add rest mass to self._PS._ps_data
        :return:
        """
        rest_mass_MeV = ps_util.get_rest_masses_from_pdg_codes(self._PS._ps_data['particle type [pdg_code]'])
        rest_mass_correct_units = rest_mass_MeV / self._PS._conversions['mass']
        self._PS._ps_data[self._PS.columns['rest mass']] = pd.Series(rest_mass_correct_units)
        self._PS._check_ps_data_format()

    def relativistic_mass(self):
        """
        add relativistic mass to ps_data
        """
        if not self._PS.columns['gamma'] in self._PS._ps_data.columns:
            self._PS.fill.beta_and_gamma()
        if not self._PS.columns['rest mass'] in self._PS._ps_data.columns:
            self._PS.fill.rest_mass()

        self._PS._ps_data[self._PS.columns['relativistic mass']] = np.multiply(
            self._PS._ps_data[self._PS.columns['gamma']],
            self._PS._ps_data[self._PS.columns['rest mass']])
        self._PS._check_ps_data_format()

    def velocity(self):
        """
        add velocities in m/s into self._PS._ps_data
        """
        if not self._PS.columns['rest mass'] in self._PS._ps_data.columns:
            self._PS.fill.rest_mass()
        if not self._PS.columns['gamma'] in self._PS._ps_data.columns:
            self._PS.fill.beta_and_gamma()
        # self._PS._ps_data[self._PS.columns['vx']] = np.divide(self._PS._ps_data[self._PS.columns['px']], (self._PS._ps_data[self._PS.columns['gamma']] * self._PS._ps_data[self._PS.columns['rest mass']]))
        # self._PS._ps_data[self._PS.columns['vy']] = np.divide(self._PS._ps_data[self._PS.columns['py']], (self._PS._ps_data[self._PS.columns['gamma']] * self._PS._ps_data[self._PS.columns['rest mass']]))
        # self._PS._ps_data[self._PS.columns['vz']] = np.divide(self._PS._ps_data[self._PS.columns['pz']], (self._PS._ps_data[self._PS.columns['gamma']] * self._PS._ps_data[self._PS.columns['rest mass']]))

        self._PS._ps_data[self._PS.columns['vx']] = np.multiply(self._PS._ps_data[self._PS.columns['beta_x']],
                                                                constants.c)
        self._PS._ps_data[self._PS.columns['vy']] = np.multiply(self._PS._ps_data[self._PS.columns['beta_y']],
                                                                constants.c)
        self._PS._ps_data[self._PS.columns['vz']] = np.multiply(self._PS._ps_data[self._PS.columns['beta_z']],
                                                                constants.c)
        self._PS._check_ps_data_format()

    def absolute_momentum(self):

        self._PS._ps_data[self._PS.columns['p_abs']] = np.sqrt((self._PS._ps_data[self._PS.columns['px']] ** 2 +
                                                                       self._PS._ps_data[self._PS.columns['py']] ** 2 +
                                                                       self._PS._ps_data[
                                                                           self._PS.columns['pz']] ** 2).
                                                               to_numpy(dtype=self._PS._ps_data[self._PS.columns['x']].dtype))

    def beta_and_gamma(self):
        """
        add the relatavistic beta and gamma factors into self._PS._ps_data
        """
        if not self._PS.columns['Ek'] in self._PS._ps_data.columns:
            self._PS.fill.kinetic_E()
        if not self._PS.columns['rest mass'] in self._PS._ps_data.columns:
            self._PS.fill.rest_mass()
        if not self._PS.columns['p_abs'] in self._PS._ps_data.columns:
            self._PS.fill.absolute_momentum()

        self._PS._ps_data['beta_abs'] = np.divide(self._PS._ps_data[self._PS.columns['p_abs']],
                                                  self._PS._ps_data[self._PS.columns['Ek']] + self._PS._ps_data[
                                                      self._PS.columns['rest mass']])
        self._PS._ps_data[self._PS.columns['beta_x']] = np.divide(self._PS._ps_data[self._PS.columns['px']],
                                                                  self._PS._ps_data[self._PS.columns['Ek']] +
                                                                  self._PS._ps_data[
                                                                      self._PS.columns['rest mass']])
        self._PS._ps_data[self._PS.columns['beta_y']] = np.divide(self._PS._ps_data[self._PS.columns['py']],
                                                                  self._PS._ps_data[self._PS.columns['Ek']] +
                                                                  self._PS._ps_data[
                                                                      self._PS.columns['rest mass']])
        self._PS._ps_data[self._PS.columns['beta_z']] = np.divide(self._PS._ps_data[self._PS.columns['pz']],
                                                                  self._PS._ps_data[self._PS.columns['Ek']] +
                                                                  self._PS._ps_data[
                                                                      self._PS.columns['rest mass']])
        # assert np.allclose(np.sqrt(self._PS._ps_data[self._PS.columns['beta_x']]**2 + self._PS._ps_data[self._PS.columns['beta_y']]**2 +self._PS._ps_data[self._PS.columns['beta_z']]**2), self._PS._ps_data['beta_abs'])
        self._PS._ps_data[self._PS.columns['gamma']] = 1 / np.sqrt(1 - np.square(self._PS._ps_data['beta_abs']))
        self._PS._check_ps_data_format()

    def direction_cosines(self):
        """
        Calculate direction cosines, which are required for topas import:
        U (direction cosine of momentum with respect to X)
        V (direction cosine of momentum with respect to Y)
        """

        V = np.sqrt(self._PS._ps_data[self._PS.columns['px']] ** 2 + self._PS._ps_data[self._PS.columns['py']] ** 2 +
                    self._PS._ps_data[
                        self._PS.columns['pz']] ** 2)
        self._PS._ps_data[self._PS.columns['Direction Cosine X']] = self._PS._ps_data[self._PS.columns['px']] / V
        self._PS._ps_data[self._PS.columns['Direction Cosine Y']] = self._PS._ps_data[self._PS.columns['py']] / V
        self._PS._ps_data[self._PS.columns['Direction Cosine Z']] = self._PS._ps_data[self._PS.columns['pz']] / V
        self._PS._check_ps_data_format()


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
        self._ps_data = data_loader.data.copy(deep=True)
        self._units = data_loader._units
        self._conversions = ps_util.get_unit_conversions(self._units, ParticlePhaseSpaceUnits()('mm_MeV'))
        self.columns = ps_cfg.get_all_column_names(self._units)
        self._unique_particles = self._ps_data[self.columns['particle type']].unique()
        self.twiss_parameters = {}
        self.energy_stats = {}
        self.plot = _Plots(PS=self)
        self.fill = _Fill_Methods(PS=self)
        self.transform = _Transform(PS=self)

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
                if not col_name in ps_cfg.get_required_column_names(self._units):
                    particle_data.drop(columns=col_name, inplace=True)
            # create a new instance of _DataImportersBase based on particle_data
            particle_data_loader = DataLoaders.Load_PandasData(particle_data, units=self._units)
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

        :param other: the other phase space to add
        :type other: PhaseSpace
        """
        if not isinstance(other, PhaseSpace):
            raise TypeError('can only add together phase space objects')
        new_data = pd.concat([self._ps_data, other.ps_data])

        for col_name in new_data.columns:
            if not col_name in ps_cfg.get_required_column_names(self._units):
                new_data.drop(columns=col_name, inplace=True)
        new_data_loader = DataLoaders.Load_PandasData(new_data, units=self._units)
        new_instance = PhaseSpace(new_data_loader)
        return new_instance

    def __sub__(self, other):
        """
        subtract phase space (remove every particle in other from self)

        :param other: the other phase space to add
        :type other: PhaseSpace
        """
        if not isinstance(other, PhaseSpace):
            raise TypeError('can only add together phase space objects')
        new_data = pd.merge(self._ps_data, other.ps_data, how='outer', indicator=True) \
            .query("_merge != 'both'") \
            .drop('_merge', axis=1) \
            .reset_index(drop=True)
        for col_name in new_data.columns:
            if not col_name in ps_cfg.get_required_column_names(self._units):
                new_data.drop(columns=col_name, inplace=True)
        new_data_loader = DataLoaders.Load_PandasData(new_data, units=self._units)
        new_instance = PhaseSpace(new_data_loader)
        return new_instance

    def __len__(self):
        return self._ps_data.shape[0]

    @property
    def ps_data(self):
        return self._ps_data

    @ps_data.setter
    def ps_data(self, new_data_frame):
        """
        gets run whenever ps_data gets changed.
        :warning!: this doesn't work in all situations; see here:
        https://stackoverflow.com/questions/75205824/pandas-dataframe-as-a-property-of-a-class-setter-not-called-when-columns-change
        So it is always advised to manually call reset_phase_space

        :param new_data_frame: the updated phase space data
        :type new_data_frame: pandas.DataFrame
        """
        self._ps_data = new_data_frame
        self.reset_phase_space()

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
        if data.shape == ():
            # rare occasion when we try to calculate the median of a single data point
            w_median = float(data)
        else:
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

        all_allowedcolumns = list(ps_cfg.get_all_column_names(self._units).values())
        for col_name in self._ps_data.columns:
            if not col_name in all_allowedcolumns:
                raise AttributeError(f'non allowed column name {col_name} in ps_data')

    def _get_ellipse_xy_points(self, ellipse_parameters, x_search_min,
                               x_search_max, xpq_search_min, xpq_search_max):  # pragma: no cover
        """
        given the parameters of an ellipse, return a set of points in XY which meet those parameters

        :param ellipse_parameters:
        :param x_search_min:
        :param x_search_max:
        :param xpq_search_min:
        :param xpq_search_max:
        :return:
        """
        gamma = ellipse_parameters[self.columns['gamma']]
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

    def _get_data_for_trace_space_plots(self, beam_direction, ps_data, particle_name):  # pragma: no cover
        if beam_direction == 'z':
            x_data_1 = ps_data[self.columns['x']]
            div_data_1 = np.divide(ps_data[self.columns['px']], ps_data[self.columns['pz']])
            x_label_1 = self.columns['x']
            y_label_1 = "x' [rad]"
            title_1 = particle_name + ': x'
            weight = ps_data['weight']
            elipse_parameters_1 = self.twiss_parameters[particle_name]['x']

            x_data_2 = ps_data[self.columns['y']]
            div_data_2 = np.divide(ps_data[self.columns['py']], ps_data[self.columns['pz']])
            x_label_2 = self.columns['y']
            y_label_2 = "y' [rad]"
            title_2 = particle_name + ': y'
            elipse_parameters_2 = self.twiss_parameters[particle_name]['y']
        elif beam_direction == 'x':
            x_data_1 = ps_data[self.columns['y']]
            div_data_1 = np.divide(ps_data[self.columns['py']], ps_data[self.columns['px']])
            x_label_1 = self.columns['y']
            y_label_1 = "y' [rad]"
            title_1 = particle_name + ': x'
            weight = ps_data['weight']
            elipse_parameters_1 = self.twiss_parameters[particle_name]['y']

            x_data_2 = ps_data[self.columns['z']]
            div_data_2 = np.divide(ps_data[self.columns['pz']], ps_data[self.columns['px']])
            x_label_2 = self.columns['z']
            y_label_2 = "z' [rad]"
            title_2 = particle_name + ': y'
            elipse_parameters_2 = self.twiss_parameters[particle_name]['z']
        elif beam_direction == 'y':
            x_data_1 = ps_data[self.columns['x']]
            div_data_1 = np.divide(ps_data[self.columns['px']], ps_data[self.columns['py']])
            x_label_1 = self.columns['x']
            y_label_1 = "x' [rad]"
            title_1 = particle_name + ': x'
            weight = ps_data['weight']
            elipse_parameters_1 = self.twiss_parameters[particle_name]['x']

            x_data_2 = ps_data[self.columns['z']]
            div_data_2 = np.divide(ps_data[self.columns['pz']], ps_data[self.columns['py']])
            x_label_2 = self.columns['z']
            y_label_2 = "z' [rad]"
            title_2 = particle_name + ': y'
            elipse_parameters_2 = self.twiss_parameters[particle_name]['z']
        else:
            raise NotImplementedError(f'beam_direction must be "x", "y", or "z", not {beam_direction}')

        return x_data_1, div_data_1, x_label_1, y_label_1, title_1, weight, elipse_parameters_1, \
               x_data_2, div_data_2, x_label_2, y_label_2, title_2, elipse_parameters_2

    def _get_quantities(self, quantities):
        """
        process user inputs and return a valid list of quantities to process;
         used by various methods that accept a quantities lsit
        :param quantities:
        :return:
        """
        if quantities is None:
            quantities = ['x', 'y', 'z', 'px', 'py', 'pz', 'time']
        if isinstance(quantities, str):
            quantities = [quantities]  # in case user passes e.g. 'x'
        for quantity in quantities:
            if not quantity in self.columns.keys():
                raise AttributeError(f'{quantity} is not a valid quantity: valid quantites are:'
                                     f'\n{self.columns.keys()}')
            if not self.columns[quantity] in self._ps_data.columns:
                raise AttributeError(f'quantity {quantity} is valid, but is not currently included in PhaseSpace data. '
                                     f'\nUse one of the "fill" methods to fill in this quantity')
        return quantities

    def _quantities_to_column_names(self, quantities):
        """
        convert generic quantities ('x' 'px' etc.) to unit appropriate column names
        :param quantities:
        :return:
        """
        column_names = []
        for quantity in quantities:
            column_names.append(self.columns[quantity])
        return column_names

    # public methods

    def print_energy_stats(self, file_name=None):  # pragma: no cover
        """
        Prints a summary of energy stats to the screen, which can optionally be saved to json

        :param file_name: if specified, the data is saved as json in file_name
        :type file_name: Path, str, optional
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
                  f'\n        mean energy: {self.energy_stats[particle]["mean energy"]: 1.2f} {self._units.energy.label}'
                  f'\n        median energy: {self.energy_stats[particle]["median energy"]: 1.2f} {self._units.energy.label}'
                  f'\n        Energy spread IQR: {self.energy_stats[particle]["energy spread IQR"]: 1.2f} {self._units.energy.label}'
                  f'\n        min energy {self.energy_stats[particle]["min energy"]: 1.2f} {self._units.energy.label}'
                  f'\n        max energy {self.energy_stats[particle]["max energy"]: 1.2f} {self._units.energy.label}')

    def print_twiss_parameters(self, file_name=None, beam_direction: str = 'z'):  # pragma: no cover
        """
        prints the twiss parameters if they exist
        they are always printed to the screen.
        if filename is specified, they are also saved to file as json

        :param file_name: optional filename to write twiss data to. should be absolute path to an existing directory
        :type file_name: str or Path, optional
        :param beam_direction: the direction the beam is travelling in. "x", "y", or "z" (default)
        :type beam_direction: str, optional
        """
        for particle in self._unique_particles:
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

    def calculate_twiss_parameters(self, beam_direction='z'):
        """
        Calculate the RMS `twiss parameters <https://en.wikipedia.org/wiki/Courant%E2%80%93Snyder_parameters>`_

        :param beam_direction: main direction in which beam is travelling. 'x', 'y', or 'z' (default)
        :type beam_direction: str, optional
        :return: None
        """
        if beam_direction == 'x':
            intersectioncolumns = [self.columns['x'], self.columns['px']]
            directioncolumns = [[self.columns['z'], self.columns['pz']], [self.columns['y'], self.columns['py']]]
        elif beam_direction == 'y':
            intersectioncolumns = [self.columns['y'], self.columns['py']]
            directioncolumns = [[self.columns['x'], self.columns['px']], [self.columns['z'], self.columns['pz']]]
        elif beam_direction == 'z':
            intersectioncolumns = [self.columns['z'], self.columns['pz']]
            directioncolumns = [[self.columns['x'], self.columns['px']], [self.columns['y'], self.columns['py']]]
        else:
            raise NotImplementedError('beam direction must be "x", "y", or "z"')
        for particle in self._unique_particles:
            particle_name = particle_cfg.particle_properties[particle]['name']
            ind = self._ps_data['particle type [pdg_code]'] == particle
            particle_data = self._ps_data[ind]
            self.twiss_parameters[particle_name] = {}
            for calc_dir in directioncolumns:
                x2 = np.average(np.square(particle_data[calc_dir[0]]), weights=particle_data['weight'])
                xp = np.divide(particle_data[calc_dir[1]], particle_data[intersectioncolumns[1]])
                xp2 = np.average(np.square(xp), weights=particle_data['weight'])
                x_xp = np.average(np.multiply(particle_data[calc_dir[0]], xp), weights=particle_data['weight'])

                epsilon = np.sqrt((x2 * xp2) - (x_xp ** 2))
                alpha = -x_xp / epsilon
                beta = x2 / epsilon
                gamma = xp2 / epsilon

                self.twiss_parameters[particle_name][calc_dir[0][0]] = {'epsilon': epsilon,
                                                                        'alpha': alpha,
                                                                        'beta': beta,
                                                                        self.columns['gamma']: gamma}

    def calculate_energy_statistics(self):
        if not self.columns['Ek'] in self._ps_data.columns:
            self.fill.kinetic_E()
        for particle in self._unique_particles:
            particle_name = particle_cfg.particle_properties[particle]['name']
            self.energy_stats[particle_name] = {}
            ind = self._ps_data['particle type [pdg_code]'] == particle
            ps_data = self._ps_data[ind]
            self.energy_stats[particle_name]['number'] = np.count_nonzero(ind)
            meanEnergy, stdEnergy = self._weighted_avg_and_std(ps_data[self.columns['Ek']], ps_data['weight'])
            self.energy_stats[particle_name]['min energy'] = ps_data[self.columns['Ek']].min()
            self.energy_stats[particle_name]['max energy'] = ps_data[self.columns['Ek']].max()
            self.energy_stats[particle_name]['mean energy'] = meanEnergy
            self.energy_stats[particle_name]['std mean'] = stdEnergy
            self.energy_stats[particle_name]['median energy'] = self._weighted_median(ps_data[self.columns['Ek']],
                                                                                      ps_data['weight'])
            q75, q25 = self._weighted_quantile(ps_data[self.columns['Ek']], [0.25, 0.75],
                                               sample_weight=ps_data['weight'])
            self.energy_stats[particle_name]['energy spread IQR'] = q25 - q75

    def reset_phase_space(self):
        """
        reduce self._ps_data to only the required columns
        delete any other derived quantities such as twiss parameters
        this can be called whenever you want to reduce the memory footprint
        """
        for col_name in self._ps_data.columns:
            if not col_name in ps_cfg.get_required_column_names(self._units):
                self._ps_data.drop(columns=col_name, inplace=True)

        self.twiss_parameters = {}
        self.energy_stats = {}
        self._check_ps_data_format()
        self._unique_particles = self._ps_data['particle type [pdg_code]'].unique()
        self._ps_data['particle id'] = self._ps_data['particle id'].astype(
            np.int64)  # seems to keep getting recast as float which is annoying

    def assess_density_versus_r(self, Rvals=None, verbose: bool = True, beam_direction: str = 'z'):
        """
        Assess how many particles are in a given radius

        :param Rvals: list of rvals to assess in mm, e.g. np.linspace(0, 2, 21)
        :param verbose: prints data to screen if True
        :param beam_direction: main direction in which beam is travelling. 'x', 'y', or 'z' (default)
        :type beam_direction: str, optional
        :return density_data: a dataframe containing the roi vals and the proportion of particles inside
        """

        if self._ps_data['weight'].max() > 1:
            warnings.warn('AssessDensityVersusR function ignores particle weights')

        if Rvals is None:
            # pick a default
            Rvals = np.linspace(0, 2, 21)
        if not isinstance(Rvals, (list, np.ndarray)):
            Rvals = list(Rvals)
        if beam_direction == 'x':
            r = np.sqrt(self._ps_data[self.columns['z']] ** 2 + self._ps_data[self.columns['y']] ** 2)
        elif beam_direction == 'y':
            r = np.sqrt(self._ps_data[self.columns['x']] ** 2 + self._ps_data[self.columns['z']] ** 2)
        elif beam_direction == 'z':
            r = np.sqrt(self._ps_data[self.columns['x']] ** 2 + self._ps_data[self.columns['y']] ** 2)

        numparticles = self._ps_data[self.columns['x']].shape[0]
        rad_prop = []

        for rcheck in Rvals:
            Rind = r < rcheck
            rad_prop.append(np.count_nonzero(Rind) * 100 / numparticles)

        density_data = pd.DataFrame({'roi [mm]': Rvals, '% particles inside': rad_prop})
        if verbose:
            print(density_data)
        return density_data

    def filter_by_time(self, t_start, t_finish, in_place: bool=False):
        """
        Generates a new PhaseSpace which only contains particles inside t_start and t_finish (inclusive).
        t_start and t_finish should be specfied in ps.

        :param t_start: particles with t<t_start are removed.
        :type t_start: float
        :param t_finish: particleswith t>t_finish are removed
        :type t_finish: float
        :return: new_instance: a new phase space object with data filtered by time
        """
        ind = np.logical_and(self._ps_data[self.columns['time']] >= t_start,
                             self._ps_data[self.columns['time']] <= t_finish)
        new_PS = self.filter_by_boolean_index(ind, in_place=in_place)
        return new_PS


    def filter_by_boolean_index(self, boolean_index, in_place: bool = False, split: bool = False,
                                verbose: bool=True):
        """
        filter data by input boolean index, keeping 'True' and discarding 'False'

        :param boolean_index: an 1D array like structure of True and False values
        :param in_place: if True, existing object is modified; if False a new object is returned
        :param split: if True, will return two phase space objects: one where boolan_index=True and one where it equals
            False
        """
        self.reset_phase_space()
        if split:
            if in_place:
                warnings.warn('cannot split the phase space while operating in place; ignoring in place and returning '
                              'two objects')
            ps_data = self.ps_data[boolean_index].reset_index().drop('index', axis=1)
            ps_data = DataLoaders.Load_PandasData(ps_data, units=self._units)
            boolean_index_true_PS = PhaseSpace(ps_data)

            ps_data = self.ps_data[np.logical_not(boolean_index)].reset_index().drop('index', axis=1)
            ps_data = DataLoaders.Load_PandasData(ps_data, units=self._units)
            boolean_index_false_PS = PhaseSpace(ps_data)
            if verbose:
                print(f'data where boolean_index=True accounts for'
                      f' {(np.count_nonzero(boolean_index) * 100 / len(boolean_index)): 1.1f} %'
                      f' of the original data')
                return boolean_index_true_PS, boolean_index_false_PS
        if in_place:
            self.ps_data = self.ps_data[boolean_index].reset_index().drop('index', axis=1)
            if verbose:
                print(f'removed {100 - (np.count_nonzero(boolean_index) * 100 / len(boolean_index)): 1.1f} % of particles')
        else:
            ps_data = self.ps_data[boolean_index].reset_index().drop('index', axis=1)
            ps_data = DataLoaders.Load_PandasData(ps_data, units=self._units)
            new_PS = PhaseSpace(ps_data)
            if verbose:
                print(f'removed {100 - (np.count_nonzero(boolean_index) * 100 / len(boolean_index)): 1.1f} % of particles')
            return new_PS

    def set_units(self, new_units: UnitSet):
        """
        converts ps_data to a new unit set.
        This will also reset the phase space to just the required columns

        :param new_units: the new units to convert to
        :type new_units: UnitSet
        """

        # reset phase space:
        self.reset_phase_space()
        # get conversion factors
        conversions = ps_util.get_unit_conversions(self._units, new_units)
        # convert data
        self._ps_data[self.columns['x']] = self._ps_data[self.columns['x']] * conversions['length']
        self._ps_data[self.columns['y']] = self._ps_data[self.columns['y']] * conversions['length']
        self._ps_data[self.columns['z']] = self._ps_data[self.columns['z']] * conversions['length']
        self._ps_data[self.columns['px']] = self._ps_data[self.columns['px']] * conversions['momentum']
        self._ps_data[self.columns['py']] = self._ps_data[self.columns['py']] * conversions['momentum']
        self._ps_data[self.columns['pz']] = self._ps_data[self.columns['pz']] * conversions['momentum']
        self._ps_data[self.columns['time']] = self._ps_data[self.columns['time']] * conversions['time']
        # get new column names
        oldcolumns_names = ps_cfg.get_required_column_names(self._units)
        new_column_names = ps_cfg.get_required_column_names(new_units)
        rename_dict = {oldcolumns_names[i]: new_column_names[i] for i in range(len(oldcolumns_names))}
        self._ps_data.rename(columns=rename_dict, inplace=True)
        # update self.columns
        self.columns = ps_cfg.get_all_column_names(new_units)
        self._units = new_units
        self._conversions = ps_util.get_unit_conversions(self._units, ParticlePhaseSpaceUnits()('mm_MeV'))

    def get_units(self):
        return self._units

    def resample_via_gaussian_kde(self, n_new_particles_factor: int = 1, interpolate_weights: (bool, None) = None):
        """
        Generate a new phase space based on the existing data, by fitting a gaussian kernel density
        estimate to the 6-D space:
        `x y z px py pz`
        If `interpolate_weights` is set to True, we instead attempt to interpolate within a 7D space:
        `x y z px py pz weight`
        This method is fairly experimental and should be used with extreme caution!

        :param n_new_particles_factor: the returned Phase space will have size len(self)*n_new_particles_factor.
            In other words, when n_new_particles_factor=1, the new PhaseSpace will be the same size as the original.
        :return: new_PS: a new PhaseSpace object
        """

        if len(self._unique_particles) > 1:
            raise Exception('This method can only be performed on single species phase spaces')
        warnings.warn('This method is quite experimental and should be used with extreme caution;'
                      'always manually compare the new PhaseSpace data to the old PhaseSpace data to ensure it is '
                      'close enough for your requirements')
        if len(np.unique(self._ps_data[self.columns['time']])) > 1:
            warnings.warn('your data has multiple time values in it: the new data, all time values will be the mean'
                          'of the input data')

        if interpolate_weights is None:
            # decide based on wheter there are multiple values of weight or not
            if len(np.unique(self._ps_data[self.columns['weight']])) > 1:
                interpolate_weights = True
            else:
                interpolate_weights = False

        if interpolate_weights:
            n_new_particles = int(len(self) * n_new_particles_factor)
            xyz_pxpypz_w = np.vstack([self.ps_data[self.columns['x']],
                                      self.ps_data[self.columns['y']],
                                      self.ps_data[self.columns['z']],
                                      self.ps_data[self.columns['px']],
                                      self.ps_data[self.columns['py']],
                                      self.ps_data[self.columns['pz']],
                                      self.ps_data[self.columns['weight']]])
            k = gaussian_kde(xyz_pxpypz_w)
            new_points = k.resample(n_new_particles)
        else:
            n_new_particles = int(len(self) * n_new_particles_factor)
            xyz_pxpypz = np.vstack([self.ps_data[self.columns['x']],
                                    self.ps_data[self.columns['y']],
                                    self.ps_data[self.columns['z']],
                                    self.ps_data[self.columns['px']],
                                    self.ps_data[self.columns['py']],
                                    self.ps_data[self.columns['pz']]])
            k = gaussian_kde(xyz_pxpypz, weights=self._ps_data[self.columns['weight']])
            new_points = k.resample(n_new_particles)
            _new_weights = np.ones(new_points.shape[1]) * np.mean(self._ps_data[self.columns['weight']])
            new_points = np.append(new_points, np.atleast_2d(_new_weights), axis=0)

        # If any inputs are single valued, replace the interpolation by this single value
        _cols_to_check = [self.columns['x'], self.columns['y'], self.columns['z'],
                          self.columns['px'], self.columns['py'], self.columns['pz'],
                          self.columns['weight']]
        _column = 0
        for col in _cols_to_check:
            if len(self._ps_data[col].unique()) == 1:
                new_points[_column, :] = self._ps_data[col][0]
            _column = _column + 1

        new_data = pd.DataFrame(
            {self.columns['x']: new_points[0, :],
             self.columns['y']: new_points[1, :],
             self.columns['z']: new_points[2, :],
             self.columns['px']: new_points[3, :],
             self.columns['py']: new_points[4, :],
             self.columns['pz']: new_points[5, :],
             self.columns['particle type']: self._ps_data[self.columns['particle type']][0],
             self.columns['weight']: new_points[6, :],
             self.columns['particle id']: np.arange(n_new_particles),
             self.columns['time']: self.ps_data[self.columns['time']].mean()})

        new_data = DataLoaders.Load_PandasData(new_data, units=self._units)
        new_PS = PhaseSpace(new_data)
        return new_PS

    def get_downsampled_phase_space(self, downsample_factor: int = 10):
        """
        return a new phase space object which randomlt samples from the larger phase space.
        the new phase space has size 'original data/downsample_factor'. the data is shuffled
        before being randomly sampled.

        :param downsample_factor: the factor to downsample the phase space by
        :type downsample_factor: int
        """
        new_data = self._ps_data.sample(frac=1).reset_index(drop=True)  # this shuffles the data
        new_data = new_data.sample(frac=1 / downsample_factor, ignore_index=True)
        for col_name in new_data.columns:
            if not col_name in ps_cfg.get_required_column_names(self._units):
                new_data.drop(columns=col_name, inplace=True)
        new_data_loader = DataLoaders.Load_PandasData(new_data, units=self._units)
        new_instance = PhaseSpace(new_data_loader)
        return new_instance

    def sort(self, quantities_to_sort: (None, list) = None):
        """
        sort the data. Data will be sorted according quantities_to_sort, in order of quantity.
        Operates in place. example::

            PS.sort(quantities_to_sort='x')
            PS.sort(quantities_to_sort=['x', 'y', 'z', 'px'])

        :param quantities_to_sort:
        """
        quantities = self._get_quantities(quantities_to_sort)
        column_names_sort = self._quantities_to_column_names(quantities)
        # sort:
        self._ps_data.sort_values(column_names_sort, axis=0, ascending=True, inplace=True,
                                  kind='quicksort', na_position='last',
                                  ignore_index=True, key=None)

    def merge(self, in_place=False):
        """
        merges identical data points by combining their weights.
        Typically, before performing a merge operation you will want to perform a 'regrid' operation.
        The underlying algorithm was developed by Leo Esnault for
        the `p2sat <https://github.com/lesnat/p2sat>`_ code.
        Merged particles retain the particle ID from the first particle in the merged group.

        :param in_place: if True, self is operated on; if False, a new PhaseSpace is returned
        :return: new_PS if in_place is False.
        """
        self.reset_phase_space()
        for col in self._ps_data.columns:
            if isinstance(self._ps_data[col].dtype, pd.CategoricalDtype):
                warnings.warn(f'forcing {col} from categorical to explicit for merge')
                self._ps_data[col] = self.ps_data[col].cat.categories[self.ps_data[col].cat.codes]
        # first sort the phase space
        quantities_to_merge = self._get_quantities(['x', 'y', 'z', 'px', 'py', 'pz', 'time', 'particle type'])
        # self.sort(quantities_to_sort=quantities_to_merge)
        column_names_merge = self._quantities_to_column_names(quantities_to_merge)
        start_time = perf_counter()
        # new_data = self._ps_data.groupby(column_names_merge).apply(_add_weights)
        merge_data = self._ps_data.groupby(column_names_merge, as_index=False)
        new_data = merge_data.mean()
        weights = merge_data.sum()[self.columns['weight']]
        id = merge_data.first()[self.columns['particle id']]
        new_data[self.columns['weight']] = weights
        new_data[self.columns['particle id']] = id
        new_data.index = np.arange(new_data.shape[0])
        # if this worked, the sum of weight should be the same:
        assert np.isclose(new_data['weight'].sum(), self._ps_data['weight'].sum())
        print(f'merge operation removed {len(self) - new_data.shape[0]: d} of {len(self): d}'
              f' particles ({100 - (new_data.shape[0] * 100 / len(self)): 1.1f} % removed)')
        print(f'merge operation took {perf_counter() - start_time: 1.1f} s')
        if in_place:
            self.ps_data = new_data
            self.reset_phase_space()
        else:
            # new_data[self.columns['particle type']] = new_data[self.columns['particle type']].astype(np.int32)
            ps_data = DataLoaders.Load_PandasData(new_data)
            new_PS = PhaseSpace(ps_data)
            return new_PS

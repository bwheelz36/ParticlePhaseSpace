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
from time import perf_counter
import ParticlePhaseSpace.__phase_space_config__ as ps_cfg
import ParticlePhaseSpace.__particle_config__ as particle_cfg
from ParticlePhaseSpace.DataLoaders import _DataLoadersBase
from ParticlePhaseSpace import utilities as ps_util
from ParticlePhaseSpace import DataLoaders
from ParticlePhaseSpace import UnitSet, ParticlePhaseSpaceUnits


class _FigureSpecs:
    """
    Thought this might be the easiest way to ensure universal parameters accross all figures
    """
    LabelFontSize = 14
    TitleFontSize = 16
    Font = 'serif'
    AxisFontSize = 14
    TickFontSize = 14




class _Plots:

    def __init__(self, PS):
        self._PS = PS

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
        if not self._PS._columns['Ek'] in self._PS._ps_data.columns:
            self._PS.fill_kinetic_E()
        legend = []
        for particle in self._PS._unique_particles:
            legend.append(particle_cfg.particle_properties[particle]['name'])
            ind = self._PS._ps_data['particle type [pdg_code]'] == particle
            Eplot = self._PS._ps_data[self._PS._columns['Ek']][ind]
            n, bins, patches = axs.hist(Eplot, bins=n_bins, weights=self._PS._ps_data['weight'][ind], alpha=.5)

        axs.set_xlabel(self._PS._columns['Ek'], fontsize=_FigureSpecs.LabelFontSize)
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
            x_plot = self._PS._ps_data[self._PS._columns['x']][ind]
            y_plot = self._PS._ps_data[self._PS._columns['y']][ind]
            z_plot = self._PS._ps_data[self._PS._columns['z']][ind]
            axs[0].hist(x_plot, bins=n_bins, weights=self._PS._ps_data['weight'][ind], alpha=alpha)
            axs[1].hist(y_plot, bins=n_bins, weights=self._PS._ps_data['weight'][ind], alpha=alpha)
            axs[2].hist(z_plot, bins=n_bins, weights=self._PS._ps_data['weight'][ind], alpha=alpha)

        axs[0].set_xlabel(self._PS._columns['x'])
        axs[0].set_ylabel('counts')
        axs[0].set_title(self._PS._columns['x'])
        axs[0].legend(legend)

        axs[1].set_xlabel(self._PS._columns['y'])
        axs[1].set_ylabel('counts')
        axs[1].set_title(self._PS._columns['y'])

        axs[2].set_xlabel(self._PS._columns['z'])
        axs[2].set_ylabel('counts')
        axs[2].set_title(self._PS._columns['z'])

        if grid:
            axs[0].grid()
            axs[1].grid()
            axs[2].grid()

        plt.tight_layout()
        plt.show()

    def momentum_hist_1D(self, n_bins: int = 100, alpha: float = 0.5, grid: bool = False):
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
            x_plot = self._PS._ps_data[self._PS._columns['px']][ind]
            y_plot = self._PS._ps_data[self._PS._columns['py']][ind]
            z_plot = self._PS._ps_data[self._PS._columns['pz']][ind]
            axs[0].hist(x_plot, bins=n_bins, weights=self._PS._ps_data['weight'][ind], alpha=alpha)
            axs[1].hist(y_plot, bins=n_bins, weights=self._PS._ps_data['weight'][ind], alpha=alpha)
            axs[2].hist(z_plot, bins=n_bins, weights=self._PS._ps_data['weight'][ind], alpha=alpha)

        axs[0].set_xlabel(self._PS._columns['px'])
        axs[0].set_ylabel('counts')
        axs[0].set_title(self._PS._columns['px'])
        axs[0].legend(legend)

        axs[1].set_xlabel(self._PS._columns['py'])
        axs[1].set_ylabel('counts')
        axs[1].set_title(self._PS._columns['py'])

        axs[2].set_xlabel(self._PS._columns['pz'])
        axs[2].set_ylabel('counts')
        axs[2].set_title(self._PS._columns['pz'])

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
                x_data = ps_data[self._PS._columns['y']]
                y_data = ps_data[self._PS._columns['z']]
                x_label = self._PS._columns['y']
                y_label = self._PS._columns['z']
            elif beam_direction == 'y':
                x_data = ps_data[self._PS._columns['x']]
                y_data = ps_data[self._PS._columns['z']]
                x_label = self._PS._columns['x']
                y_label = self._PS._columns['z']
            elif beam_direction == 'z':
                x_data = ps_data[self._PS._columns['x']]
                y_data = ps_data[self._PS._columns['y']]
                x_label = self._PS._columns['x']
                y_label = self._PS._columns['y']
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

        if (not self._PS._columns['Ek'] in self._PS._ps_data.columns):
            self._PS.fill_kinetic_E()
        for particle in self._PS._unique_particles:
            ind = self._PS._ps_data['particle type [pdg_code]'] == particle
            ps_data = self._PS._ps_data.loc[ind]
            if beam_direction == 'x':
                loop_data = zip(ps_data[self._PS._columns['z']], ps_data[self._PS._columns['y']], ps_data[self._PS._columns['Ek']],
                                ps_data['weight'])
                _xlabel = self._PS._columns['z']
                _ylabel = self._PS._columns['y']
            if beam_direction == 'y':
                loop_data = zip(ps_data[self._PS._columns['x']], ps_data[self._PS._columns['z']], ps_data[self._PS._columns['Ek']],
                                ps_data['weight'])
                _xlabel = self._PS._columns['x']
                _ylabel = self._PS._columns['z']
            if beam_direction == 'z':
                loop_data = zip(ps_data[self._PS._columns['x']], ps_data[self._PS._columns['y']], ps_data[self._PS._columns['Ek']],
                                ps_data['weight'])
                _xlabel = self._PS._columns['x']
                _ylabel = self._PS._columns['y']
            if xlim is None:
                xlim = [ps_data[self._PS._columns['x']].min(), ps_data[self._PS._columns['x']].max()]
            if ylim is None:
                ylim = [ps_data[self._PS._columns['y']].min(), ps_data[self._PS._columns['y']].max()]
            if quantity == 'intensity':
                _title = f"n_particles intensity;\n{particle_cfg.particle_properties[particle]['name']}"
                _weight = ps_data['weight']
            elif quantity == 'energy':
                _title = f"energy intensity;\n{particle_cfg.particle_properties[particle]['name']}"
                _weight = np.multiply(ps_data['weight'], ps_data[self._PS._columns['Ek']])
            X = np.linspace(xlim[0], xlim[1], bins)
            Y = np.linspace(ylim[0], ylim[1], bins)
            h, xedges, yedges = np.histogram2d(ps_data[self._PS._columns['x']],
                                               ps_data[self._PS._columns['y']],
                                               bins=[X, Y], weights=_weight, )
            if normalize:
                h = h * 100 / h.max()
            # _im1 = axs[0, n_axs].hist2d(ps_data[self._PS._columns['x']], ps_data[self._PS._columns['y']],
            #                           bins=[X,Y],
            #                           weights=_weight, norm=LogNorm(vmin=1, vmax=100),
            #                           cmap='inferno',
            #                           vmin=vmin, vmax=vmax)[3]
            _im1 = axs[0, n_axs].pcolormesh(xedges, yedges, h.T, cmap='inferno',
                                            norm=_scale, rasterized=False, vmin=vmin, vmax=vmax)

            fig.colorbar(_im1, ax=axs[0, n_axs])

            axs[0, n_axs].set_title(_title)
            axs[0, n_axs].set_xlabel(_xlabel, fontsize=_FigureSpecs.LabelFontSize)
            axs[0, n_axs].set_ylabel(_ylabel, fontsize=_FigureSpecs.LabelFontSize)
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
                                               weights=ps_data[self._PS._columns['weight']])
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
                                               weights=ps_data[self._PS._columns['weight']])
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
        plt.hist(self._PS._ps_data[self._PS._columns['time']], n_bins)
        plt.xlabel(f'time {self._PS._units.time.label}')
        plt.ylabel('N particles')
        if grid:
            plt.grid()
        plt.tight_layout()

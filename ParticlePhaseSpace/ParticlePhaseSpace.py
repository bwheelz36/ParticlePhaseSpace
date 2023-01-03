

# PhaseSpace.py
import numpy as np
import pandas as pd
import scipy.constants
from matplotlib import pyplot as plt
from scipy import constants
from scipy.stats import norm
import os, sys
import glob
import logging

logging.basicConfig(level=logging.WARNING)
import warnings
from scipy.stats import gaussian_kde
from scipy import constants
from time import perf_counter


class ParticlePhaseSpace:
    """
    """

    def __init__(self, ps_data, verbose=False, weight_position_plot=False):

        self.ps_data = ps_data
        self._c = constants.c
        self.verbose = verbose
        self._weight_position_plot = weight_position_plot  # weights plots by density. Looks better but is slow.

        if self.verbose == True:
            self.PrintData()

    def __add__(self, other):
        """
        add two phase spaces together
        """
        pass

    def __sub__(self, other):
        """
        subtract phase space (remove ever
        """
        pass

    def __call__(self, particle_list):
        """
        this function allows users to seperate the phase space based on particle types
        """
        print('hello')


    def __weighted_median(self, data, weights):
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

    def __weighted_avg_and_std(self, values, weights):
        """
        credit: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values - average) ** 2, weights=weights)
        return (average, np.sqrt(variance))

    def __weighted_quantile(self, values, quantiles, sample_weight=None):
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

    def __AnalyseEnergyDistribution(self):
        self.meanEnergy, self.stdEnergy = self.__weighted_avg_and_std(self.E, self.weight)
        self.medianEnergy = self.__weighted_median(self.E, self.weight)
        self.EnergySpreadSTD = np.std(np.multiply(self.E, self.weight))
        q75, q25 = self.__weighted_quantile(self.E, [0.25, 0.75], sample_weight=self.weight)
        self.EnergySpreadIQR = q75 - q25

    def __CalculateTwissParameters(self):
        """
        Calculate the twiss parameters
        """
        # Calculate in X direction
        self.x2 = np.average(np.square(self.x), weights=self.weight)
        self.xp = np.divide(self.px, self.pz) * 1e3
        self.xp2 = np.average(np.square(self.xp), weights=self.weight)
        self.x_xp = np.average(np.multiply(self.x, self.xp), weights=self.weight)

        self.twiss_epsilon = np.sqrt((self.x2 * self.xp2) - (self.x_xp ** 2)) * np.pi
        self.twiss_alpha = -self.x_xp / self.twiss_epsilon
        self.twiss_beta = self.x2 / self.twiss_epsilon
        self.twiss_gamma = self.xp2 / self.twiss_epsilon

    def __CheckEnergyCalculation(self):
        """
        For the SLAC data, if we understand the units correctly, we should be able to recover the energy from the momentum....
        """
        Totm = np.sqrt((self.px ** 2 + self.py ** 2 + self.pz ** 2))
        self.TOT_E = np.sqrt(Totm ** 2 + self._me_MeV ** 2)
        Kin_E = np.subtract(self.TOT_E, self._me_MeV)

        E_error = max(self.E - Kin_E)
        if E_error > .01:
            sys.exit('Energy check failed: read in of data is wrong.')

    def __CalculateBetaAndGamma(self):
        """
        Calculate the beta and gamma factors from the momentum data

        input momentum is assumed to be in units of MeV/c
        I need to figure out if BetaX and BetaY make sense, or it's just Beta
        """

        if self.ParticleType == 'gamma':
            # then this stuff makes no sense
            return

        self.TOT_P = np.sqrt(self.px ** 2 + self.py ** 2 + self.pz ** 2)
        self.Beta = np.divide(self.TOT_P, self.TOT_E)
        self.Gamma = 1 / np.sqrt(1 - np.square(self.Beta))

    def __CalculateDirectionCosines(self):
        """
        Calculate direction cosines, which are required for topas import:

        U (direction cosine of momentum with respect to X)
        V (direction cosine of momentum with respect to Y)

        nb: using velocity or momentum seem to give the same results

        """
        V = np.sqrt(self.px ** 2 + self.py ** 2 + self.pz ** 2)
        U = self.px / V
        V = self.py / V
        return U, V

    def __GenerateTopasHeaderFile(self):
        """
        Generate the header file required for a topas phase space source.
        This is only intended to be used from within the class (private method)
        """

        WritefilePath = self.OutputDataLoc + '/' + self.OutputFile + '_tpsImport.header'

        ParticlesInPhaseSpace = str(len(self.x))
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
        TopasHeader.append('Number of e-: ' + ParticlesInPhaseSpace + '\n')
        TopasHeader.append('Minimum Kinetic Energy of e-: ' + str(min(self.E)) + ' MeV\n')
        TopasHeader.append('Maximum Kinetic Energy of e-: ' + str(max(self.E)) + ' MeV')

        # open file:
        try:
            f = open(WritefilePath, 'w')
        except FileNotFoundError:
            sys.exit('couldnt open file for writing')

        # Write file line by line:
        for Line in TopasHeader:
            f.write(Line)
            f.write('\n')

        f.close

    # public methods

    def PlotPhaseSpaceX(self):

        if self.weight.max() > 1:
            warnings.warn('this plot does not take into account particle weights')
        plt.figure()

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

    def PlotEnergyHistogram(self):

        try:
            test = self.Eaxs
        except AttributeError:
            self.Efig, self.Eaxs = plt.subplots()
        n, bins, patches = self.Eaxs.hist(self.E, bins=1000, weights=self.weight)
        # self.fig.set_size_inches(10, 5)
        self.Eaxs.set_xlabel('Energy [Mev]', fontsize=self.FigureSpecs.LabelFontSize)
        self.Eaxs.set_ylabel('N counts', fontsize=self.FigureSpecs.LabelFontSize)
        plot_title = self.OutputFile
        self.Eaxs.set_title(plot_title, fontsize=self.FigureSpecs.TitleFontSize)
        # self.Eaxs.tick_params(axis="y", labelsize=14)
        # self.Eaxs.tick_params(axis="x", labelsize=14)

        # self.Eaxs.set_xlim([0, 10.5])
        plt.tight_layout()
        plt.show()
        self.Eaxs.set_title(plot_title, fontsize=self.FigureSpecs.TitleFontSize)

    def PlotParticlePositions(self):

        try:
            test = self.axs[0]
        except AttributeError:
            self.fig, self.axs = plt.subplots(1, 2)
            self.fig.set_size_inches(10, 5)

        if self._weight_position_plot:
            _kde_data_grid = 150 ** 2
            print('generating weighted scatter plot...')

            xy = np.vstack([self.x, self.y])
            k = gaussian_kde(xy, weights=self.weight)
            _end_time = perf_counter()

            if self.x.shape[0] > _kde_data_grid:
                down_sample_factor = np.round(self.x.shape[0] / _kde_data_grid)
                print(f'down sampling scatter plot data by factor of {down_sample_factor}')
                # in this case we can downsample the grid...
                rng = np.random.default_rng()
                rng.shuffle(xy)  # operates in place for some confusing reason
                xy = rng.choice(xy, np.int(self.x.shape[0] / down_sample_factor), replace=False, axis=1, shuffle=False)
            z = k(xy)


            z = z / max(z)
            SP = self.axs[0].scatter(xy[0], xy[1], c=z, s=1)
            self.axs[0].set_aspect(1)
            # self.fig.colorbar(SP,ax=self.axs[0])
            self.axs[0].set_title('Particle Positions', fontsize=self.FigureSpecs.TitleFontSize)
            self.axs[0].set_xlim([-2, 2])
            self.axs[0].set_ylim([-2, 2])
            self.axs[0].set_xlabel('X position [mm]', fontsize=self.FigureSpecs.LabelFontSize)
            self.axs[0].set_ylabel('Y position [mm]', fontsize=self.FigureSpecs.LabelFontSize)
            plt.show()
            plt.tight_layout()
        else:
            self.axs[0].set_title('Particle Positions', fontsize=self.FigureSpecs.TitleFontSize)
            self.axs[0].scatter(self.x, self.y, s=1, c=self.weight)
            self.axs[0].set_xlabel('X position [mm]', fontsize=self.FigureSpecs.LabelFontSize)
            self.axs[0].set_ylabel('Y position [mm]', fontsize=self.FigureSpecs.LabelFontSize)

        self.axs[1].hist(self.x, bins=100, weights=self.weight)
        self.axs[1].set_xlabel('X position [mm]', fontsize=self.FigureSpecs.LabelFontSize)
        self.axs[1].set_ylabel('counts', fontsize=self.FigureSpecs.LabelFontSize)
        plt.tight_layout()
        plt.show()

    def PlotPositionHistogram(self):

        try:
            test = self.self.PosHistAxs[0]
        except AttributeError:
            self.fig, self.PosHistAxs = plt.subplots(1, 2)
            self.fig.set_size_inches(10, 5)

            n, bins, patches = self.PosHistAxs[0].hist(self.x, bins=100, density=True)
            self.PosHistAxs[0].set_xlabel('X position [mm]')
            self.PosHistAxs[0].set_ylabel('counts')
            # self.PosHistAxs[0].set_xlim([-2, 2])
            # Plot the PDF.
            mu, std = norm.fit(self.x)
            xmin, xmax = self.PosHistAxs[0].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            self.PosHistAxs[0].plot(x, p, 'k', linewidth=2)
            title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
            self.PosHistAxs[0].set_title(title)

            n, bins, patches = self.PosHistAxs[1].hist(self.y, bins=100)
            self.PosHistAxs[1].set_xlabel('X position [mm]')
            self.PosHistAxs[1].set_ylabel('counts')
            # self.PosHistAxs[1].set_xlim([-2, 2])
            # Plot the PDF.
            mu, std = norm.fit(self.y)
            xmin, xmax = self.PosHistAxs[1].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            self.PosHistAxs[1].plot(x, p * n.max(), 'k', linewidth=2)
            title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
            self.PosHistAxs[1].set_title(title)

            plt.tight_layout()
            plt.show()

    def ProjectParticles(self, zdiff):
        """
        Update the X,Y,Z position of each particle by projecting it forward/back by zdiff.

        This serves as a crude approximation to more advanced particle transport codes, but can be used to quickly
        check results
        """
        self.x = self.x + np.divide(self.vx, self.vz) * zdiff
        self.y = self.y + np.divide(self.vy, self.vz) * zdiff
        self.z = self.z + zdiff

        # update the info on particle distribution
        self.__CalculateBetaAndGamma()
        self.__CalculateTwissParameters()  # at the moment just caclculates transverse phse space in X.
        self.__AnalyseEnergyDistribution()

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

    def PrintData(self):
        """
        can be used to print info to the termainl
        """
        if (self.DataType == 'topas') or (self.DataType == 'SLAC') or (self.DataType == 'CST'):
            Filepath, filename = os.path.split(self.Data)
            print(f'\nFor file: {filename}')
        print(
            f'\u03C0\u03B5: {self.twiss_epsilon: 1.1f} mm mrad, \u03B1: {self.twiss_alpha: 1.1f}, \u0392: {self.twiss_beta: 1.1f}, '
            f'\u03B3: {self.twiss_gamma: 1.1f}')
        print(f'Median energy: {self.medianEnergy: 1.2f} MeV \u00B1 {self.EnergySpreadIQR: 1.2f} (IQR) ')
        print(f'Mean energy: {self.meanEnergy: 1.2f} MeV \u00B1 {self.stdEnergy: 1.2f} (std) ')
        _mean_z, _std_z = self.__weighted_avg_and_std(self.z, self.weight)
        print(f'Mean Z position of input data is {_mean_z: 3.1f} mm \u00B1 {_std_z: 1.1f} (std)')
        medianEnergy = self.medianEnergy
        CutOff = .05
        ind = np.logical_and(self.E > (medianEnergy - (.05 * medianEnergy)),
                             self.E < (medianEnergy + (.05 * medianEnergy)))
        weighted_ind = np.multiply(ind, self.weight)
        weighted_percent = np.sum(weighted_ind) * 100 / np.sum(self.weight)

        print(
            f'{weighted_percent:1.1f} % of particles are within \u00B1 5% of the median dose ({medianEnergy: 1.1f} MeV) ')

        if hasattr(self, 'zOut'):
            _mean_zOut, _std_zOut = self.__weighted_avg_and_std(self.zOut, self.weight)
            print(
                f'Mean Z position of output data is {_mean_zOut: 3.1f} \u00B1 {_std_zOut: 1.1f} (std)')

    # export methods

    def export_to_cst_pid(self, Zoffset=None):
        """
        Generate a phase space which can be directly imported into CST
        For a constant emission model: generate a .pid ascii file
        Below is the example from CST:

        % Use always SI units.
        % The momentum (mom) is equivalent to beta* gamma.
        %
        % Columns: pos_x  pos_y  pos_z  mom_x  mom_y  mom_z  mass  charge  current

        1.0e-3   4.0e-3  -1.0e-3   1.0   2.0   1.0   9.11e-31  -1.6e-19   1.0e-6
        2.0e-3   4.0e-3   1.0e-3   1.0   2.0   1.0   9.11e-31  -1.6e-19   1.0e-6
        3.0e-3   2.0e-3   1.0e-3   1.0   2.0   2.0   9.11e-31  -1.6e-19   1.0e-6
        4.0e-3   4.0e-3   5.0e-3   1.0   2.0   1.0   9.11e-31  -1.6e-19   2.0e-6
        """
        warnings.warn('I havent tested this function for a very long time, so please verify that it works..')
        # Split the original file and extract the file name
        NparticlesToWrite = np.size(self.x)  # use this to create a smaller PID flie for easier trouble shooting
        WritefilePath = self.OutputDataLoc + '/' + self.OutputFile + '.pid'
        # generate other information required by pid file:

        Charge = self.weight * constants.elementary_charge * -1
        Mass = self.weight * constants.electron_mass
        total_weight = self.weight.sum()
        relative_weight = self.weight/total_weight
        Current = self.TotalCurrent * relative_weight # very crude approximation!!
        x = self.x * 1e-3  ## convert to m
        y = self.y * 1e-3
        if Zoffset == None:
            # Zoffset is an optional parameter to change the starting location of the particle beam (which
            # assume propogates in the Z direction)
            self.zOut = self.z * 1e-3
        else:
            self.zOut = (self.z + Zoffset) * 1e-3
        px = self.px / self._me_MeV
        py = self.py / self._me_MeV
        pz = self.pz / self._me_MeV
        # generate PID file
        Data = [x[0:NparticlesToWrite], y[0:NparticlesToWrite], self.zOut[0:NparticlesToWrite],
                px[0:NparticlesToWrite], py[0:NparticlesToWrite], pz[0:NparticlesToWrite],
                Mass[0:NparticlesToWrite], Charge[0:NparticlesToWrite], Current[0:NparticlesToWrite]]

        Data = np.transpose(Data)
        np.savetxt(WritefilePath, Data, fmt='%01.3e', delimiter='      ')

    def export_to_cst_pit(self, Zoffset=None):
        """
        % Use always SI units.
        % The momentum (mom) is equivalent to beta * gamma.
        % The data need not to be chronological ordered.
        %
        % Columns: pos_x  pos_y  pos_z  mom_x  mom_y  mom_z  mass  charge  charge(macro)  time

        1.0e-3   4.0e-3  -1.0e-3   1.0   2.0   1.0   9.11e-31  -1.6e-19   -2.6e-15   0e-6
        2.0e-3   4.0e-3   1.0e-3   1.0   2.0   1.0   9.11e-31  -1.6e-19   -3.9e-15   1e-6
        3.0e-3   2.0e-3   1.0e-3   1.0   2.0   2.0   9.11e-31  -1.6e-19   -3.9e-15   2e-6
        4.0e-3   4.0e-3   5.0e-3   1.0   2.0   1.0   9.11e-31  -1.6e-19   -2.6e-15   3e-6
        """

        warnings.warn('I havent tested this function for a very long time, so please verify that it works..')
        # Split the original file and extract the file name
        NparticlesToWrite = np.size(self.x)  # use this to create a smaller PID flie for easier trouble shooting
        WritefilePath = self.OutputDataLoc + '/' + self.OutputFile + '.pid'
        # generate other information required by pid file:

        Charge = self.weight * constants.elementary_charge * -1
        Mass = self.weight * constants.electron_mass
        Weight = self.weight
        x = self.x * 1e-3  ## convert to m
        y = self.y * 1e-3
        if Zoffset == None:
            # Zoffset is an optional parameter to change the starting location of the particle beam (which
            # assume propogates in the Z direction)
            self.zOut = self.z * 1e-3
        else:
            self.zOut = (self.z + Zoffset) * 1e-3
        px = self.px / self._me_MeV
        py = self.py / self._me_MeV
        pz = self.pz / self._me_MeV
        time = np.zeros(self.x.shape)

        # generate PID file
        Data = [x[0:NparticlesToWrite], y[0:NparticlesToWrite], self.zOut[0:NparticlesToWrite],
                px[0:NparticlesToWrite], py[0:NparticlesToWrite], pz[0:NparticlesToWrite],
                Mass[0:NparticlesToWrite], Charge[0:NparticlesToWrite], Weight[0:NparticlesToWrite],
                time[0:NparticlesToWrite]]

        Data = np.transpose(Data)
        np.savetxt(WritefilePath, Data, fmt='%01.3e', delimiter='      ')

    def export_to_comsol(self, Zoffset=None):
        """
        Generate a phase space which can be directly imported into CST
        For a constant emission model: generate a .pid ascii file
        Below is the example from CST:

        % Use always SI units.
        % The momentum (mom) is equivalent to beta* gamma.
        %
        % Columns: pos_x  pos_y  pos_z  mom_x  mom_y  mom_z  mass  charge  current

        1.0e-3   4.0e-3  -1.0e-3   1.0   2.0   1.0   9.11e-31  -1.6e-19   1.0e-6
        2.0e-3   4.0e-3   1.0e-3   1.0   2.0   1.0   9.11e-31  -1.6e-19   1.0e-6
        3.0e-3   2.0e-3   1.0e-3   1.0   2.0   2.0   9.11e-31  -1.6e-19   1.0e-6
        4.0e-3   4.0e-3   5.0e-3   1.0   2.0   1.0   9.11e-31  -1.6e-19   2.0e-6
        """
        # Split the original file and extract the file name
        NparticlesToWrite = np.size(self.x)  # use this to create a smaller PID flie for easier trouble shooting
        WritefilePath = self.OutputDataLoc + '/' + self.OutputFile + '.txt'
        # generate other information required by pid file:

        x = self.x * 1e-3  ## convert to m
        y = self.y * 1e-3
        if Zoffset == None:
            # Zoffset is an optional parameter to change the starting location of the particle beam (which
            # assume propogates in the Z direction)
            self.zOut = self.z
        else:
            self.zOut = (self.z + Zoffset)
        # generate PID file
        Data = [x, y, self.zOut, self.vx*self._c, self.vy*self._c, self.vz*self._c]

        Data = np.transpose(Data)
        np.savetxt(WritefilePath, Data, fmt='%01.12e', delimiter='      ')

    def export_to_topas(self, Zoffset=None):
        """
        Convert Phase space into format appropriate for topas.
        You can read more about the required format
        `Here <https://topas.readthedocs.io/en/latest/parameters/scoring/phasespace.html>`_

        :param Zoffset: number to add to the Z position of each particle. To move it upstream, Zoffset should be negative.
         No check is made for units, the user has to figure this out themselves! If Zoffset=None, the read in X value
         will be used.
        :type Zoffset: None or double
        """
        print('generating topas data file')
        import platform
        if 'windows' in platform.system().lower():
            warnings.warn('to generate a file that topas will accept, you need to do this from linux. I think'
                          'its the line endings.')
        WritefilePath = self.OutputDataLoc + '/' + self.OutputFile + '_tpsImport.phsp'

        # write the header file:
        self.__GenerateTopasHeaderFile()

        # generare the required data and put it all in an ndrray
        self.__ConvertMomentumToVelocity()
        DirCosX, DirCosY = self.__CalculateDirectionCosines()
        Weight = self.weight  # i think weight is scaled relative to particle type
        # Weight = np.ones(len(self.x))  # i think weight is scaled relative to particle type
        ParticleType = 11 * np.ones(len(self.x))  # 11 is the particle ID for electrons
        ThirdDirectionFlag = np.zeros(len(self.x))  # not really sure what this means.
        FirstParticleFlag = np.ones(
            len(self.x))  # don't actually know what this does but as we import a pure phase space
        if Zoffset == None:
            # Zoffset is an optional parameter to change the starting location of the particle beam (which
            # assume propogates in the Z direction)
            self.zOut = self.z
        else:
            self.zOut = self.z + Zoffset

        # Nb: topas seems to require units of cm
        Data = [self.x * 0.1, self.y * 0.1, self.zOut * 0.1, DirCosX, DirCosY, self.E, Weight,
                ParticleType, ThirdDirectionFlag, FirstParticleFlag]

        # write the data to a text file
        Data = np.transpose(Data)
        FormatSpec = ['%11.5f', '%11.5f', '%11.5f', '%11.5f', '%11.5f', '%11.5f', '%11.5f', '%2d', '%2d', '%2d']
        np.savetxt(WritefilePath, Data, fmt=FormatSpec, delimiter='      ')
        print('success')

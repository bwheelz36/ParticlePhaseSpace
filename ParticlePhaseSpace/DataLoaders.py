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
            warnings.warn(f'{n_negative_locations: d} entries returned invalid pz values and were set to zero.'
                          f'\nWe will now check that momentum and energy are consistent to within '
                          f'{self._energy_consistency_check_cutoff: 1.4f} {self._units.energy.label}')

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
        self.data[self._columns['x']] = Data[:, 0] * 1e3
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
    'txt' loader
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


class Load_p2sat_Smilei_Screen_1d(_DataLoadersBase):
    pass

class Load_p2sat_Smilei_TrackParticles(_DataLoadersBase):
    pass

class Load_p2sat_gp3m2_csv(_DataLoadersBase):
    pass

class Load_p2sat_TrILEns_output(_DataLoadersBase):
    pass

class Load_p2sat_TrILEns_prop_ph(_DataLoadersBase):
    pass



def Smilei_Screen_1d(self,path,nb,r,x=0,verbose=True):
    r"""
    Extract phase space from Smilei 1D Screen diagnostic.
    Parameters
    ----------
    path : str
        path to the simulation folder
    nb : int
        Screen number
    r : float
        typical radius to consider in transverse direction (in um)
    x : float, optional
        diagnostic position
    verbose : bool, optional
    Notes
    -----
    On a 1D Smilei simulation, a typical DiagScreen must be declared as follows
    ::
        DiagScreen(
            shape               = 'plane',
            point               = [xtarget[1] - 5*um],
            vector              = [1.],
            direction           = 'forward',
            deposited_quantity  = 'weight',
            species             = ['e'],
            axes                = [
                 ['px' , pmin     , pmax     , 301],
                 ['py' , -pmax/5  , pmax/5   , 301]
            ],
            every               = every
        )
    """
    if verbose: print("Extracting screen data from %s ..."%path)

    # Import Smilei simulation
    import happi
    S = happi.Open(path,verbose=False)
    nl = S.namelist

    # Define physical constants
    m_e = 9.11e-31
    epsilon_0 = 8.85e-12
    e = 1.6e-19
    c = 2.99792458e8
    epsilon_0 = 8.854187817e-12
    # Smilei's unit in SI
    Wr = nl.Main.reference_angular_frequency_SI
    Tr = 1/Wr
    Lr = c/Wr
    Pr = 0.511 # MeV/c
    # Calculate normalizations
    nc = m_e * epsilon_0 * (Wr/e)**2
    Lx = nl.Main.grid_length[0] * Lr # Use a try/except ?
    vol = Lx * np.pi * (r * 1e-6)**2
    wnorm = nc * vol # Weight normalization : Before -> in Nc/Pr/Pr, After -> in Number/Pr/Pr
    tnorm = 1e-15/Tr
    xnorm = 1e-6/Lr
    # Save diag position
    xdiag = x

    # Initialize phase space lists
    w         = []
    x,y,z     = [],[],[]
    px,py,pz  = [],[],[]
    t         = []

    # Retrieve Screen data
    times = S.Screen(nb).getTimes()
    timesteps= S.Screen(nb).getTimesteps()

    Px  = S.Screen(nb).getAxis("px") * Pr
    Py  = S.Screen(nb).getAxis("py") * Pr

    # Compensate happi correction on weights
    wnorm /= Pr**2 # Weights are now in Nb/(MeV/c)/(MeV/c) (independant of bin size)
    wnorm *= (max(Px)-min(Px))/len(Px) # Multiply by bin size : weights are now in Nb/(MeV/c)/bin
    wnorm *= (max(Py)-min(Py))/len(Py) # Weight are now in Nb/bin/bin (dependant of bin size, it counts number of particles for given conf)

    # Current data is initialized as an empty matrix
    cdata=np.array([[0.]*len(Px)]*len(Py))

    # Loop over times
    for it,et in enumerate(timesteps):
        ldata = cdata
        # Retrieve data for given time
        cdata = S.Screen(nb,timesteps=et).getData()[0]
        # Loop over px then py
        if verbose and it % (len(times)//10) == 0: print("Retrieving timestep %i/%i ..."%(et,timesteps[-1]))
        for ipx,epx in enumerate(cdata):
            for ipy,epy in enumerate(epx):
                # Get weight difference for given configuration
                depy = epy-ldata[ipx][ipy]
                # If non-zero, save config
                if depy!=0.:
                    w.append(depy * wnorm)
                    px.append(Px[ipx])
                    py.append(Py[ipy])
                    t.append(times[it] * tnorm)

    # Reconstruct missing data
    pz = [0.0] * len(w)
    x = [xdiag] * len(w)
    y = [0.0] * len(w)
    z = [0.0] * len(w)

    # Update current phase space
    if verbose: print("Done !")
    self._ps.edit.update(w,x,y,z,px,py,pz,t,in_code_units=True,verbose=verbose)

def Smilei_TrackParticles(self,path,species,dscale=1.,verbose=True):
    r"""
    Extract phase space from a TrackParticles Smilei diagnostic.
    Parameters
    ----------
    path : str
        path to the simulation folder
    species : str
        name of the specie in the Smilei namelist
    dscale : float
        Typical diameter to consider in the transverse direction if needed (in 1D or 2D). Should be given in meters.
    verbose : bool, optional
        verbosity
    """
    if verbose: print("Extracting %s phase space from %s TrackParticles ..."%(self._ps.metadata.specie["name"],species))
    # Open simulation
    import happi
    S = happi.Open(path,verbose=False)
    nl = S.namelist

    # Define physical constants
    m_e = 9.11e-31
    epsilon_0 = 8.85e-12
    e = 1.6e-19
    c = 2.99792458e8
    epsilon_0 = 8.854187817e-12

    # Smilei's unit in SI
    Wr = nl.Main.reference_angular_frequency_SI
    Tr = 1/Wr
    Lr = c/Wr
    Pr = 0.511 # MeV/c

    # Calculate normalizations
    geom = nl.Main.geometry
    nc = m_e * epsilon_0 * (Wr/e)**2
    if geom == "1Dcartesian":
        wnorm = nc * Lr * np.pi * dscale**2
    elif geom == "2Dcartesian":
        wnorm = nc * Lr**2 * dscale
    elif geom == "AMcylindrical":
        wnorm = nc * Lr**3
    elif geom == "3Dcartesian":
        wnorm = nc * Lr**3
    else:
        raise NameError("Unknown geometry profile.")
    # tnorm = Tr/1e-15    # in fs
    # xnorm = Lr/1e-6     # in um
    # pnorm = Pr          # in MeV/c
    tnorm = Tr          # in s
    xnorm = Lr          # in m
    pnorm = Pr*1e6      # in eV/c

    # Initialize ps list
    w         = []
    x,y,z     = [],[],[]
    px,py,pz  = [],[],[]
    t         = []

    # Get timesteps
    timesteps = S.TrackParticles(species=species,sort=False).getTimesteps()
    dt = nl.Main.timestep

    # Loop over timesteps
    for ts in timesteps:
        if verbose:print("Timestep %i/%i ..."%(ts,timesteps[-1]))
        # Get data from current timestep
        data = S.TrackParticles(species=species,timesteps=ts,sort=False).get()[ts]
        # Get macro-particle's id. id == 0 means the macro-particle have already been exported
        id = data["Id"]
        # If no positive id, continue to the next iteration
        if len(id[id>0]) == 0: continue
        # Append phase space data of current timestep
        w += list(data["w"][id>0] * wnorm)
        x += list(data["x"][id>0] * xnorm)
        if geom == "1Dcartesian":
            y += [0.] * len(id>0)
            z += [0.] * len(id>0)
        elif geom == "2Dcartesian":
            y += list(data["y"][id>0] * xnorm)
            z += [0.] * len(id>0)
        elif geom == "AMcylindrical":
            y += list(data["y"][id>0] * xnorm)
            z += list(data["z"][id>0] * xnorm)
        elif geom == "3Dcartesian":
            y += list(data["y"][id>0] * xnorm)
            z += list(data["z"][id>0] * xnorm)
        px += list(data["px"][id>0] * pnorm)
        py += list(data["py"][id>0] * pnorm)
        pz += list(data["pz"][id>0] * pnorm)
        t += [ts*dt * tnorm] * len(id[id>0])

    if verbose: print("Done !")

    self._ps.edit.update(w,x,y,z,px,py,pz,t,in_code_units=True,verbose=verbose)

def gp3m2_csv(self,base_name,path="./",thread=None,multiprocessing=False,in_code_units=False,verbose=True):
    r"""
    Extract simulation results from a gp3m2 NTuple csv output file
    Parameters
    ----------
    base_name : str
        base file name
    path : str
        path to the simulation folder
    thread : int, optional
        number of the thread to import. By default it get the data of all the threads
    multiprocessing : bool, optional
        use or not the multiprocessing to paralelize import. Incompatible with thread != None.
    verbose : bool, optional
        verbosity
    Examples
    --------
    For the gp3m2 output file name `Al_target_nt_electron_t0.csv`, the base_name
    is `Al_target`.
    Assuming a `p2sat.PhaseSpace` object is instanciated for particle `e-` as eps,
    you can import simulation results for all the threads as follows
    >>> eps = ExamplePhaseSpace()
    >>> # eps.extract.gp3m2_csv("Al_target")
    """
    # Get gp3m2 particle name from p2sat particle name
    part = self._ps.metadata.specie["name"]
    if part=="e-":
        part_name = "electron"
    elif part=="e+":
        part_name = "positron"
    elif part=="gamma":
        part_name = "gamma"
    elif part=="photon":
        part_name = "OpPhoton"

    # Construct file base name
    fbase = base_name+"_nt_"+part_name+"_t"
    fext = ".csv"

    if multiprocessing and thread is None:
        # Import modules
        import os
        import multiprocessing as mp

        # Create the queue
        q = mp.Manager().Queue()
        # Create the loading function, that putting data in the queue
        loader = lambda fname: q.put(np.loadtxt(fname, delimiter=","))

        # Initialize thread id, data array and list of all processes
        id = 0
        data = np.array([])
        processes = []
        # Loop over threads
        while True:
            fname = path + fbase + str(id) + fext
            # Check if the file name is in the given folder
            if os.path.isfile(fname):
                if verbose:print("Extracting %s ..."%fname)
                # Call the loader function for each thread
                proc = mp.Process(target=loader, args=(fname,))
                processes.append(proc)
                proc.start()
                id += 1
            else:
                break

        # Retrieve data
        for proc in processes:
            proc.join()

        while not q.empty():
            data = np.append(data, q.get())
    else:
        # Initialize data list
        data = []
        # Loop over threads
        id = 0
        while True:
            # Construct file name for current thread
            if thread is not None:
                fname = path + fbase + str(thread) + fext
            else:
                fname = path + fbase + str(id) + fext
                id    += 1
            # Try to append data
            try:
                # Open file for thread id-1
                with open(fname,'r') as f:
                    if verbose:print("Extracting %s ..."%fname)
                    # Loop over lines
                    for line in f.readlines():
                        # Save data if current line is not a comment
                        if line[0]!='#':
                            for e in line.split(','):
                                data.append(float(e))
            # If no more thread, break the loop
            except IOError:
                break
            # If only one thread, break the loop
            if thread is not None:
                break

    # Get phase space from data list
    w   = data[0::8]
    x   = data[1::8]
    y   = data[2::8]
    z   = data[3::8]
    px  = data[4::8]
    py  = data[5::8]
    pz  = data[6::8]
    t   = data[7::8]
    if verbose:print("Done !")

    # Save phase space data in PhaseSpace object
    self._ps.edit.update(w,x,y,z,px,py,pz,t,in_code_units=in_code_units,verbose=verbose)

def TrILEns_output(self,path,verbose=True):
    r"""
    Extract simulation results from a TrILEns output.txt file
    Parameters
    ----------
    path : str
        simulation path
    verbose : bool, optional
        verbosity
    """
    particle = self._ps.metadata.specie["name"]
    if verbose:print("Extracting {} phase space from {}output.txt ...".format(particle,path))

    # Get TrILEns particle label from p2sat particle name
    if particle == "e-":
        label = "electrons"
    elif particle == "e+":
        label = "positrons"
    elif particle == "gamma":
        label = "photons"

    # Initialize phase space lists
    w         = []
    x,y,z     = [],[],[]
    px,py,pz  = [],[],[]
    t         = []

    # Boolean to extract only the data of correct particle
    is_correct_species=False

    # Open output file
    with open(path+'output.txt','r') as f:
        # 34 first lines are informations about the simulation
        for _ in range(3):
            f.readline()
        line = f.readline()
        if line.split()[1]=="T":
            chi_to_t = True
        else:
            chi_to_t = False
        for _ in range(30):
            f.readline()
        # Loop over data
        for line in f.readlines():
            try:
                # Photons do not have Chi value
                if label == "photons":
                    W,X,Y,Z,Px,Py,Pz,Gamma=line.split()
                    chi_to_t = False
                else:
                    W,X,Y,Z,Px,Py,Pz,Gamma,Chi=line.split()
                # If correct particle, save data
                if is_correct_species:
                    w.append(float(W))
                    x.append(float(X))     ; y.append(float(Y))   ; z.append(float(Z))
                    px.append(float(Px)*0.511)   ; py.append(float(Py)*0.511) ; pz.append(float(Pz)*0.511)
                    if chi_to_t:
                        t.append(float(Chi)*1e3) # convert ps to fs
                    else:
                        t.append(0.)

            # If current line is a string (not possible to read data), test if particle label in current line
            except ValueError:
                if label in line.split():
                    is_correct_species = True
                else:
                    is_correct_species = False

    if verbose:print("Done !")

    # Save data in PhaseSpace object
    self._ps.edit.update(w,x,y,z,px,py,pz,t,in_code_units=True,verbose=verbose)

def TrILEns_prop_ph(self,path,verbose=True):
    r"""
    Extract simulation results from a TrILEns prop_ph file
    Parameters
    ----------
    path : str
        simulation path
    verbose : bool, optional
        verbosity
    """
    if self._ps.metadata.specie["name"]!="gamma":
        raise NameError("prop_ph.t contains informations about gamma photons ! Current particle name is %s"%self._ps.metadata.specie["name"])
    if verbose: print("Extracting %s phase space from %s ..."%(self._ps.metadata.specie["name"],path+"prop_ph.t"))

    # Initialize data lists
    w         = []
    x,y,z     = [],[],[]
    px,py,pz  = [],[],[]
    t         = []

    with open(path+"prop_ph.t",'r') as f:
        # First line gives information about time
        line = f.readline()
        if line == "8 1.\n":
            with_time = False
        elif line == "9 1.\n":
            with_time = True
        else:
            raise NameError("Unknown time identifier at line 1 : %s"%line)
        # second line is a legend
        _ = f.readline()
        # Loop over data lines
        for line in f.readlines():
            # If current line is not a comment, save data
            data=line.split()
            w.append(float(data[0]))
            x.append(float(data[1]))  ; y.append(float(data[2]))  ; z.append(float(data[3]))
            px.append(float(data[4])) ; py.append(float(data[5])) ; pz.append(float(data[6]))
            if with_time:
                t.append(float(data[8])*1e3)
            else:
                t.append(0.)

    if verbose: print('Done !')

    self._ps.edit.update(w,x,y,z,px,py,pz,t,in_code_units=True,verbose=verbose)

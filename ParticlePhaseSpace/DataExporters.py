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
    relative_weight = self.weight / total_weight
    Current = self.TotalCurrent * relative_weight  # very crude approximation!!
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
    Data = [x, y, self.zOut, self.vx * self._c, self.vy * self._c, self.vz * self._c]

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

def _CalculateBetaAndGamma(self):
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

def _CalculateDirectionCosines(self):
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

def _GenerateTopasHeaderFile(self):
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

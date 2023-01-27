from scipy import constants

class _unit:
    """
    Each unit has two attributes:

    :param label: the string of the unit. This is used for labelling graphs etc.
    :param conversion: the factor needed to convert the unit to the native unit set mm_MeV.

    conversion = this_unit / native_unit_set_unit
    e.g. for cm, conversion = .1/1 = .1
    """
    def __init__(self, label: str, conversion: float):
        self.label = label
        self.conversion = conversion

class _UnitSet:

    def __init__(self, length_units: _unit,
                 energy_units: _unit,
                 momentum_units: _unit,
                 velocity_units: _unit,
                 time_units: _unit,
                 mass_units: _unit):
        
        self.length = length_units
        self.energy = energy_units
        self.momentum = momentum_units
        self.velocity = velocity_units
        self.time = time_units
        self.mass = mass_units

    def __str__(self):
        string_rep = f'Length: {self.length.label},' \
                     f'\nEnergy: {self.energy.label},' \
                     f'\nMomentum: {self.momentum.label}' \
                     f'\nVelocity: {self.velocity.label}' \
                     f'\nMass: {self.mass.label}'
        return string_rep

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text('MyList(...)')
        else:
            p.text(str(self))


class ParticlePhaseSpaceUnits:

    def __init__(self):
        self.mm_MeV = _UnitSet(length_units=_unit('mm', 1),
                              energy_units=_unit('MeV', 1),
                              momentum_units=_unit('MeV/c', 1),
                               time_units=_unit('ps', 1),
                               mass_units=_unit('MeV/c^2', 1),
                               velocity_units=_unit('m/s', 1))

        self.cm_MeV = _UnitSet(length_units=_unit('cm', 1e-1),
                              energy_units=_unit('MeV', 1),
                              momentum_units=_unit('MeV/c', 1),
                               time_units=_unit('ps', 1),
                               mass_units=_unit('MeV/c^2', 1),
                               velocity_units=_unit('m/s', 1))

        self.um_keV = _UnitSet(length_units=_unit('um', 1e3),
                              energy_units=_unit('keV', 1e3),
                              momentum_units=_unit('keV/c', 1e3),
                               time_units=_unit('ps', 1),
                               mass_units=_unit('keV/c^2', 1e3),
                               velocity_units=_unit('m/s', 1))

        self.SI = _UnitSet(length_units=_unit('cm', 1e-1),
                          energy_units=_unit('MeV', 1),
                          momentum_units=_unit('MeV/c', 1),
                            time_units=_unit('s', 1),
                           mass_units=_unit('kg', constants.elementary_charge * 1e6 / (constants.c**2)),
                           velocity_units=_unit('m/s', 1))

        self._check_attributes()

    def __call__(self, unit_tag):

        try:
            return getattr(self, unit_tag)
        except AttributeError:
            raise AttributeError(f'unrecognised unit set {unit_tag}. '
                                 f'\nFor a list of valid unit sets please call print on this object')


    def _check_attributes(self):
        attributes = filter(lambda a: not a.startswith('__'), dir(self))
        for attribute in attributes:
            if attribute in ['_check_attributes', '_repr_pretty_']:
                continue
            if not isinstance(getattr(self, attribute), _UnitSet):
                raise TypeError('ParticlePhaseSpaceUnits units must have only _UnitSet type attributes')


    def __str__(self):

        attributes = filter(lambda a: not a.startswith('__'), dir(self))
        attribute_string = 'Available Unit Sets'
        attribute_string = '\n-------------------\n'
        for attribute in attributes:
            if attribute in ['_check_attributes', '_repr_pretty_']:
                continue
            attribute_string = attribute_string + attribute + '\n'
        attribute_string = attribute_string[:-1]

        return attribute_string

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text('MyList(...)')
        else:
            p.text(str(self))


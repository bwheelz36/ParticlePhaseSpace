from scipy import constants

class _unit:
    """
    Each unit has two attributes:

    :param label: the string of the unit. This is used for labelling graphs etc.
    :type label: str
    :param conversion: the factor needed to convert the unit to the native unit set mm_MeV.
    :type conversion: float

    conversion = this_unit / native_unit_set_unit
    e.g. for cm, conversion = .1/1 = .1
    """
    def __init__(self, label: str, conversion: float):
        self.label = label
        self.conversion = conversion


class UnitSet:
    """
    Class storing a complet set of units required for representing PhaseSpace data.
    For documentation on working with units see `here <https://bwheelz36.github.io/ParticlePhaseSpace/units.html>`_


    :param label: The name of the unit set
    :type label: str
    :param length_units: e.g. _unit('mm', 1)
    :type length_units: _unit
    :param energy_units: e.g. _unit('MeV', 1)
    :type energy_units: _unit
    :param momentum_units: e.g _unit('MeV/c', 1)
    :type momentum_units: _unit
    :param velocity_units:  e.g. _unit('m/s', 1)
    :type velocity_units: _unit
    :param time_units: e.g. _unit('s', 1)
    :type time_units: _unit
    :param mass_units: e.g. _unit('MeV/c^2',1)
    :type mass_units: _unit
    """

    def __init__(self, label: str,
                 length_units: _unit,
                 energy_units: _unit,
                 momentum_units: _unit,
                 velocity_units: _unit,
                 time_units: _unit,
                 mass_units: _unit):

        self.label = label
        self.length = length_units
        self.energy = energy_units
        self.momentum = momentum_units
        self.velocity = velocity_units
        self.time = time_units
        self.mass = mass_units

    def __str__(self):
        string_rep = f'{self.label}:' \
                     f'\n=================' \
                     f'\nLength: {self.length.label},' \
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

    def _check_attributes(self):
        attributes = filter(lambda a: not a.startswith('__'), dir(self))
        for attribute in attributes:
            if attribute in ['_check_attributes', '_repr_pretty_']:
                continue
            if not isinstance(getattr(self, attribute), _unit):
                raise TypeError('_UnitSet units should have only _unit type attributes')


class ParticlePhaseSpaceUnits:

    def __init__(self):
        self.mm_MeV = UnitSet(label='mm_MeV',
                              length_units=_unit('mm', 1),
                              energy_units=_unit('MeV', 1),
                              momentum_units=_unit('MeV/c', 1),
                              time_units=_unit('ps', 1),
                              mass_units=_unit('MeV/c^2', 1),
                              velocity_units=_unit('m/s', 1))

        self.cm_MeV = UnitSet(label='cm_MeV',
                              length_units=_unit('cm', 1e-1),
                              energy_units=_unit('MeV', 1),
                              momentum_units=_unit('MeV/c', 1),
                              time_units=_unit('ps', 1),
                              mass_units=_unit('MeV/c^2', 1),
                              velocity_units=_unit('m/s', 1))

        self.um_keV = UnitSet(label='um_keV',
                              length_units=_unit('um', 1e3),
                              energy_units=_unit('keV', 1e3),
                              momentum_units=_unit('keV/c', 1e3),
                              time_units=_unit('ps', 1),
                              mass_units=_unit('keV/c^2', 1e3),
                              velocity_units=_unit('m/s', 1))

        self.m_eV = UnitSet(label='m_eV',
                              length_units=_unit('m', 1e-3),
                              energy_units=_unit('eV', 1e6),
                              momentum_units=_unit('eV/c', 1e6),
                              time_units=_unit('s', 1e-9),
                              mass_units=_unit('eV/c^2', 1e6),
                              velocity_units=_unit('m/s', 1))

        self._get_unit_attributes()
        self._check_attributes()

    def __call__(self, unit_tag):
        try:
            return getattr(self, unit_tag)
        except AttributeError:
            raise AttributeError(f'unrecognised unit set {unit_tag}. '
                                 f'\nFor a list of valid unit sets please call print on this object')

    def __str__(self):
        attribute_string = 'Available Unit Sets'
        attribute_string = attribute_string + '\n=================\n'
        for attribute in self._attributes:
            unit_string = self(attribute).__str__()
            attribute_string = attribute_string + unit_string + '\n'
            attribute_string = attribute_string + '\n=================\n'
        attribute_string = attribute_string[:-19]

        return attribute_string

    def _get_unit_attributes(self):
        all_attributes = filter(lambda a: not a.startswith('__'), dir(self))
        self._attributes = []
        for attribute in all_attributes:
            if attribute in ['_check_attributes', '_repr_pretty_', '_get_unit_attributes', 'get_available_unit_strings']:
                continue
            self._attributes.append(attribute)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text('MyList(...)')
        else:
            p.text(str(self))

    def _check_attributes(self):
        for attribute in self._attributes:
            if not isinstance(getattr(self, attribute), UnitSet):
                raise TypeError('ParticlePhaseSpaceUnits units must have only _UnitSet type attributes')

    def get_available_unit_strings(self):
        return self._attributes

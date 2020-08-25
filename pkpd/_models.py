#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import collections as cs

import myokit
import numpy as np
import pints


class Model(object):
    """
    PKPD Model building class.

    Model building relies almost entirely on the myokit model api. Wrapper
    adds functionality to simplify route of administration specification and
    parameter value as well as unit setting.
    """
    def __init__(self, name=None):
        self._model = myokit.Model()
        self._roa_set = False  # Route of administration
        self._dose_comp = None
        self._depot_comp = None
        self._dose_rate = None
        self._regimen = None

    def add_compartment(self, name):
        """
        Adds and returns a model compartment using `myokit.add_component`.
        """
        return self._model.add_component(name)

    def check_units(self, mode=myokit.UNIT_TOLERANT):
        """
        Checks consistency of units using myokit.check_units.
        """
        self._model.check_units(mode)

    def clone(self):
        """
        See myokit.Model.cone.
        """
        return self._model.clone()

    def code(self, line_numbers=False):
        """
        Returns this model in ``mmt`` syntax using myokit.Model.code.
        """
        return self._model.code(line_numbers)

    def is_roa_set(self):
        """
        Returns flag whether route of administration has been set.
        """
        return self._roa_set

    def is_valid(self):
        """
        See myokit.Model.is_valid.
        """
        return self._model.is_valid()

    def set_roa(self, dose_comp, indirect):
        """
        Sets route of administration.

        Arguments:
            dose_comp -- Compartment that is either infused with drug (direct)
                         or connected to the depot compartment (indirect).
            indirect -- Flag whether dose is administered directly (False) into
                        the dose compartment or indriectly (True) through a
                        depot.
        """
        # Check that model contains dose_comp
        if not self._model.has_component(dose_comp):
            raise ValueError

        # Check that dose compartment has a variable called amount
        amount = dose_comp + '.amount'
        if not self._model.has_variable(amount):
            raise ValueError

        # Check that amount is a state variable
        amount = self._model.var(amount)
        if not amount.is_state():
            raise ValueError

        # Check that model has time bound variable
        time = self._model.binding(binding='time')
        if time is None:
            raise ValueError

        # Get amount and time unit from dose compartment
        amount_unit = amount.unit()
        time_unit = time.unit()

        # TODO: Remove previously set roa compartments, variables and
        # expressions.

        # Remember name of dose_comp
        self._dose_comp = dose_comp

        # Get dose compartment object
        dose_comp = self._model.get(dose_comp)

        # Add depot compartment if dose is administered indirectly
        if indirect:
            # Add depot compartment
            depot_comp = self._model.add_component_allow_renaming('depot')

            # Remember name of depot compartment
            self._depot_comp = depot_comp.name()

            # Add drug amount and absorption rate to the depot compartment
            amount_de = depot_comp.add_variable('amount')
            k_a = depot_comp.add_variable('k_a')

            # Set default values for amount and absorption rate
            amount_de.set_rhs(0)
            k_a.set_rhs(0)

            # Set units
            amount_de.set_unit(amount_unit)
            k_a.set_unit(1/time_unit)

            # Promote amount to state variable
            amount_de.promote()

            # Add outflow expression to depot compartment
            amount_de.set_rhs(myokit.Multiply(
                myokit.PrefixMinus(myokit.Name(k_a)),
                myokit.Name(amount_de)
            ))

            # Add inflow expression to dose compartment
            expr = amount.rhs()
            amount.set_rhs(myokit.Plus(
                expr,
                myokit.Multiply(
                    myokit.Name(k_a),
                    myokit.Name(amount_de)
                )
            ))

            # Set the depot to the dose compartment
            dose_comp = depot_comp
            amount = amount_de

            del depot_comp
            del amount_de

        # Add dose rate and regimen variables
        dose_rate = dose_comp.add_variable_allow_renaming('dose_rate')
        regimen = dose_comp.add_variable_allow_renaming('regimen')

        # Remember dose rate variable name
        comp_name = dose_comp.name()
        self._dose_rate = comp_name + '.' + dose_rate.name()

        # Set default values for dose_rate and regimen
        dose_rate.set_rhs(0)
        regimen.set_rhs(0)

        # Set units
        dose_rate.set_unit(amount_unit/time_unit)
        regimen.set_unit('dimensionless')

        # Bind regimen to myokit pacer (so regimen can be adjusted with
        # myokit.Protocol)
        regimen.set_binding('pace')

        # Add dose infusion expression to dose compartment
        expr = amount.rhs()
        amount.set_rhs(myokit.Plus(
            expr,
            myokit.Multiply(
                myokit.Name(dose_rate),
                myokit.Name(regimen)
            )
        ))

        # Flag route of administration to be set
        self._roa_set = True

        # Check whether a valid model has been created
        self.validate()

    def dose_comp(self):
        """
        Returns the name of the compartment the dose is applied to.
        """
        if self._depot_comp:
            return self._depot_comp
        return self._dose_comp

    def _set_units(self, var, unit):
        """
        Sets the unit of a variable.
        """
        # Set variable unit
        v = self._model.var(var)
        v.set_unit(unit)

        # Set preferred representation of unit
        u = v.unit()
        myokit.Unit.register_preferred_representation(unit, u)

    def _set_parameter_values(self, parameter, value):
        """
        Sets the value of a parameter.
        """
        param = self._model.var(parameter)
        param.set_rhs(value)

    def _set_initial_values(self, state, value):
        """
        Sets the initial value of a state.
        """
        s = self._model.var(state)
        s.set_state_value(float(value))

    def set_values(self, name, value, unit=None):
        """
        Sets values of variable names in the specified unit.

        For state variables the initial value is set. For variables defined
        by an algerbaic expression an error is thorwn.

        Arguments:
            name -- String or list of strings of variable names.
            value -- Float or list of floats of values.
            unit -- Unit or list of units specfications that can be parsed by
                    myokit.
        """
        # If a single name is passed, it's converted to a list
        if isinstance(name, str):
            name = [name]

        # If a single value is passed, it's converted to a list
        if isinstance(value, str) or not isinstance(
                value, cs.Iterable):
            value = [value]

        # Check that each name has a value
        if len(name) != len(value):
            raise ValueError

        # If a single unit is passed, all variables are assigned with that unit
        if isinstance(unit, str) or not isinstance(unit, cs.Iterable):
            unit = [unit] * len(name)

        # Check that every variable has a unit
        if len(name) != len(unit):
            raise ValueError

        # Map names, values and units
        # TODO: Refactor this bit
        try:
            name_val_unit_zip = zip(name, value, unit)
        except ValueError:
            raise ValueError(
                'Try passing `name`, `value` and `unit` as lists.')

        for n, v, u in name_val_unit_zip:
            # Check that variable is defined
            if not self._model.has_variable(n):
                raise ValueError

            # Set variable value
            var = self._model.var(n)
            if var.is_intermediary():
                raise ValueError
            elif var.is_state():
                self._set_initial_values(state=n, value=v)
            else:
                self._set_parameter_values(parameter=n, value=v)

            # Set unit of variable
            if u is not None:
                self._set_units(var=n, unit=u)

    def set_regimen(
            self, amount, duration=1.0E-6, start=0, period=0, multiplier=0):
        """
        Sets the dosing regimen.

        The dosing regimen is set by updating the dose_rate variable in the
        dosing compartment to dose_rate = amount / duration and setting the
        regimen variable to a myokit.Protocol with the specified regimen. If a
        route of administration has not been set with `self.set_roa()` an
        error will be thrown.

        Arguments:
            amount -- Amount of injected dose. Units are specified by
                      dose_rate.
            duration -- Duration of injection. Units are specfied by the
                        global time.

        Keyword Arguments:
            start -- See myokit.Protocol documentation. (default: {0})
            period -- See myokit.Protocol documentation. (default: {0})
            multiplier -- See myokit.Protocol documentation. (default: {0})
        """
        # Check whether route of administration has been set.
        if not self._roa_set:
            raise ValueError

        # Set dose rate variable to amount / duration
        # TODO: handle ZeroDivionError and inf
        var = self._model.var(self._dose_rate)
        var.set_rhs(amount / duration)

        # Set regimen
        # TODO: check whether more efficient to clean and fill protocol
        self._regimen = myokit.Protocol()
        self._regimen.schedule(
            level=1,
            start=start,
            duration=duration,
            period=period,
            multiplier=multiplier
        )

    def states(self):
        """
        See myokit.model.states.
        """
        return self._model.states()

    def parameters(self):
        """
        Returns an iterator over the parameters of the model.
        """
        return self._model.variables(const=True)

    def dosing_regimen(self):
        """
        Returns dose rate and regimen.
        """
        return self._dose_rate, self._regimen

    def validate(self):
        """
        Validates model using myokit.validate.
        """
        self._model.validate()

    def var(self, name):
        """
        See myokit.Model.var.
        """
        return self._model.var(name)


# TODO: CHECK whether Model inheritance screws something up
class TumourGrowthModel(pints.ForwardModel, Model):
    r"""
    Creates a `pints.ForwardModel` which wraps a tumour growth model of the
    form

    .. math::
        \frac{\text{d}V^s_T}{\text{d}t} =
        \frac{2\lambda _0\lambda _1 V^s_T}{2\lambda _0 V^s_T + \lambda _1}.

    Here, :math:`V^s_T` is the predicted tumour volume by the structural model,
    :math:`\lambda _0` is the exponential growth rate, and :math:`\lambda _1`
    is the linear growth rate.

    If the flag `dimless` is `True` the dimensionless representation

    .. math::
        \frac{\text{d}v}{\text{d}\tau} =
        \frac{a_1v}{v + a_0},

    is insantiated instead, where (:math:`v`, :math:`\tau `) are related to
    (:math:`V^s_T`, :math:`t`) by some characteristic volume and time scale
    (:math:`V^c`, :math:`t^c`).

    The model is parameterised by either (:math`V_0`, :math:`\lambda _0`,
    :math:`\lambda _1`) or (:math`v_0`, :math:`a _0`, :math:`a _1`) which
    are related by

    .. math::
        (V_0, \lambda _0, \lambda _1) =
        \left( v_0\, V^c_T, \frac{a_1}{2a_0}\frac{1}{t^c},
        a_1 \frac{V^c_T}{t^c} \right) .

    Arguments:
        is_treated -- Boolean flag indicating whether the tumour is simulated
                      in absence or in presence of treatment.
        dimless -- Boolean flag indicating whether the model is used in its
                   dimensionfull or dimensionless representation.
        log_transformed -- Boolean flag which indicates whether the model
                           parameters are assumed to be log-transformed.
    """
    def __init__(self, is_treated=False, dimless=False, log_transformed=False):
        super(TumourGrowthModel, self).__init__()

        if is_treated:
            raise NotImplementedError

        self._dimless = dimless
        self._log_transformed = log_transformed

        if dimless:
            self._model = self._create_dimless_tumour_growth_model()
        else:
            self._model = self._create_tumour_growth_model()

        # Create simulator
        self._sim = myokit.Simulation(self._model)

    def n_parameters(self):
        """
        Returns the number of parameters to fit. Either the initial tumour
        volume, the exponential tumour growth and the linear tumour growth, or
        the three equivalent dimensionless parameters.
        """
        return 3

    def n_outputs(self):
        """
        Returns the number of output dimensions. Here 1 (the tumour volume).
        """
        return 1

    def simulate(self, parameters, times):
        """
        Solve the system of ODEs numerically for given parameters.

        Parameters are assumed to be ordered as follows:
        Dimensionfull model:
            [initial tumour volume, exp. growth rate, lin. growth rate]
        Dimensionless model:
            [initial tumour volume, a_0, a_1].
        """
        # Reset simulation
        self._sim.reset()

        # Sort input parameters
        initial_state, param_0, param_1 = np.exp(parameters) if \
            self._log_transformed else parameters

        # Set initial condition
        self._sim.set_state([initial_state])

        # Set growth constants
        if self._dimless:
            self._sim.set_constant('central.a_0', param_0)
            self._sim.set_constant('central.a_1', param_1)
        else:
            self._sim.set_constant('central.lambda_0', param_0)
            self._sim.set_constant('central.lambda_1', param_1)

        # Define logged variable
        loggedVariable = 'central.volume_t'

        # Simulate
        output = self._sim.run(
            times[-1] + 1, log=[loggedVariable], log_times=times)
        result = output[loggedVariable]

        return np.array(result)

    def _create_dimless_tumour_growth_model(self):
        r"""
        Returns a non-dimensionalised tumour growth myokit model.

        .. math::
            \frac{\text{d}v}{\text{d}\tau} = \frac{a_1 v}
            {v + a_0},

        where the tumour volume :math:`v` and time :math:`\tau` are
        dimensionless and measured in characteristic scales :math:`V^c_T` and
        :math:`t^c`. The model parameters :math:`a_0` and :math:`a_1` are also
        dimensionless.
        """
        # Instantiate model
        model = Model()

        # Add central compartment
        central_comp = model.add_compartment('central')

        # Add tumour growth variables to central compartment
        volume_t = central_comp.add_variable('volume_t')
        a_0 = central_comp.add_variable('a_0')
        a_1 = central_comp.add_variable('a_1')

        # Bind time
        time = central_comp.add_variable('time')
        time.set_binding('time')

        # Set intial values (some default values) and units
        time.set_rhs(0)

        volume_t.set_rhs(0)
        a_0.set_rhs(1)  # Avoid ZeroDivisionError
        a_1.set_rhs(0)

        # Set units
        time.set_unit('dimensionless')

        volume_t.set_unit('dimensionless')
        a_0.set_unit('dimensionless')
        a_1.set_unit('dimensionless')

        # Set rhs of tumor volume
        # dot(volume_t) =
        #  (a_1 * volume_t) /
        #  (volume_t + a_0)
        volume_t.promote()
        volume_t.set_rhs(
            myokit.Divide(
                myokit.Multiply(
                    myokit.Name(a_1),
                    myokit.Name(volume_t)
                ),
                myokit.Plus(
                    myokit.Name(volume_t),
                    myokit.Name(a_0)
                )
            )
        )

        # Validate model
        model.validate()

        # Check units
        model.check_units()

        return model

    def _create_tumour_growth_model(self):
        r"""
        Returns a tumour growth myokit model.

        .. math::
            \frac{\text{d}V^s_T}{\text{d}t} = \frac{2\lambda _0\lambda _1 V^s_T}
            {2\lambda _0 V^s_T + \lambda _1},

        where the tumour volume :math:`V^s_T` is measured in :math:`\text{cm}^3`,
        the exponential growth rate :math:`\lambda _0` is mesured in
        :math:`\text{day}` and the linear growth rate :math:`\lambda _1` is
        measured in :math:`\text{cm}^3/\text{day}`.
        """
        # Instantiate model
        model = Model()

        # add central compartment
        central_comp = model.add_compartment('central')

        # add tumour growth variables to central compartment
        volume_t = central_comp.add_variable('volume_t')
        lambda_0 = central_comp.add_variable('lambda_0')
        lambda_1 = central_comp.add_variable('lambda_1')

        # bind time
        time = central_comp.add_variable('time')
        time.set_binding('time')

        # set preferred representation of units
        # time in days
        unit = myokit.parse_unit('day')
        myokit.Unit.register_preferred_representation('day', unit)
        # rates in 1 / day
        unit = myokit.parse_unit('1/day')
        myokit.Unit.register_preferred_representation('1/day', unit)
        # tumor volume
        unit = myokit.parse_unit('cm^3')
        myokit.Unit.register_preferred_representation('cm^3', unit)
        # linear growth
        unit = myokit.parse_unit('cm^3/day')
        myokit.Unit.register_preferred_representation('cm^3/day', unit)

        # set intial values (some default values) and units
        time.set_rhs(0)

        volume_t.set_rhs(0)
        lambda_0.set_rhs(0)
        lambda_1.set_rhs(1)  # avoid ZeroDivisionError

        # set units
        time.set_unit('day')  # time in days

        volume_t.set_unit('cm^3')  # milimeter cubed
        lambda_0.set_unit('1 / day')  # per day
        lambda_1.set_unit('cm^3 / day')  # milimiter cubed per day

        # set rhs of tumor volume
        # dot(volume_t) =
        #  (2 * lambda_0 * lambda_1 * volume_t) /
        #  (2 * lambda_0 * volume_t + lambda_1)
        volume_t.promote()
        volume_t.set_rhs(
            myokit.Divide(
                myokit.Multiply(
                    myokit.Number(2),
                    myokit.Multiply(
                        myokit.Name(lambda_0),
                        myokit.Multiply(
                            myokit.Name(lambda_1),
                            myokit.Name(volume_t)
                        )
                    )
                ),
                myokit.Plus(
                    myokit.Multiply(
                        myokit.Number(2),
                        myokit.Multiply(
                            myokit.Name(lambda_0),
                            myokit.Name(volume_t)
                        )
                    ),
                    myokit.Name(lambda_1)
                )
            )
        )

        # Validate model
        model.validate()

        # TODO: Check units
        # model.check_units()

        return model

    def __str__(self):
        return self._model.code()

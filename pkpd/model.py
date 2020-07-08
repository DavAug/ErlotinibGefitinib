#
# Model class to define structural model, route of administration and dosing
# strategy.
#
# Any caught error will raise a ValueError.
#

import collections as cs

import myokit
import numpy as np
import pints


class PintsModel(pints.ForwardModel):
    """
    Wrapper for myokit.Model to allow inference with pints.
    """
    def __init__(self, model, regimen):
        super(PintsModel, self).__init__()

        # Instantiate simulation and sensitivity simulation
        self._simulation = myokit.Simulation(model, regimen)
        self._psimulation = myokit.PSimulation(model, regimen)

        # Create iterator over state and parameter variables
        self._state_names = np.array(
            [state.name() for state in model.states()])
        self._param_names = np.array(
            [param.name() for param in model.parameters()])

        # Create mask for inferable initial conditions and parameters
        self._infer_init_states = np.ones(
                shape=model.count_states(), dtype=bool)
        self._infer_params = np.ones(
                shape=model.count_variables(const=True), dtype=bool)

        # Number inferable initial state values
        self._n_infer_init_states = model.count_states()

        # Create container for initial values
        self._initial_states = np.zeros(shape=model.count_states())

        # Create total number of inferable parameters
        self._n_parameters = model.count_states() + model.count_variables(
            const=True)

        # Define which variables are logged upon simulation
        self._logged_vars = self._state_names

    def parameters(self):
        """
        Returns names of all parameters and names of parameters that are
        inferred.
        """

        # All parameter names (states have initial values)
        all_parameters = np.append(self._state_names, self._param_names)

        # Parameter that are inferred.
        infer_parameters = np.append(
            self._state_names[self._infer_init_states],
            self._param_names[self._infer_params])

        return all_parameters, infer_parameters

    def fix_parameters(self, values=None):
        """
        Fixes parameter values and returns arrays of all parameters and those
        used for inference. As a result, those parameters are not inferred.
        If None is passed, all parameters can be inferred. If only specific
        parameters are supposed to be inferred, a list with None at the
        appropriate ids can be passed.
        """
        # TODO:
        # myokit.PSimulation does not support to initial values
        if values is None:
            # Reset infer mask, so all parameters are inferred
            self._infer_init_states = np.ones(
                shape=len(self._infer_init_states), dtype=bool)
            self._infer_params = np.ones(
                shape=len(self._infer_params), dtype=bool)

            # Reset initial values for states to all zeros
            self._initial_states = np.zeros(shape=len(self._initial_states))

            # Set number inferable initial state values to number of all states
            self._n_infer_init_states = len(self._initial_states)

            # Reset number of inferable parameters
            self._n_parameters = self._n_infer_init_states + len(
                self._infer_params)

            return self.parameters

        # Check that one value is passed for each initial condition and
        # parameter
        n_states = len(self._n_infer_init_states)
        n_params = len(self._infer_params)
        if len(values) != n_states + n_params:
            raise ValueError(
                'Not enough values passed. If you do not want to certain'
                'parameters for inference, please pass `None`.')

        # Get initial states (None are converted to np.nan)
        self._initial_states[:] = values[:n_states]

        # Update infer mask
        self._infer_init_states = np.isnan(self._initial_states)

        # Replace np.nan by zeros as default value
        for ids in range(np.sum(self._infer_init_states)):
            self._initial_states[ids] = 0

        # Update initial values of states
        self._simulation.set_default_state(self._initial_states)

        # Update number of inferable initial states
        self._n_infer_init_states = int(np.sum(self._infer_init_states))

        # Instanstiate parameter values
        param_values = np.empty(shape=n_params)

        # Get parameter values
        param_values[:] = values[n_states:]

        # Update infer mask
        self._infer_params = np.isnan(param_values)

        # Mask fixed parameter values
        param_values = param_values[~self._infer_params]

        # Fix parameter values
        for idp, p in enumerate(self._param_names[~self._infer_params]):
            # Get value
            value = param_values[idp]

            # Fix parameter
            self._simulation.set_constant(
                    var=p, value=value)

        # Update number of parameters
        self._n_parameters = int(
            np.sum(self._infer_init_states) + np.sum(self._infer_params))

        return self.parameters

    def simulate(self, parameters, times):
        """
        Returns simulated output of model at specified time points.

        Parameters are assumed to be ordered accroding to myokit.model, and
        initial values before parameter values.
        """
        # Reset simulation to default state
        self._simulation.reset()

        # Update initial state values
        self._initial_states[self._infer_init_states] = parameters[
            :self._n_infer_init_states]
        self._simulation.set_default_state(self._initial_states)

        # Remove initial states from parameter list
        parameters = parameters[self._n_infer_init_states:]

        # Update parameters
        for idp, p in enumerate(self._param_names[self._infer_params]):
            # Fix parameter
            self._simulation.set_constant(
                    var=p, value=parameters[idp])

        # Run simulation
        result = self._simulation.run(
            duration=times[-1]+1,
            log=self._logged_vars,
            log_times=times)

        return result

    def simulateS1(self, parameters, times):
        pass

    def n_parameters(self):
        """
        Returns the number of initial conditions and model parameters
        that are learned in the inference.
        """
        return self._n_parameters


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

        #TODO: Set units

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


def create_one_comp_pk_model():
    """
    Returns 1 compartmental PK model.
    """
    # Instantiate model
    model = Model()

    # add central compartment
    central_comp = model.add_compartment('central')

    # add variables and constants to central compartment
    amount = central_comp.add_variable('amount')
    volume = central_comp.add_variable('volume')
    k_e = central_comp.add_variable('k_e')
    conc = central_comp.add_variable('conc')

    # bind time
    time = central_comp.add_variable('time')
    time.set_binding('time')

    # set intial values (some default values)
    time.set_rhs(0)

    amount.set_rhs(0)
    volume.set_rhs(1)
    k_e.set_rhs(0)
    conc.set_rhs(0)

    # set units
    time.set_unit('day')  # time in days

    amount.set_unit('mg')  # miligram
    volume.set_unit('L')  # liter
    k_e.set_unit('1 / day')  # 1 / day
    conc.set_unit('mg / L')  # miligram / liter

    # set preferred representation of units
    # time days
    unit = myokit.parse_unit('day')
    myokit.Unit.register_preferred_representation('day', unit)
    # rates in 1 / day
    unit = myokit.parse_unit('1/day')
    myokit.Unit.register_preferred_representation('1/day', unit)
    # amount in mg
    unit = myokit.parse_unit('mg')
    myokit.Unit.register_preferred_representation('mg', unit)
    # dose rate in mg / day
    unit = myokit.parse_unit('mg/day')
    myokit.Unit.register_preferred_representation('mg/day', unit)
    # concentration mg / L
    unit = myokit.parse_unit('mg/L')
    myokit.Unit.register_preferred_representation('mg/L', unit)

    # set rhs of state variables
    # (dot(amount) = - k_e * amount)
    amount.promote()
    amount.set_rhs(
        myokit.Multiply(
            myokit.PrefixMinus(myokit.Name(k_e)),
            myokit.Name(amount)
        )
    )

    # set algebraic relation between drug and concentration
    conc.set_rhs(
        myokit.Divide(
            myokit.Name(amount),
            myokit.Name(volume)
        )
    )

    return model


def create_pktgi_model():
    """
    Returns 1 compartmental PK model.
    """
    # Instantiate model
    model = Model()

    # add central compartment
    central_comp = model.add_compartment('central')

    # add PK variables and constants to central compartment
    amount = central_comp.add_variable('amount')
    volume = central_comp.add_variable('volume')
    k_e = central_comp.add_variable('k_e')
    conc = central_comp.add_variable('conc')

    # add PD variables to central compartment
    volume_T = central_comp.add_variable('volume_tumor')
    lambda_0 = central_comp.add_variable('lambda_0')
    lambda_1 = central_comp.add_variable('lambda_1')
    kappa = central_comp.add_variable('kappa')

    # bind time
    time = central_comp.add_variable('time')
    time.set_binding('time')

    # set intial values (some default values)
    time.set_rhs(0)

    amount.set_rhs(0)
    volume.set_rhs(1)  # avoid ZeroDivisionError
    k_e.set_rhs(0)
    conc.set_rhs(0)

    volume_T.set_rhs(0)
    lambda_0.set_rhs(0)
    lambda_1.set_rhs(1)  # avoid ZeroDivisionError
    kappa.set_rhs(0)

    # set units
    time.set_unit('day')  # time in days

    amount.set_unit('mg')  # miligram
    volume.set_unit('L')  # liter
    k_e.set_unit('1 / day')  # 1 / day
    conc.set_unit('mg / L')  # miligram / liter

    volume_T.set_unit('mm^3')  # milimeter cubed
    lambda_0.set_unit('1 / day')  # per day
    lambda_1.set_unit('mm^3 / day')  # milimiter cubed per day
    kappa.set_unit('L / mg / day')  # in reference L / ug / day,

    # set preferred representation of units
    # time days
    unit = myokit.parse_unit('day')
    myokit.Unit.register_preferred_representation('day', unit)
    # rates in 1 / day
    unit = myokit.parse_unit('1/day')
    myokit.Unit.register_preferred_representation('1/day', unit)
    # amount in mg
    unit = myokit.parse_unit('mg')
    myokit.Unit.register_preferred_representation('mg', unit)
    # dose rate in mg / day
    unit = myokit.parse_unit('mg/day')
    myokit.Unit.register_preferred_representation('mg/day', unit)
    # concentration mg / L
    unit = myokit.parse_unit('mg/L')
    myokit.Unit.register_preferred_representation('mg/L', unit)

    # tumor volume
    unit = myokit.parse_unit('mm^3')
    myokit.Unit.register_preferred_representation('mm^3', unit)
    # linear growth
    unit = myokit.parse_unit('mm^3/day')
    myokit.Unit.register_preferred_representation('mm^3/day', unit)
    # potency
    unit = myokit.parse_unit('L/mg/day')
    myokit.Unit.register_preferred_representation('L/mg/day', unit)

    # set rhs of drug amount
    # (dot(amount) = - k_e * amount)
    amount.promote()
    amount.set_rhs(
        myokit.Multiply(
            myokit.PrefixMinus(myokit.Name(k_e)),
            myokit.Name(amount)
        )
    )

    # set rhs of tumor volume
    # dot(volume_T) =
    #  (2 * lambda_0 * lambda_1 * volume_T) /
    #  (2 * lambda_0 * volume_T + lambda_1) -
    #  kappa * conc * volume_T
    volume_T.promote()
    volume_T.set_rhs(
        myokit.Minus(
            myokit.Divide(
                myokit.Multiply(
                    myokit.Number(2),
                    myokit.Multiply(
                        myokit.Name(lambda_0),
                        myokit.Multiply(
                            myokit.Name(lambda_1),
                            myokit.Name(volume_T)
                        )
                    )
                ),
                myokit.Plus(
                    myokit.Multiply(
                        myokit.Number(2),
                        myokit.Multiply(
                            myokit.Name(lambda_0),
                            myokit.Name(volume_T)
                        )
                    ),
                    myokit.Name(lambda_1)
                )
            ),
            myokit.Multiply(
                myokit.Name(kappa),
                myokit.Multiply(
                    myokit.Name(conc),
                    myokit.Name(volume_T)
                )
            )
        )
    )

    # set algebraic relation between drug and concentration
    # conc = amount / volume
    conc.set_rhs(
        myokit.Divide(
            myokit.Name(amount),
            myokit.Name(volume)
        )
    )

    return model


def create_tumour_growth_model():
    r"""
    Returns a tumour growth myokit model.

    .. math::
        \frac{\text{d}V^s_T}{\text{d}t} = \frac{2\lambda _0\lambda _1 V^s_T}
        {2\lambda _0 V^s_T + \lambda _1}.
    """
    # Instantiate model
    model = Model()

    # add central compartment
    central_comp = model.add_compartment('central')

    # add tumour growth variables to central compartment
    volume_T = central_comp.add_variable('volume_t')
    lambda_0 = central_comp.add_variable('lambda_0')
    lambda_1 = central_comp.add_variable('lambda_1')

    # bind time
    time = central_comp.add_variable('time')
    time.set_binding('time')

    # set intial values (some default values)
    time.set_rhs(0)

    volume_T.set_rhs(0)
    lambda_0.set_rhs(0)
    lambda_1.set_rhs(1)  # avoid ZeroDivisionError

    # set units
    time.set_unit('day')  # time in days

    volume_T.set_unit('mm^3')  # milimeter cubed
    lambda_0.set_unit('1 / day')  # per day
    lambda_1.set_unit('mm^3 / day')  # milimiter cubed per day

    # set preferred representation of units
    # time days
    unit = myokit.parse_unit('day')
    myokit.Unit.register_preferred_representation('day', unit)
    # rates in 1 / day
    unit = myokit.parse_unit('1/day')
    myokit.Unit.register_preferred_representation('1/day', unit)
    # tumor volume
    unit = myokit.parse_unit('mm^3')
    myokit.Unit.register_preferred_representation('mm^3', unit)
    # linear growth
    unit = myokit.parse_unit('mm^3/day')
    myokit.Unit.register_preferred_representation('mm^3/day', unit)

    # set rhs of tumor volume
    # dot(volume_T) =
    #  (2 * lambda_0 * lambda_1 * volume_T) /
    #  (2 * lambda_0 * volume_T + lambda_1)
    volume_T.promote()
    volume_T.set_rhs(
        myokit.Divide(
            myokit.Multiply(
                myokit.Number(2),
                myokit.Multiply(
                    myokit.Name(lambda_0),
                    myokit.Multiply(
                        myokit.Name(lambda_1),
                        myokit.Name(volume_T)
                    )
                )
            ),
            myokit.Plus(
                myokit.Multiply(
                    myokit.Number(2),
                    myokit.Multiply(
                        myokit.Name(lambda_0),
                        myokit.Name(volume_T)
                    )
                ),
                myokit.Name(lambda_1)
            )
        )
    )

    return model

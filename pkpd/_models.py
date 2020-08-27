#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import myokit
import numpy as np
import pints


class TumourGrowthModel(pints.ForwardModel):
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
        model = myokit.Model()

        # Add central compartment
        central_comp = model.add_component('central')

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
            \frac{\text{d}V^s_T}{\text{d}t} =
            \frac{2\lambda _0\lambda _1 V^s_T}
            {2\lambda _0 V^s_T + \lambda _1},

        where the tumour volume :math:`V^s_T` is measured in
        :math:`\text{cm}^3`,
        the exponential growth rate :math:`\lambda _0` is mesured in
        :math:`\text{day}` and the linear growth rate :math:`\lambda _1` is
        measured in :math:`\text{cm}^3/\text{day}`.
        """
        # Instantiate model
        model = myokit.Model()

        # add central compartment
        central_comp = model.add_component('central')

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

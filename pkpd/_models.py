#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import myokit
import myokit.formats.sbml as sbml
import numpy as np
import pints


class PharmacodynamicModel(pints.ForwardModel):
    """
    Creates a `pints.ForwardModel` from a SBML model.

    Arguments:
        path -- Absolute path to SBML model file.
        is_log_transformed -- Flag whether model parameters are
                              log-transformed.
    """

    def __init__(self, path, is_log_transformed=True):
        super(PharmacodynamicModel, self).__init__()

        model = sbml.SBMLImporter().model(path)
        self._is_log_transformed = is_log_transformed

        # Get the number of states and parameters
        self._n_states = model.count_states()
        n_const = model.count_variables(const=True)
        self._n_parameters = self._n_states + n_const

        # Get iterator over constant variables and states
        self._state_names = [var.qname() for var in model.states()]
        self._const_names = [
            var.qname() for var in model.variables(const=True)]

        # Set default outputs
        self._output_names = self._state_names
        self._n_outputs = self._n_states

        # Create simulator
        self._sim = myokit.Simulation(model)

    def _set_const(self, parameters):
        """
        Sets values of constant model parameters.
        """
        for id_var, var in enumerate(self._const_names):
            self._sim.set_constant(var, float(parameters[id_var]))

    def n_outputs(self):
        """
        Returns the number of output dimensions.

        By default this is the number of states.
        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the number of parameters in the model.

        Parameters of the model are initial state values and structural
        parameter values.
        """
        return self._n_parameters

    def outputs(self):
        """
        Returns the output names of the model.
        """
        return self._output_names

    def parameters(self):
        """
        Returns the parameter names of the model.
        """
        return self._state_names + self._const_names

    def set_outputs(self, outputs):
        """
        Sets outputs of the model.

        Outputs has to be a list of quantifiable variable names of the
        myokit.Model, e.g. `compartment.variable`.
        """
        # Check that outputs are valid
        for output in outputs:
            try:
                self._sim._model.get(output)
            except KeyError:
                raise KeyError(
                    'The variable <' + str(output) + '> does not exist in the '
                    'model.')

        self._output_names = outputs
        self._n_outputs = len(outputs)

    def simulate(self, parameters, times):
        """
        Returns the numerical solution of the model outputs for specified
        parameters and times.
        """
        # Reset simulation
        self._sim.reset()

        # Transform parameters back to linear scale if log-transfomed
        if self._is_log_transformed:
            parameters = np.exp(parameters)

        # Set initial conditions
        self._sim.set_state(parameters[:self._n_states])

        # Set constant model parameters
        self._set_const(parameters[self._n_states:])

        # Simulate
        output = self._sim.run(
            times[-1] + 1, log=self._output_names, log_times=times)
        result = [output[name] for name in self._output_names]

        # Transform shape of output to be compatible with
        # pints.SingleOutputProblem/pints.MultiOutputProblem
        if self._n_outputs == 1:
            result = np.array(result).flatten()
        else:
            result = np.array(result).transpose()

        return result

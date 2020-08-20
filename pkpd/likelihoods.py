#
# This file belongs to the ErlotinibGefitinib repository.
#

import numpy as np
import pints


class SharedNoiseLogLikelihood(pints.LogPDF):
    """
    Calculates a sum of :class:`pints.LogPDF` objects, defined on potentially
    different parameter spaces, but with shared noise parameters.

    This likelihood is useful for example when measurements are taken from two
    individuals :math:`X^{\text{obs}_1}` and :math:`X^{\text{obs}_2}`. The
    measured process may be expected to be the same for both individuals, but
    the details (i.e. the parameters) maye still differ due to individual
    difference. However, since the measurement process is the same, the noise
    model is shared between the indiviudals.

    Extends :class:`pints.LogPDF`.

    Parameters
    ----------
    problems
        A sequence of :class:`pints.SingleOutputProblem` or
        :class:`pints.MultiOutputProblem` objects. The number of outputs of the
        problems needs to be same.
    log_likelihood
        The likelihood class which models the noise across the problems.
    """
    def __init__(self, problems, log_likelihood, *args):
        super(SharedNoiseLogLikelihood, self).__init__()

        # Check input arguments
        if len(problems) < 2:
            raise ValueError(
                'SharedNoiseLogLikelihood requires at least two problems.')
        if not isinstance(log_likelihood, type):
            raise ValueError(
                'The log-likelihood needs to be a :class:`pints.LogPDF` class.'
            )

        # Check number of outputs of problems
        self._no = problems[0].n_outputs()
        for idx, problem in enumerate(problems):
            if problem.n_outputs() != self._no:
                raise ValueError(
                    'The input problems need to have the same number of '
                    'outputs. The problem with index <' + idx + '> has <'
                    + problem.n_outputs() + '> outputs, while the previous'
                    ' problems had <' + self._no + '> outputs.')

        # Check log_likelihood can be instantiated
        try:
            log_likelihood(problems[0], *args)
        except ValueError:
            raise ValueError(
                'Something went wrong!')

        # Get number of parameters for each problem
        n_problem_parameters = []
        for problem in problems:
            n_problem_parameters.append(problem.n_parameters())

        # Create log-likelihoods for each problem
        self._log_likelihoods = []
        for problem in problems:
            self._log_likelihoods.append(log_likelihood(problem, *args))

        # Check number of noise parameters
        self._n_noise_params = self._log_likelihoods[0].n_parameters() - \
            problems[0].n_parameters()
        if self._n_noise_params == 0:
            raise ValueError(
                'This method needs at least one noise parameter that can be'
                ' shared between models.')

        # Get total number of parameters
        self._cum_n_problem_params = np.cumsum(n_problem_parameters)
        self._n_parameters = self._n_noise_params + \
            self._cum_n_problem_params[-1]

    def __call__(self, parameters):

        start = 0
        total = 0
        for idx, log_likelihood in enumerate(self._log_likelihoods):
            # Get relevant parameters
            params = np.hstack((
                parameters[start:self._cum_n_problem_params[idx]],
                parameters[self._cum_n_problem_params[-1]:]))

            # Compute contribution to log-likelihood
            total += log_likelihood(params)

            # Move start of slicing
            start = self._cum_n_problem_params[idx]

        return total

    def evaluateS1(self, x):
        """
        See :meth:`LogPDF.evaluateS1()`.

        *This method only works if all the underlying :class:`LogPDF` objects
        implement the optional method :meth:`LogPDF.evaluateS1()`!*
        """
        raise NotImplementedError
        # total = 0
        # dtotal = np.zeros(self._n_parameters)
        # for e in self._log_likelihoods:
        #     a, b = e.evaluateS1(x)
        #     total += a
        #     dtotal += np.asarray(b)
        # return total, dtotal

    def n_parameters(self):
        """ See :meth:`LogPDF.n_parameters()`. """
        return self._n_parameters


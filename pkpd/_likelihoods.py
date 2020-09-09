#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import numpy as np
import pints


class ConstantAndMultiplicativeGaussianLogLikelihood(
        pints.ProblemLogLikelihood):
    r"""
    Calculates the log-likelihood assuming a mixed error model of a
    Gaussian base-level noise and a Gaussian heteroscedastic noise.

    For a time series model :math:`f(t| \theta)` with parameters :math:`\theta`
    , the ConstantAndMultiplicativeGaussianLogLikelihood assumes that the
    model predictions :math:`X` are Gaussian distributed according to

    .. math::
        X(t| \theta , \sigma _{\text{base}}, \sigma _{\text{rel}}) =
        f(t| \theta) + (\sigma _{\text{base}} + \sigma _{\text{rel}}
        f(t| \theta)^\eta ) \, \epsilon ,

    where :math:`\epsilon` is a i.i.d. standard Gaussian random variable

    .. math::
        \epsilon \sim \mathcal{N}(0, 1).

    For each output in the problem, this likelihood introduces three new scalar
    parameters: a base-level scale :math:`\sigma _{\text{base}}`; an
    exponential power :math:`\eta`; and a scale relative to the model output
    :math:`\sigma _{\text{rel}}`.

    The resulting log-likelihood of a constant and multiplicative Gaussian
    error model is

    .. math::
        \log L(\theta, \sigma _{\text{base}}, \eta ,
        \sigma _{\text{rel}} | X^{\text{obs}})
        = -\frac{n_t}{2} \log 2 \pi
        -\sum_{i=1}^{n_t}\log \sigma _{\text{tot}, i}
        - \sum_{i=1}^{n_t}
        \frac{(X^{\text{obs}}_i - f(t_i| \theta))^2}
        {2\sigma ^2_{\text{tot}, i}},

    where :math:`n_t` is the number of measured time points in the time series,
    :math:`X^{\text{obs}}_i` is the observation at time point :math:`t_i`, and
    :math:`\sigma _{\text{tot}, i}=\sigma _{\text{base}} +\sigma _{\text{rel}}
    f(t_i| \theta)^\eta` is the total standard deviation of the error at time
    :math:`t_i`.

    For a system with :math:`n_o` outputs, this becomes

    .. math::
        \log L(\theta, \sigma _{\text{base}}, \eta ,
        \sigma _{\text{rel}} | X^{\text{obs}})
        = -\frac{n_tn_o}{2} \log 2 \pi
        -\sum_{j=1}^{n_0}\sum_{i=1}^{n_t}\log \sigma _{\text{tot}, ij}
        - \sum_{j=1}^{n_0}\sum_{i=1}^{n_t}
        \frac{(X^{\text{obs}}_{ij} - f_j(t_i| \theta))^2}
        {2\sigma ^2_{\text{tot}, ij}},

    where :math:`n_o` is the number of outputs of the model,
    :math:`X^{\text{obs}}_{ij}` is the observation at time point :math:`t_i`
    of output :math:`j`, and
    :math:`\sigma _{\text{tot}, ij}=\sigma _{\text{base}, j} +
    \sigma _{\text{rel}, j}f_j(t_i| \theta)^{\eta _j}` is the total standard
    deviation of the error at time :math:`t_i` of output :math:`j`.

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    ``problem``
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`. For a
        single-output problem three parameters are added
        (:math:`\sigma _{\text{base}}`, :math:`\eta`,
        :math:`\sigma _{\text{rel}}`),
        for a multi-output problem :math:`3n_o` parameters are added
        (:math:`\sigma _{\text{base},1},\ldots , \sigma _{\text{base},n_o},
        \eta _1,\ldots , \eta _{n_o}, \sigma _{\text{rel},1}, \ldots ,
        \sigma _{\text{rel},n_o})`.
    """

    def __init__(self, problem):
        super(ConstantAndMultiplicativeGaussianLogLikelihood, self).__init__(
            problem)

        # Get number of times and number of noise parameters
        self._nt = len(self._times)
        self._no = problem.n_outputs()
        self._np = 3 * self._no

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + self._np

        # Pre-calculate the constant part of the likelihood
        self._logn = -0.5 * self._nt * self._no * np.log(2 * np.pi)

    def __call__(self, parameters):
        # Get parameters from input
        noise_parameters = np.asarray(parameters[-self._np:])
        sigma_base = noise_parameters[:self._no]
        eta = noise_parameters[self._no:2 * self._no]
        sigma_rel = noise_parameters[2 * self._no:]

        # Evaluate noise-free model (n_times, n_outputs)
        function_values = self._problem.evaluate(parameters[:-self._np])

        # Compute error (n_times, n_outputs)
        error = self._values - function_values

        # Compute total standard deviation
        sigma_tot = sigma_base + sigma_rel * function_values**eta

        # Compute log-likelihood
        # (inner sums over time points, outer sum over parameters)
        log_likelihood = self._logn - np.sum(
            np.sum(np.log(sigma_tot), axis=0)
            + 0.5 * np.sum(error**2 / sigma_tot**2, axis=0))

        return log_likelihood

    def evaluateS1(self, parameters):
        r"""
        See :meth:`LogPDF.evaluateS1()`.

        The partial derivatives of the log-likelihood w.r.t. the model
        parameters are

        .. math::
            \frac{\partial \log L}{\partial \theta _k}
            =& -\sum_{i,j}\sigma _{\text{rel},j}\eta _j\frac{
            f_j(t_i| \theta)^{\eta _j-1}}
            {\sigma _{\text{tot}, ij}}
            \frac{\partial f_j(t_i| \theta)}{\partial \theta _k}
            + \sum_{i,j}
            \frac{X^{\text{obs}}_{ij} - f_j(t_i| \theta)}
            {\sigma ^2_{\text{tot}, ij}}
            \frac{\partial f_j(t_i| \theta)}{\partial \theta _k} \\
            &+\sum_{i,j}\sigma _{\text{rel},j}\eta _j
            \frac{(X^{\text{obs}}_{ij} - f_j(t_i| \theta))^2}
            {\sigma ^3_{\text{tot}, ij}}f_j(t_i| \theta)^{\eta _j-1}
            \frac{\partial f_j(t_i| \theta)}{\partial \theta _k} \\
            \frac{\partial \log L}{\partial \sigma _{\text{base}, j}}
            =& -\sum ^{n_t}_{i=1}\frac{1}{\sigma _{\text{tot}, ij}}
            +\sum ^{n_t}_{i=1}
            \frac{(X^{\text{obs}}_{ij} - f_j(t_i| \theta))^2}
            {\sigma ^3_{\text{tot}, ij}} \\
            \frac{\partial \log L}{\partial \eta _j}
            =& -\sigma _{\text{rel},j}\eta _j\sum ^{n_t}_{i=1}
            \frac{f_j(t_i| \theta)^{\eta _j}\log f_j(t_i| \theta)}
            {\sigma _{\text{tot}, ij}}
            + \sigma _{\text{rel},j}\eta _j \sum ^{n_t}_{i=1}
            \frac{(X^{\text{obs}}_{ij} - f_j(t_i| \theta))^2}
            {\sigma ^3_{\text{tot}, ij}}f_j(t_i| \theta)^{\eta _j}
            \log f_j(t_i| \theta) \\
            \frac{\partial \log L}{\partial \sigma _{\text{rel},j}}
            =& -\sum ^{n_t}_{i=1}
            \frac{f_j(t_i| \theta)^{\eta _j}}{\sigma _{\text{tot}, ij}}
            + \sum ^{n_t}_{i=1}
            \frac{(X^{\text{obs}}_{ij} - f_j(t_i| \theta))^2}
            {\sigma ^3_{\text{tot}, ij}}f_j(t_i| \theta)^{\eta _j},

        where :math:`i` sums over the measurement time points and :math:`j`
        over the outputs of the model.
        """
        # Get parameters from input
        # Shape sigma_base, eta, sigma_rel = (n_outputs,)
        noise_parameters = np.asarray(parameters[-self._np:])
        sigma_base = noise_parameters[:self._no]
        eta = noise_parameters[self._no:2 * self._no]
        sigma_rel = noise_parameters[-self._no:]

        # Evaluate noise-free model, and get residuals
        # y shape = (n_times,) or (n_times, n_outputs)
        # dy shape = (n_times, n_model_parameters) or
        # (n_times, n_outputs, n_model_parameters)
        y, dy = self._problem.evaluateS1(parameters[:-self._np])

        # Reshape y and dy, in case we're working with a single-output problem
        # Shape y = (n_times, n_outputs)
        # Shape dy = (n_model_parameters, n_times, n_outputs)
        y = y.reshape(self._nt, self._no)
        dy = np.transpose(
            dy.reshape(self._nt, self._no, self._n_parameters - self._np),
            axes=(2, 0, 1))

        # Compute error
        # Note: Must be (data - simulation), sign now matters!
        # Shape: (n_times, output)
        error = self._values.reshape(self._nt, self._no) - y

        # Compute total standard deviation
        sigma_tot = sigma_base + sigma_rel * y**eta

        # Compute likelihood
        L = self.__call__(parameters)

        # Compute derivative w.r.t. model parameters
        dtheta = -np.sum(sigma_rel * eta * np.sum(
            y**(eta - 1) * dy / sigma_tot, axis=1), axis=1) + \
            np.sum(error * dy / sigma_tot**2, axis=(1, 2)) + np.sum(
                sigma_rel * eta * np.sum(
                    error**2 * y**(eta - 1) * dy / sigma_tot**3, axis=1),
                axis=1)

        # Compute derivative w.r.t. sigma base
        dsigma_base = - np.sum(1 / sigma_tot, axis=0) + np.sum(
            error**2 / sigma_tot**3, axis=0)

        # Compute derivative w.r.t. eta
        deta = -sigma_rel * (
            np.sum(y**eta * np.log(y) / sigma_tot, axis=0) -
            np.sum(
                error**2 / sigma_tot**3 * y**eta * np.log(y),
                axis=0))

        # Compute derivative w.r.t. sigma rel
        dsigma_rel = -np.sum(y**eta / sigma_tot, axis=0) + np.sum(
            error**2 / sigma_tot**3 * y**eta, axis=0)

        # Collect partial derivatives
        dL = np.hstack((dtheta, dsigma_base, deta, dsigma_rel))

        # Return
        return L, dL


class FixedEtaLogLikelihoodWrapper(pints.LogPDF):
    """
    Implements a log-likelihood wrapper for a
    `pints.MultiplicativeGaussianLogLikelihood` or a
    `pkpd.ConstantAndMultiplicativeGaussianLogLikelihood` that allows fixing
    the parameter `eta`.
    """

    def __init__(self, log_likelihood, eta):
        if not isinstance(
            log_likelihood, (
                pints.MultiplicativeGaussianLogLikelihood,
                ConstantAndMultiplicativeGaussianLogLikelihood)):
            raise ValueError(
                'This likelihood wrapper is only defined for a '
                '`pints.MultiplicativeLogLikelihood` or '
                '`pkpd.ConstantAndMultiplicativeGaussianLogLikelihood.')
        if log_likelihood._problem.n_outputs() != 1:
            raise ValueError(
                'This likelihood wrapper is only defined for a '
                '`pints.SingleOutputProblem`.')

        self._log_pdf = log_likelihood
        self._eta = eta

    def __call__(self, parameters):
        # Create parameter container
        params = np.empty(shape=len(parameters)+1)

        # Fill container with parameters
        # (Eta is at second last position for SingleOutputProblems)
        params[:-2] = np.asarray(parameters[:-1])
        params[-2] = self._eta
        params[-1] = parameters[-1]

        return self._log_pdf(params)

    def evaluateS1(self, parameters):
        raise NotImplementedError(
            'Method has not been implemented.')

    def n_parameters(self):
        return self._log_pdf.n_parameters() - 1


class PooledLogPDF(pints.LogPDF):
    r"""
    Calculates a sum of :class:`LogPDF` objects, while pooling selected
    parameters across log-pdfs.

    This :class:`LogPDF` requires that all :class:`LogPDF` objects have the
    same number of parameters.

    This is useful for e.g. modelling the time-series of multiple individuals
    (each individual defines a separate :class:`LogPDF`), and some parameters
    are expected to be the same across individuals. This may be a realistic
    expectation for the noise parameter, if the experimental set up didn't vary
    for the indiviudals.

    For two :class:`LogPDF` objects :math:`L _1`  and :math:`L _2` with
    parameters :math:`(\psi _1, \theta _1`) and :math:`(\psi _2, \theta _2`)
    respectively, a pooling of the :math:`\theta _i` results in a pooled
    log-pdf of the form

    .. math::
        \log L(\psi _1, \psi _2, \theta | D_1, D_2) =
            \log L(\psi _1, \theta | D_1) + \log L(\psi _2, \theta | D_2),

    where :math:`\theta := \theta _1 = \theta _2`, and $D_i$ is the measured
    time-series of individual :math:`i`.

    In general, the search of a :class:`PooledLogPDF` has dimensionality
    :math:`nd_{\text{up}} + d_{\text{p}}`, where :math:`n` is the number of
    individuals, :math:`d_{\text{up}}` is the dimension of the unpooled
    :math:`psi` parameters, and :math:`d_{\text{up}}` is the dimension of the
    pooled :math:`\theta` parameter.

    The order of parameters for evaluation is largely kept, and the parameters
    of individual log-pdfs are concatenated. However, pooled parameters are
    always prepended. Consider for example two log-pdf with parameters
    :math:`(\psi _{1,1}, \psi _{1, 2}, \psi _{1,3}, \psi _{1,4})` and
    :math:`(\psi _{2,1}, \psi _{2, 2}, \psi _{2,3}, \psi _{2,4})`. Pooling the
    first and the last parameter, i.e.
    :math:`\theta _1 := \psi _{1,1} = \psi _{2, 1}` and
    :math:`\theta _4 := \psi _{1,4} = \psi _{2, 4}` results in and expect
    order of parameters
    :math:`(\psi _{1, 2}, \psi _{1,3}, \psi _{2, 2}, \psi _{2,3}, \theta _1,
    \theta _4)`.

    Extends :class:`LogPDF`.

    Parameters
    ----------
    log_pdfs
        A sequence of :class:`LogPDF` objects.
    pooled
        An array-like object of dtype bool, indicating which parameters across
        the likelihoods are pooled (`True`) or remain unpooled (`False`).

    Example
    -------
    ::
        pooled_log_likelihood = pints.PooledLogPDFs(
            log_pdfs=[
                pints.GaussianLogLikelihood(problem1),
                pints.GaussianLogLikelihood(problem2)],
            pooled=[False, True])
    """
    def __init__(self, log_pdfs, pooled):
        super(PooledLogPDF, self).__init__()

        # Check input arguments
        if len(log_pdfs) < 2:
            raise ValueError(
                'PooledLogPDF requires at least two log-pdfs.')
        for index, pdf in enumerate(log_pdfs):
            if not isinstance(pdf, pints.LogPDF):
                raise ValueError(
                    'All log-pdfs passed to PooledLogPDFs must be instances of '
                    'pints.LogPDF (failed on argument '
                    + str(index) + ').')

        # Check parameter dimension across log-pdfs
        self._log_pdfs = log_pdfs
        n_parameters = self._log_pdfs[0].n_parameters()
        for pdf in self._log_pdfs:
            if pdf.n_parameters() != n_parameters:
                raise ValueError(
                    'All log-pdfs passed to PooledLogPDFs must have '
                    'same dimension.')

        # Check that pooled matches number of parameters
        self._pooled = np.asarray(pooled)
        if len(self._pooled) != n_parameters:
            raise ValueError(
                'The array-like input `pooled` needs to have the same length '
                'as the number of parameters of the individual log-pdfs.')

        # Check that pooled contains only booleans
        for p in self._pooled:
            if not isinstance(p, np.bool_):
                raise ValueError(
                    'The array-like input `pooled` passed to PooledLogPDFs '
                    'has to contain booleans exclusively.')

        # Get dimension of search space
        self._n_pooled = np.sum(self._pooled)
        n_individuals = len(self._log_pdfs)
        self._n_unpooled = np.sum(~self._pooled)
        self._n_parameters = \
            self._n_pooled + n_individuals * self._n_unpooled

    def __call__(self, parameters):
        # Get parameters of pooled log-pdf
        parameters = np.asarray(parameters)

        # Create container for parameters of individuals log-pdf and fill with
        # pooled parameters
        params_ind = np.empty(shape=self._n_unpooled + self._n_pooled)
        if self._n_pooled > 0:
            params_ind[self._pooled] = parameters[-self._n_pooled:]

        # Compute pdf score
        total = 0
        for idx, pdf in enumerate(self._log_pdfs):
            # Get unpooled parameters for individual
            params_ind[~self._pooled] = parameters[
                idx * self._n_unpooled: (idx + 1) * self._n_unpooled]

            # Compute pdf score contribution
            total += pdf(params_ind)
        return total

    def evaluateS1(self, parameters):
        """
        See :meth:`LogPDF.evaluateS1()`.

        *This method only works if all the underlying :class:`LogPDF` objects
        implement the optional method :meth:`LogPDF.evaluateS1()`!*
        """
        # Get parameters of pooled log-pdf
        parameters = np.asarray(parameters)

        # Create container for parameters of individuals log-pdf and fill with
        # pooled parameters
        params_ind = np.empty(shape=self._n_unpooled + self._n_pooled)
        if self._n_pooled > 0:
            params_ind[self._pooled] = parameters[-self._n_pooled:]

        # Compute pdf score and partials
        total = 0
        dtotal = np.zeros(shape=self._n_parameters)
        for idx, pdf in enumerate(self._log_pdfs):
            # Get unpooled parameters for individual
            params_ind[~self._pooled] = parameters[
                idx * self._n_unpooled: (idx + 1) * self._n_unpooled]

            # Compute pdf score and partials for individual
            score, partials = pdf.evaluateS1(params_ind)

            # Add contributions to score and partials. Note that partials
            # w.r.t. unpooled parameters receive only one contribution
            total += score
            dtotal[idx * self._n_unpooled: (idx + 1) * self._n_unpooled] = \
                partials[~self._pooled]
            if self._n_pooled > 0:
                dtotal[-self._n_pooled:] += partials[self._pooled]

        return total, dtotal

    def n_parameters(self):
        """ See :meth:`LogPDF.n_parameters()`. """
        return self._n_parameters


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

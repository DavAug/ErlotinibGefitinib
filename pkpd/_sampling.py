#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import numpy as np
import pints


def sample(
        log_pdf, sampler, initial_params, n_chains, warm_up=None,
        max_iterations=None):
    """
    Returns parameters that optimise the objective function.
    """
    if not isinstance(log_pdf, pints.LogPDF):
        raise ValueError(
            'Objective function has to be an instance of `pints.LogPDF`.')
    if not issubclass(sampler, pints.MCMCSampler):
        raise ValueError(
            'Sampler has to be an instance of `pints.MCMCSampler`.')

    n_parameters = log_pdf.n_parameters()
    initial_params = np.asarray(initial_params)
    if initial_params.shape != (n_chains, n_parameters):
        raise ValueError(
            'Initial parameters has the wrong shape! Expected shape = '
            '(%d, %d).' % (n_chains, n_parameters))

    # Run sampling multiple times
    sampler = pints.MCMCController(
        log_pdf=log_pdf, chains=n_chains, x0=initial_params, method=sampler)

    # Configure sampling routine
    sampler.set_log_to_screen(False)
    sampler.set_parallel(True)
    if max_iterations:
        sampler.set_max_iterations(max_iterations)

    # Find optimal parameters
    traces = np.asarray(sampler.run())
    if warm_up:
        traces = traces[:, warm_up:, :]

    return traces

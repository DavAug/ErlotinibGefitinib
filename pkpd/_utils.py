#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import numpy as np


def get_median_parameters(parameters, pooled):
    """
    Returns median parameters across runs.

    For pooled parameters the median across runs and individuals is returned.
    The median parmeters are reordered such that pooled parameters always come
    last.

    Arguments:
        parameters -- Parameters of individuals across individuals and runs.
                      Shape: (n_individuals, n_runs, n_parameters)

    Returns:
        Median parameters across runs (and if pooled, also across individuals)
        of shape (n_unpooled, n_pooled).
    """
    parameters = np.asarray(parameters)
    if parameters.ndim != 3:
        raise ValueError(
            'Parameters has to be of dimension 3. Expected shape: '
            '(n_individuals, n_runs, n_parameters)')
    pooled = np.asarray(pooled)
    if len(pooled) != parameters.shape[2]:
        raise ValueError(
            'The array-like object `pooled` is expected to have the shape '
            '(n_parameters,)')
    if pooled.dtype != np.bool_:
        raise TypeError(
            'The array-like object has to contain exclusively Booleans.')

    n_pooled = np.sum(pooled)
    n_ids = parameters.shape[0]
    n_unpooled = np.sum(~pooled)
    medians = np.empty(shape=n_ids * n_unpooled + n_pooled)

    # Get unpooled medians
    medians[:n_ids * n_unpooled] = np.median(
        parameters[:, :, ~pooled], axis=1).flatten()

    # Get pooled medians
    if n_pooled > 0:
        medians[-n_pooled:] = np.median(parameters[:, :, pooled], axis=(0, 1))

    return medians


'''
def compute_cumulative_dose_amount(
        times, doses, end_exp, duration=1E-03, start_exp=0):
    """
    Converts bolus dose amounts to a cumulative dose amount series that can be
    plotted nicely.

    Optionally the start and end of the experiment can be provided, so a
    constant cumulative amount
    is displayed for the entire duration experiment.
    """
    # Get number of measurements
    n = len(times)

    # Define how many cumulative time points are needed
    # (add start and end if needed)
    m = 2 * n + 2

    # Create time container
    cum_times = np.empty(shape=m)

    # Create dose container
    cum_doses = np.empty(shape=m)

    # Add first entry (assuming no base level drug)
    cum_times[0] = 0
    cum_doses[0] = 0
    cum_doses[1] = 0  # At start of first dose there will also be no drug

    # Add start and end time of dose to container
    cum_times[1:-2:2] = times  # start of dose
    cum_times[2:-1:2] = times + duration  # end of dose
    cum_times[-1] = end_exp

    # Add cumulative dose amount at start and end of dose to container
    cum_doses[3:-2:2] = np.cumsum(doses[:-1])  # start of doses
    cum_doses[2:-1:2] = np.cumsum(doses)  # end of doses
    cum_doses[-1] = np.cumsum(doses)[-1]  # final dose level

    return cum_times, cum_doses
'''

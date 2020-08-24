#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import numpy as np


def find_gaussian_hyperparams(left, right):
    """
    Finds mean and standard deviation of Gaussian prior such that the 2
    sigma intervals are the values `left` and `right`. As a result, about
    95% of the prbability mass will be within the defined interval.
    """
    # Compute width of interval
    width = abs(left) + abs(right)

    # Compute mean
    mean = left + width / 2

    # Compute standard deviation
    std = width / 4

    return mean, std


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

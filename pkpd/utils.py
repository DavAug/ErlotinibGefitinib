#
# Convenience functions to tune prior hyperparameters.
#

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

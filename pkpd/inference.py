#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import pints


class HierarchicalLogPosterior(pints.LogPDF):
    r"""
    Represents the log-posterior of an hierarchical model of the form

    .. math::
        \mathbb{P}(y | \theta) = \int \text{d}\psi \, \mathbb{P}(y | \psi )
        \mathbb{P}(\psi | \theta),

    where :math:`\psi` is some hidden state that is parameterised by the
    model parameters :math:`\theta`.

    The hierarchical log-posterior is then constructed by the sum of two
    :class:`LogPDF` and  a :class:`LogPrior`

    .. math::
        \log \mathbb{P}(\psi ,\theta | y^{\text{obs}}) =
        \log \mathbb{P}(y | \psi ) + \mathbb{P}(\psi | \theta) +
        \mathbb{P}(\theta ) + \text{constant},

    where :math:`\log \mathbb{P}(y | \psi )` is the log_likelihood,
    :math:`\log \mathbb{P}(\psi | \theta )` is the log_likelihood_hidden, and
    :math:`\log \mathbb{P}(\theta )` is the log_prior.

    Extends :class:`LogPDF`
    """
    def __init__(self, log_likelihood, log_likelihood_hidden, log_prior):
        super(HierarchicalLogPosterior, self).__init__()

        # Check arguments
        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError(
                'Given prior must extend pints.LogPrior.')
        if not isinstance(log_likelihood_hidden, pints.LogPDF):
            raise ValueError(
                'Given log_likelihood_hidden must extend pints.LogPDF.')
        if not isinstance(log_likelihood, pints.LogPDF):
            raise ValueError(
                'Given log_likelihood must extend pints.LogPDF.')

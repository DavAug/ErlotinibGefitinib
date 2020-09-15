#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import os

from ._likelihoods import (  # noqa
    ConstantAndMultiplicativeGaussianLogLikelihood,
    FixedEtaLogLikelihoodWrapper,
    PooledLogPDF
)

from ._models import (  # noqa
    PharmacodynamicModel
)

from ._optimisation import (  # noqa
    optimise
)

from ._sampling import (  # noqa
    sample
)

from ._utils import (  # noqa
    get_median_parameters
)


class ModelLibrary(object):
    """
    Contains references to SBML models in the pkpd module
    """

    def __init__(self):
        # Get path to model library
        path = os.path.dirname(os.path.abspath(__file__))
        path += '/model_library/'

        # Create model library
        self._model_library = {
            'Tumour growth without treatment':
                path + 'tumour_growth_without_treatment.xml',
            'Tumour growth without treatment - dimensionless':
                path + 'tumour_growth_without_treatment_dimensionless.xml',
            'Tumour growth without treatment - Eigenmann et. al.':
                path + 'tumour_growth_without_treatment_Eigenmann.xml',
        }

    def models(self):
        """
        Returns the names of models in the library.
        """
        return list(self._model_library.keys())

    def get_path(self, name):
        """
        Returns the path to the model.
        """
        return self._model_library[name]

#
# This file is part of the ErlotinibGefitinib repository.
#

#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import pkpd.plots


class TestPlotMeasurements(unittest.TestCase):
    """
    Tests the `pkpd.plots.plot_mesurements` function.
    """

    @classmethod
    def setUpClass(cls):

        # Create test dataset
        ids = [0, 0, 0, 1, 1, 1, 2, 2]
        times = [0, 1, 2, 2, np.nan, 4, 1, 3]
        volumes = [np.nan, 0.3, 0.2, 0.5, 0.1, 0.2, 0.234, 0]
        masses = [1, 1, 1, 1, 1, 1, 1, 1]
        cls.data = pd.DataFrame({
            '#ID': ids,
            'TIME in day': times,
            'TUMOUR VOLUME in cm^3': volumes,
            'BODY WEIGHT in g': masses})

    def test_create_figure(self):

        fig = pkpd.plots.plot_measurements(self.data)

        self.assertIsInstance(fig, go.Figure)
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


# Unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:  # pragma: no python 3 cover
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestPlotMeasurementsAndErrorModel(unittest.TestCase):
    """
    Tests the `pkpd.plots.plot_measurements_and_error_model` function.
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

        # Create test structural model
        path = pkpd.ModelLibrary().get_path('Tumour growth without treatment')
        cls.model = pkpd.PharmacodynamicModel(path)

        # Create test error model
        cls.error = 'multiplicative Gaussian'

        # Create test parameters
        cls.parameters = [[1, 1, 1, 0.1], [2, 2, 2, 0.05], [3, 3, 3, 0.2]]

    def test_bad_data_type(self):

        data = np.ones(shape=(10, 4))

        self.assertRaisesRegex(
            TypeError, 'Data <',
            pkpd.plots.plot_measurements_and_error_model, data, self.model,
            self.error, self.parameters)

    def test_bad_column_keys(self):

        data = self.data.rename(columns={'TIME in day': 'SOMETHING ELSE'})

        self.assertRaisesRegex(
            ValueError, 'Data <',
            pkpd.plots.plot_measurements_and_error_model, data, self.model,
            self.error, self.parameters)

    def test_bad_struc_model_type(self):

        model = 'bad model'

        self.assertRaisesRegex(
            TypeError, 'Structural model `struc_model` needs to be an',
            pkpd.plots.plot_measurements_and_error_model, self.data, model,
            self.error, self.parameters)

    def test_bad_model_n_output(self):

        # Create model
        path = pkpd.ModelLibrary().get_path('Tumour growth without treatment')
        model = pkpd.PharmacodynamicModel(path)

        # Change to multi-output model
        model.set_outputs(['myokit.tumour_volume', 'myokit.tumour_volume'])

        self.assertRaisesRegex(
            ValueError, 'Structural model output dimension has to be 1.',
            pkpd.plots.plot_measurements_and_error_model, self.data, model,
            self.error, self.parameters)

    def test_bad_error_model(self):

        error_model = 'invalid error model'

        self.assertRaisesRegex(
            ValueError, 'Error model <',
            pkpd.plots.plot_measurements_and_error_model, self.data,
            self.model, error_model, self.parameters)

    def test_bad_parameters_dimension(self):

        parameters = np.ones(shape=(3, 3, 3))

        self.assertRaisesRegex(
            ValueError, 'Parameters does not have the correct shape.',
            pkpd.plots.plot_measurements_and_error_model, self.data,
            self.model, self.error, parameters)

    def test_bad_parameter_shape(self):

        parameters = np.ones(shape=(10, 4))

        self.assertRaisesRegex(
            ValueError, 'Parameters does not have the correct shape.',
            pkpd.plots.plot_measurements_and_error_model, self.data,
            self.model, self.error, parameters)

        parameters = np.ones(shape=(3, 10))

        self.assertRaisesRegex(
            ValueError, 'Parameters does not have the correct shape.',
            pkpd.plots.plot_measurements_and_error_model, self.data,
            self.model, self.error, parameters)

    def test_bad_pooled_error(self):

        self.assertRaisesRegex(
            ValueError, 'Pooling of the error model makes only sense',
            pkpd.plots.plot_measurements_and_error_model, self.data,
            self.model, self.error, self.parameters, True)

    def test_unpooled_error(self):

        fig = pkpd.plots.plot_measurements_and_error_model(
            self.data, self.model, self.error, self.parameters)

        self.assertIsInstance(fig, go.Figure)

        # Check buttons
        buttons = fig.layout.updatemenus[0].buttons
        self.assertEqual(len(buttons), 3)

        # Per subplot there is a plot of:
        # [measurements, mean, 2 times 1, 2, 3 sigma interval]
        n_subplots = 2
        n_graph_objects = 8
        n_tot = n_subplots * n_graph_objects

        # Check visibility
        visible = buttons[0].args[0]['visible']
        self.assertEqual(len(visible), n_tot * 3)
        self.assertEqual(visible, [True] * n_tot + [False] * n_tot * 2)

        visible = buttons[1].args[0]['visible']
        self.assertEqual(len(visible), n_tot * 3)
        self.assertEqual(
            visible, [False] * n_tot + [True] * n_tot + [False] * n_tot)

        visible = buttons[2].args[0]['visible']
        self.assertEqual(len(visible), n_tot * 3)
        self.assertEqual(
            visible, [False] * n_tot * 2 + [True] * n_tot)

        # Check button labels
        self.assertEqual(buttons[0].label, 'ID: 0')
        self.assertEqual(buttons[1].label, 'ID: 1')
        self.assertEqual(buttons[2].label, 'ID: 2')

    def test_pooled_error(self):

        parameters = [[1, 1, 1, 0.1], [2, 2, 2, 0.1], [3, 3, 3, 0.1]]

        fig = pkpd.plots.plot_measurements_and_error_model(
            self.data, self.model, self.error, parameters, True)

        self.assertIsInstance(fig, go.Figure)

        # Check no buttons exist
        self.assertEqual(fig.layout.updatemenus, ())

        # Check that all data is visible
        n_subplots = 2
        n_error_plots = 7
        n_ids = 3
        n_tot = n_subplots * n_ids + n_subplots * n_error_plots

        plots = fig.data
        self.assertEqual(len(plots), n_tot)
        self.assertTrue(plots[0].visible)
        self.assertTrue(plots[1].visible)
        self.assertTrue(plots[2].visible)
        self.assertTrue(plots[3].visible)
        self.assertTrue(plots[4].visible)
        self.assertTrue(plots[5].visible)
        self.assertTrue(plots[6].visible)
        self.assertTrue(plots[7].visible)
        self.assertTrue(plots[8].visible)
        self.assertTrue(plots[9].visible)
        self.assertTrue(plots[10].visible)
        self.assertTrue(plots[11].visible)
        self.assertTrue(plots[12].visible)
        self.assertTrue(plots[13].visible)
        self.assertTrue(plots[14].visible)
        self.assertTrue(plots[15].visible)
        self.assertTrue(plots[16].visible)
        self.assertTrue(plots[17].visible)
        self.assertTrue(plots[18].visible)
        self.assertTrue(plots[19].visible)

    def test_error_models(self):

        # Test constant Gaussian error
        error_model = 'constant Gaussian'
        fig = pkpd.plots.plot_measurements_and_error_model(
            self.data, self.model, error_model, self.parameters)

        self.assertIsInstance(fig, go.Figure)

        # Test multiplicative Gaussian error
        error_model = 'multiplicative Gaussian'
        fig = pkpd.plots.plot_measurements_and_error_model(
            self.data, self.model, error_model, self.parameters)

        self.assertIsInstance(fig, go.Figure)

        # Test constant and multiplicative Gaussian error
        error_model = 'combined Gaussian'
        parameters = [
            [1, 1, 1, 0.1, 1],
            [2, 2, 2, 0.05, 1],
            [3, 3, 3, 0.2, 1]]
        fig = pkpd.plots.plot_measurements_and_error_model(
            self.data, self.model, error_model, parameters)

        self.assertIsInstance(fig, go.Figure)


if __name__ == '__main__':
    unittest.main()

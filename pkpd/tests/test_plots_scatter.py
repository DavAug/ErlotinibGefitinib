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

    def test_bad_input(self):

        data = np.ones(shape=(10, 4))

        self.assertRaisesRegex(
            ValueError, 'Data has to be pandas.DataFrame.',
            pkpd.plots.plot_measurements, data)

    def test_bad_column_keys(self):

        data = self.data.rename(columns={'TIME in day': 'SOMETHING ELSE'})

        self.assertRaisesRegex(
            ValueError, 'Data must have key <',
            pkpd.plots.plot_measurements, data)

    def test_create_figure(self):

        fig = pkpd.plots.plot_measurements(self.data)

        self.assertIsInstance(fig, go.Figure)


class TestPlotMeasurementsAndPredictions(unittest.TestCase):
    """
    Tests the `pkpd.plots.plot_mesurements_and_predictions` function.
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

        # Create test model
        path = pkpd.ModelLibrary().get_path('Tumour growth without treatment')
        cls.model = pkpd.PharmacodynamicModel(path)

        # Create test parameters
        cls.parameters = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]

    def test_bad_data_type(self):

        data = np.ones(shape=(10, 4))

        self.assertRaisesRegex(
            TypeError, 'Data has to be pandas.DataFrame.',
            pkpd.plots.plot_measurements_and_predictions, data, self.model,
            self.parameters)

    def test_bad_column_keys(self):

        data = self.data.rename(columns={'TIME in day': 'SOMETHING ELSE'})

        self.assertRaisesRegex(
            ValueError, 'Data must have key <',
            pkpd.plots.plot_measurements_and_predictions, data, self.model,
            self.parameters)

    def test_bad_model_type(self):

        model = 'bad model'

        self.assertRaisesRegex(
            TypeError, 'Model needs to be an instance of',
            pkpd.plots.plot_measurements_and_predictions, self.data, model,
            self.parameters)

    def test_bad_model_n_output(self):

        # Create model
        path = pkpd.ModelLibrary().get_path('Tumour growth without treatment')
        model = pkpd.PharmacodynamicModel(path)

        # Change to multi-output model
        model.set_outputs(['myokit.tumour_volume', 'myokit.tumour_volume'])

        self.assertRaisesRegex(
            ValueError, 'Model output dimension has to be 1.',
            pkpd.plots.plot_measurements_and_predictions, self.data, model,
            self.parameters)

    def test_bad_parameters_dimension(self):

        parameters = np.ones(shape=(3, 3, 3))

        self.assertRaisesRegex(
            ValueError, 'Parameters needs to have dimension 2.',
            pkpd.plots.plot_measurements_and_predictions, self.data,
            self.model, parameters)

    def test_bad_parameter_shape(self):

        parameters = np.ones(shape=(10, 3))

        self.assertRaisesRegex(
            ValueError, 'Parameters does not have the correct shape.',
            pkpd.plots.plot_measurements_and_predictions, self.data,
            self.model, parameters)

        parameters = np.ones(shape=(3, 10))

        self.assertRaisesRegex(
            ValueError, 'Parameters does not have the correct shape.',
            pkpd.plots.plot_measurements_and_predictions, self.data,
            self.model, parameters)

    def test_create_figure(self):

        fig = pkpd.plots.plot_measurements_and_predictions(
            self.data, self.model, self.parameters)

        self.assertIsInstance(fig, go.Figure)


class TestPlotMeasurementsAndSimulation(unittest.TestCase):
    """
    Tests the `pkpd.plots.plot_mesurements_and_simulation` function.
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

        # Create test model
        path = pkpd.ModelLibrary().get_path('Tumour growth without treatment')
        cls.model = pkpd.PharmacodynamicModel(path)

        # Create test parameters
        cls.default_params = [0.1, 0.1, 0.1]
        cls.min_params = [0.01, 0.02, 0.03]
        cls.max_params = [1, 2, 3]

    def test_bad_data_type(self):

        data = np.ones(shape=(10, 4))

        self.assertRaisesRegex(
            TypeError, 'Data has to be pandas.DataFrame.',
            pkpd.plots.plot_measurements_and_simulation, data, self.model,
            self.default_params, self.min_params, self.max_params)

    def test_bad_column_keys(self):

        data = self.data.rename(columns={'TIME in day': 'SOMETHING ELSE'})

        self.assertRaisesRegex(
            ValueError, 'Data must have key <',
            pkpd.plots.plot_measurements_and_simulation, data, self.model,
            self.default_params, self.min_params, self.max_params)

    def test_bad_model_type(self):

        model = 'bad model'

        self.assertRaisesRegex(
            TypeError, 'Model needs to be an instance of',
            pkpd.plots.plot_measurements_and_simulation, self.data, model,
            self.default_params, self.min_params, self.max_params)

    def test_bad_model_n_output(self):

        # Create model
        path = pkpd.ModelLibrary().get_path('Tumour growth without treatment')
        model = pkpd.PharmacodynamicModel(path)

        # Change to multi-output model
        model.set_outputs(['myokit.tumour_volume', 'myokit.tumour_volume'])

        self.assertRaisesRegex(
            ValueError, 'Model output dimension has to be 1.',
            pkpd.plots.plot_measurements_and_simulation, self.data, model,
            self.default_params, self.min_params, self.max_params)

    def test_bad_default_parameters_dimension(self):

        parameters = np.ones(shape=(3, 3))

        self.assertRaisesRegex(
            ValueError, 'Default parameters need to have dimension 1.',
            pkpd.plots.plot_measurements_and_simulation, self.data, self.model,
            parameters, self.min_params, self.max_params)

    def test_bad_min_parameters_dimension(self):

        parameters = np.ones(shape=(3, 3))

        self.assertRaisesRegex(
            ValueError, 'Minimum parameters need to have dimension 1.',
            pkpd.plots.plot_measurements_and_simulation, self.data, self.model,
            self.default_params, parameters, self.max_params)

    def test_bad_max_parameters_dimension(self):

        parameters = np.ones(shape=(3, 3))

        self.assertRaisesRegex(
            ValueError, 'Maximum parameters need to have dimension 1.',
            pkpd.plots.plot_measurements_and_simulation, self.data, self.model,
            self.default_params, self.min_params, parameters)

    def test_bad_default_parameters_shape(self):

        parameters = np.ones(shape=(10,))

        self.assertRaisesRegex(
            ValueError, 'Default parameters do not have the correct shape.',
            pkpd.plots.plot_measurements_and_simulation, self.data, self.model,
            parameters, self.min_params, self.max_params)

    def test_bad_min_parameters_shape(self):

        parameters = np.ones(shape=(10,))

        self.assertRaisesRegex(
            ValueError, 'Minimum parameters do not have the correct shape.',
            pkpd.plots.plot_measurements_and_simulation, self.data, self.model,
            self.default_params, parameters, self.max_params)

    def test_bad_max_parameters_shape(self):

        parameters = np.ones(shape=(10,))

        self.assertRaisesRegex(
            ValueError, 'Maximum parameters do not have the correct shape.',
            pkpd.plots.plot_measurements_and_simulation, self.data, self.model,
            self.default_params, self.min_params, parameters)

    def test_parameter_names(self):

        # Test bad parameter names
        param_names = ['param 1', 'param 2']

        self.assertRaisesRegex(
            ValueError, 'Number of parameter names does not match number of',
            pkpd.plots.plot_measurements_and_simulation, self.data, self.model,
            self.default_params, self.min_params, self.max_params, param_names)

        # Good parameter names
        param_names = ['param 1', 'param 2', 'param 3']
        fig = pkpd.plots.plot_measurements_and_simulation(
            self.data, self.model, self.default_params, self.min_params,
            self.max_params, param_names)

        # Check all labels
        sliders = fig.layout.sliders
        self.assertEqual(len(sliders), 3)
        self.assertEqual(sliders[0].currentvalue.prefix, param_names[0] + ': ')
        self.assertEqual(sliders[1].currentvalue.prefix, param_names[1] + ': ')
        self.assertEqual(sliders[2].currentvalue.prefix, param_names[2] + ': ')

    def test_create_figure(self):

        fig = pkpd.plots.plot_measurements_and_simulation(
            self.data, self.model, self.default_params, self.min_params,
            self.max_params)

        self.assertIsInstance(fig, go.Figure)


if __name__ == '__main__':
    unittest.main()

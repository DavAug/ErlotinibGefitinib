#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np
import plotly.graph_objects as go

import pkpd.plots


# Unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:  # pragma: no python 3 cover
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestPlotOptimisedParameters(unittest.TestCase):
    """
    Tests the `pkpd.plots.plot_optimised_parameters` function.
    """

    @classmethod
    def setUpClass(cls):
        cls.parameters = np.ones(shape=(3, 5, 4))
        cls.scores = np.ones(shape=(3, 5))

    def test_bad_n_individuals(self):

        parameters = np.ones(shape=(5, 3, 4))
        scores = np.ones(shape=(6, 3))

        self.assertRaisesRegex(
            ValueError, 'Parameters and score do not have the same number of',
            pkpd.plots.plot_optimised_parameters, parameters, scores)

    def test_bad_n_runs(self):

        parameters = np.ones(shape=(5, 4, 4))
        scores = np.ones(shape=(5, 3))

        self.assertRaisesRegex(
            ValueError, 'Parameters and score do not have the same number of',
            pkpd.plots.plot_optimised_parameters, parameters, scores)

    def test_xlabels(self):

        # Test bad xlabels
        xlabels = ['param 1', 'param 2']

        self.assertRaisesRegex(
            ValueError, 'Number of x labels does not match number of',
            pkpd.plots.plot_optimised_parameters, self.parameters, self.scores,
            xlabels)

        # Good xlabels
        xlabels = ['param 1', 'param 2', 'param 3', 'param 4', 'score']
        fig = pkpd.plots.plot_optimised_parameters(
            self.parameters, self.scores, xlabels)

        # Check all labels
        boxes = fig.data
        self.assertEqual(len(boxes), 15)
        self.assertEqual(boxes[0].name, 'param 1')
        self.assertEqual(boxes[1].name, 'param 2')
        self.assertEqual(boxes[2].name, 'param 3')
        self.assertEqual(boxes[3].name, 'param 4')
        self.assertEqual(boxes[4].name, 'score')
        self.assertEqual(boxes[5].name, 'param 1')
        self.assertEqual(boxes[6].name, 'param 2')
        self.assertEqual(boxes[7].name, 'param 3')
        self.assertEqual(boxes[8].name, 'param 4')
        self.assertEqual(boxes[9].name, 'score')
        self.assertEqual(boxes[10].name, 'param 1')
        self.assertEqual(boxes[11].name, 'param 2')
        self.assertEqual(boxes[12].name, 'param 3')
        self.assertEqual(boxes[13].name, 'param 4')
        self.assertEqual(boxes[14].name, 'score')

    def test_ids(self):

        # Test bad ids
        ids = ['1', '2']

        self.assertRaisesRegex(
            ValueError, 'Number of provided ids do not match number',
            pkpd.plots.plot_optimised_parameters, self.parameters, self.scores,
            None, ids)

        # Test good ids
        ids = ['id 1', 'id 2', 'id 3']
        fig = pkpd.plots.plot_optimised_parameters(
            self.parameters, self.scores, None, ids)

        # Check all button labels
        buttons = fig.layout.updatemenus[0].buttons
        self.assertEqual(len(buttons), 3)
        self.assertEqual(buttons[0].label, 'ID: id 1')
        self.assertEqual(buttons[1].label, 'ID: id 2')
        self.assertEqual(buttons[2].label, 'ID: id 3')

    def test_create_figure(self):

        fig = pkpd.plots.plot_optimised_parameters(
            self.parameters, self.scores)

        self.assertIsInstance(fig, go.Figure)


if __name__ == '__main__':
    unittest.main()

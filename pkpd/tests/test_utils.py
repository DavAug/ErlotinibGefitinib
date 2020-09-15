#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import pkpd

# Unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:  # pragma: no python 3 cover
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestGetMedianParameters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 2 individuals, 3 runs, 4 params
        id_1 = [
            [1, 2, 3, 4],
            [0.9, 1.9, 3.1, 4.1],
            [1.1, 2.1, 2.9, 3.9]
        ]
        id_2 = [
            [5, 6, 7, 8],
            [5.1, 6.1, 6.9, 7.9],
            [4.9, 5.9, 7.1, 8.1]
        ]
        cls.parameters = [id_1, id_2]
        cls.pooled = [False, True, False, True]

    def test_bad_parameters(self):
        parameters = self.parameters[0]

        self.assertRaisesRegex(
            ValueError, 'Parameters has to be of dimension 3.',
            pkpd.get_median_parameters, parameters, self.pooled)

    def test_bad_pooled_length(self):
        pooled = [True]

        self.assertRaisesRegex(
            ValueError, 'The array-like object `pooled`',
            pkpd.get_median_parameters, self.parameters, pooled)

    def test_bad_pooled_content(self):
        pooled = ['True', 'True', 'True', 'True']

        self.assertRaisesRegex(
            TypeError, 'The array-like object has to contain',
            pkpd.get_median_parameters, self.parameters, pooled)

    def test_call_partially_pooled(self):
        medians = pkpd.get_median_parameters(self.parameters, self.pooled)

        self.assertEqual(medians.shape, (6,))
        self.assertEqual(medians[0], 1)
        self.assertEqual(medians[1], 3)
        self.assertEqual(medians[2], 5)
        self.assertEqual(medians[3], 7)
        self.assertEqual(medians[4], (2.1 + 5.9) / 2)
        self.assertEqual(medians[5], (4.1 + 7.9) / 2)

    def test_call_pooled(self):
        pooled = [True, True, True, True]
        medians = pkpd.get_median_parameters(self.parameters, pooled)

        self.assertEqual(medians.shape, (4,))
        self.assertEqual(medians[0], (1.1 + 4.9) / 2)
        self.assertEqual(medians[1], (2.1 + 5.9) / 2)
        self.assertEqual(medians[2], (3.1 + 6.9) / 2)
        self.assertEqual(medians[3], (4.1 + 7.9) / 2)

    def test_call_unpooled(self):
        pooled = [False, False, False, False]
        medians = pkpd.get_median_parameters(self.parameters, pooled)

        self.assertEqual(medians.shape, (8,))
        self.assertEqual(medians[0], 1)
        self.assertEqual(medians[1], 2)
        self.assertEqual(medians[2], 3)
        self.assertEqual(medians[3], 4)
        self.assertEqual(medians[4], 5)
        self.assertEqual(medians[5], 6)
        self.assertEqual(medians[6], 7)
        self.assertEqual(medians[7], 8)


if __name__ == '__main__':
    unittest.main()

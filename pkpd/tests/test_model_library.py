#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import os.path
import unittest

import pkpd


class TestModelLibrary(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_library = pkpd.ModelLibrary()

    def test_models(self):
        models = self.model_library.models()

        self.assertEqual(models[0], 'Tumour growth without treament')
        self.assertEqual(
            models[1], 'Tumour growth without treament - dimensionless')
        self.assertEqual(
            models[2], 'Tumour growth without treament - Eigenmann et. al.')

    def test_get_path(self):
        models = self.model_library.models()

        path = self.model_library.get_path(models[0])
        self.assertTrue(os.path.exists(path))
        path = self.model_library.get_path(models[1])
        self.assertTrue(os.path.exists(path))
        path = self.model_library.get_path(models[2])
        self.assertTrue(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()

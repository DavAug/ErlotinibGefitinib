#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import os.path
import unittest

import myokit
import myokit.formats.sbml as sbml

import pkpd


class TestModelLibrary(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_library = pkpd.ModelLibrary()

    def test_models(self):
        models = self.model_library.models()

        self.assertIn('Tumour growth without treatment', models)
        self.assertIn(
            'Tumour growth without treatment - dimensionless', models)
        self.assertIn(
            'Tumour growth without treatment - Eigenmann et. al.', models)

    def test_get_path(self):
        models = self.model_library.models()

        path = self.model_library.get_path(models[0])
        self.assertTrue(os.path.exists(path))
        path = self.model_library.get_path(models[1])
        self.assertTrue(os.path.exists(path))
        path = self.model_library.get_path(models[2])
        self.assertTrue(os.path.exists(path))

    def test_model_validity(self):
        importer = sbml.SBMLImporter()
        models = self.model_library.models()

        # Test model 1
        path = self.model_library.get_path(models[0])
        myokit_model = importer.model(path)
        self.assertIsInstance(myokit_model, myokit.Model)

        # Test model 2
        path = self.model_library.get_path(models[1])
        myokit_model = importer.model(path)
        self.assertIsInstance(myokit_model, myokit.Model)

        # Test model 3
        path = self.model_library.get_path(models[2])
        myokit_model = importer.model(path)
        self.assertIsInstance(myokit_model, myokit.Model)


if __name__ == '__main__':
    unittest.main()

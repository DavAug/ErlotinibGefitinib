#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np
import pints
import pints.toy

import pkpd


# Unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:  # pragma: no python 3 cover
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestConstantAndMultiplicativeGaussianLogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(3)

        # Generate test data
        cls.times = np.array([1, 2, 3, 4])
        cls.n_times = len(cls.times)
        cls.data_single = np.array([1, 2, 3, 4]) / 5.0
        cls.data_multi = np.array([
            [10.7, 3.5, 3.8],
            [1.1, 3.2, -1.4],
            [9.3, 0.0, 4.5],
            [1.2, -3, -10]])

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        test_parameters = [2.0, 0.5, 1.1, 1.0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -8.222479586661642)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        test_parameters = [2.0, 0.5, 1.1, 1.0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -8.222479586661642)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        test_parameters = [2.0, 0.5, 1.1, 1.0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -8.222479586661642)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        log_likelihood = pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        test_parameters = [
            2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -42.87921520701031)

    def test_call_gaussian_log_likelihood_agrees_single(self):
        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, self.data_single)

        # Create CombinedGaussianLL and GaussianLL
        log_likelihood = pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)
        gauss_log_likelihood = pints.GaussianLogLikelihood(problem)

        # Check that CombinedGaussianLL agrees with GaussianLoglikelihood when
        # sigma_rel = 0 and sigma_base = sigma
        test_parameters = [2.0, 0.5, 1.1, 0.0]
        gauss_test_parameters = [2.0, 0.5]
        score = log_likelihood(test_parameters)
        gauss_score = gauss_log_likelihood(gauss_test_parameters)
        self.assertAlmostEqual(score, gauss_score)

    def test_call_gaussian_log_likelihood_agrees_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create CombinedGaussianLL and GaussianLL
        log_likelihood = pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)
        gauss_log_likelihood = pints.GaussianLogLikelihood(problem)

        # Check that CombinedGaussianLL agrees with GaussianLoglikelihood when
        # sigma_rel = 0 and sigma_base = sigma
        test_parameters = [
            2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 1.1, 1.1, 1.1, 0.0, 0.0, 0.0]
        gauss_test_parameters = [2.0, 2.0, 2.0, 0.5, 0.5, 0.5]
        score = log_likelihood(test_parameters)
        gauss_score = gauss_log_likelihood(gauss_test_parameters)
        self.assertAlmostEqual(score, gauss_score)

    def test_call_multiplicative_gaussian_log_likelihood_agrees_single(self):
        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, self.data_single)

        # Create CombinedGaussianLL and MultplicativeGaussianLL
        log_likelihood = pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)
        multi_log_likelihood = pints.MultiplicativeGaussianLogLikelihood(
            problem)

        # Check that CombinedGaussianLL agrees with
        # MultiplicativeGaussianLoglikelihood when sigma_base = 0,
        # eta = eta, and sigma_rel = sigma
        test_parameters = [2.0, 0.0, 1.1, 1.0]
        multi_test_parameters = [2.0, 1.1, 1.0]
        score = log_likelihood(test_parameters)
        multi_score = multi_log_likelihood(multi_test_parameters)
        self.assertAlmostEqual(score, multi_score)

    def test_call_multiplicative_gaussian_log_likelihood_agrees_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create CombinedGaussianLL and MultplicativeGaussianLL
        log_likelihood = pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)
        multi_log_likelihood = pints.MultiplicativeGaussianLogLikelihood(
            problem)

        # Check that CombinedGaussianLL agrees with
        # MultiplicativeGaussianLoglikelihood when sigma_base = 0,
        # eta = eta, and sigma_rel = sigma
        test_parameters = [
            2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0]
        multi_test_parameters = [2.0, 2.0, 2.0, 1.1, 1.0, 1.1, 1.0, 1.1, 1.0]
        score = log_likelihood(test_parameters)
        multi_score = multi_log_likelihood(multi_test_parameters)
        self.assertAlmostEqual(score, multi_score)

    def test_evaluateS1_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        test_parameters = [2.0, 0.5, 1.1, 1.0]
        score, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that likelihood score agrees with call
        # There are floating point deviations because in evaluateS1
        # log(sigma_tot) is for efficiency computed as -log(1/sigma_tot)
        self.assertAlmostEqual(score, log_likelihood(test_parameters))

        # Check that number of partials is correct
        self.assertAlmostEqual(deriv.shape, (4,))

        # Check that partials are computed correctly
        self.assertAlmostEqual(deriv[0], -2.055351334007383)
        self.assertAlmostEqual(deriv[1], -1.0151215581116324)
        self.assertAlmostEqual(deriv[2], -1.5082610203777322)
        self.assertAlmostEqual(deriv[3], -2.1759606944650822)

    def test_evaluateS1_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        test_parameters = [2.0, 0.5, 1.1, 1.0]
        score, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that likelihood score agrees with call
        # There are floating point deviations because in evaluateS1
        # log(sigma_tot) is for efficiency computed as -log(1/sigma_tot)
        self.assertAlmostEqual(score, log_likelihood(test_parameters))

        # Check that number of partials is correct
        self.assertAlmostEqual(deriv.shape, (4,))

        # Check that partials are computed correctly
        self.assertAlmostEqual(deriv[0], -2.055351334007383)
        self.assertAlmostEqual(deriv[1], -1.0151215581116324)
        self.assertAlmostEqual(deriv[2], -1.5082610203777322)
        self.assertAlmostEqual(deriv[3], -2.1759606944650822)

    def test_evaluateS1_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        test_parameters = [2.0, 0.5, 1.1, 1.0]
        score, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that likelihood score agrees with call
        # There are floating point deviations because in evaluateS1
        # log(sigma_tot) is for efficiency computed as -log(1/sigma_tot)
        self.assertAlmostEqual(score, log_likelihood(test_parameters))

        # Check that number of partials is correct
        self.assertAlmostEqual(deriv.shape, (4,))

        # Check that partials are computed correctly
        self.assertAlmostEqual(deriv[0], -2.0553513340073835)
        self.assertAlmostEqual(deriv[1], -1.0151215581116324)
        self.assertAlmostEqual(deriv[2], -1.5082610203777322)
        self.assertAlmostEqual(deriv[3], -2.1759606944650822)

    def test_evaluateS1_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        log_likelihood = pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        test_parameters = [
            2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0]
        score, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that likelihood score agrees with call
        self.assertAlmostEqual(score, log_likelihood(test_parameters))

        # Check that number of partials is correct
        self.assertAlmostEqual(deriv.shape, (12,))

        # Check that partials are computed correctly
        self.assertAlmostEqual(deriv[0], 8.585990509232376)
        self.assertAlmostEqual(deriv[1], -1.6726936107293917)
        self.assertAlmostEqual(deriv[2], -0.6632862192355309)
        self.assertAlmostEqual(deriv[3], 5.547071959874058)
        self.assertAlmostEqual(deriv[4], -0.2868738955802226)
        self.assertAlmostEqual(deriv[5], 0.1813851785335695)
        self.assertAlmostEqual(deriv[6], 8.241803503682762)
        self.assertAlmostEqual(deriv[7], -1.82731103999105)
        self.assertAlmostEqual(deriv[8], 2.33264086991343)
        self.assertAlmostEqual(deriv[9], 11.890409042744405)
        self.assertAlmostEqual(deriv[10], -1.3181262877783717)
        self.assertAlmostEqual(deriv[11], 1.3018716574264304)

    def test_evaluateS1_gaussian_log_likelihood_agrees_single(self):
        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, self.data_single)

        # Create CombinedGaussianLL and GaussianLL
        log_likelihood = pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)
        gauss_log_likelihood = pints.GaussianLogLikelihood(problem)

        # Check that CombinedGaussianLL agrees with GaussianLoglikelihood when
        # sigma_rel = 0 and sigma_base = sigma
        test_parameters = [2.0, 0.5, 1.1, 0.0]
        gauss_test_parameters = [2.0, 0.5]
        score, deriv = log_likelihood.evaluateS1(test_parameters)
        gauss_score, gauss_deriv = gauss_log_likelihood.evaluateS1(
            gauss_test_parameters)

        # Check that scores are the same
        self.assertAlmostEqual(score, gauss_score)

        # Check that partials for model params and sigma_base agree
        self.assertAlmostEqual(deriv[0], gauss_deriv[0])
        self.assertAlmostEqual(deriv[1], gauss_deriv[1])

    def test_evaluateS1_gaussian_log_likelihood_agrees_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create CombinedGaussianLL and GaussianLL
        log_likelihood = pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)
        gauss_log_likelihood = pints.GaussianLogLikelihood(problem)

        # Check that CombinedGaussianLL agrees with GaussianLoglikelihood when
        # sigma_rel = 0 and sigma_base = sigma
        test_parameters = [
            2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 1.1, 1.1, 1.1, 0.0, 0.0, 0.0]
        gauss_test_parameters = [2.0, 2.0, 2.0, 0.5, 0.5, 0.5]
        score, deriv = log_likelihood.evaluateS1(test_parameters)
        gauss_score, gauss_deriv = gauss_log_likelihood.evaluateS1(
            gauss_test_parameters)

        # Check that scores are the same
        self.assertAlmostEqual(score, gauss_score)

        # Check that partials for model params and sigma_base agree
        self.assertAlmostEqual(deriv[0], gauss_deriv[0])
        self.assertAlmostEqual(deriv[1], gauss_deriv[1])
        self.assertAlmostEqual(deriv[2], gauss_deriv[2])
        self.assertAlmostEqual(deriv[3], gauss_deriv[3])
        self.assertAlmostEqual(deriv[4], gauss_deriv[4])
        self.assertAlmostEqual(deriv[5], gauss_deriv[5])

    def test_evaluateS1_finite_difference_single(self):
        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, self.data_single)

        # Create log-likelihood
        log_likelihood = pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Compute derivatives with evaluateS1
        test_parameters = np.array([2.0, 0.5, 1.1, 1.0])
        _, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that finite difference approximately agrees with evaluateS1
        # Theta
        eps = np.array([1E-3, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[0], (score_after - score_before) / eps[0])

        # Sigma base
        eps = np.array([0, 1E-3, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[1], (score_after - score_before) / eps[1])

        # Eta
        eps = np.array([0, 0, 1E-4, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[2], (score_after - score_before) / eps[2])

        # Sigma rel
        eps = np.array([0, 0, 0, 1E-4])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[3], (score_after - score_before) / eps[3])

    def test_evaluateS1_finite_difference_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log-likelihood
        log_likelihood = pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Compute derivatives with evaluateS1
        test_parameters = [
            2.0, 1.9, 2.1, 0.5, 0.4, 0.6, 1.1, 1.0, 1.2, 1.0, 0.9, 1.1]
        _, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that finite difference approximately agrees with evaluateS1
        # Theta output 1
        eps = np.array([1E-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[0], (score_after - score_before) / eps[0])

        # Theta output 2
        eps = np.array([0, 1E-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[1], (score_after - score_before) / eps[1])

        # Theta output 3
        eps = np.array([0, 0, 1E-4, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[2], (score_after - score_before) / eps[2])

        # Sigma base output 1
        eps = np.array([0, 0, 0, 1E-4, 0, 0, 0, 0, 0, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[3], (score_after - score_before) / eps[3])

        # Sigma base output 2
        eps = np.array([0, 0, 0, 0, 1E-4, 0, 0, 0, 0, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[4], (score_after - score_before) / eps[4])

        # Sigma base output 3
        eps = np.array([0, 0, 0, 0, 0, 1E-4, 0, 0, 0, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[5], (score_after - score_before) / eps[5])

        # Eta output 1
        eps = np.array([0, 0, 0, 0, 0, 0, 1E-4, 0, 0, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[6], (score_after - score_before) / eps[6])

        # Eta output 2
        eps = np.array([0, 0, 0, 0, 0, 0, 0, 1E-4, 0, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[7], (score_after - score_before) / eps[7])

        # Eta output 3
        eps = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1E-4, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[8], (score_after - score_before) / eps[8])

        # Sigma rel ouput 1
        eps = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1E-4, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[9], (score_after - score_before) / eps[9])

        # Sigma rel ouput 2
        eps = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1E-4, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(
            deriv[10], (score_after - score_before) / eps[10])

        # Sigma rel ouput 3
        eps = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1E-4])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(
            deriv[11], (score_after - score_before) / eps[11])


class TestFixedEtaLogLikelihoodWrapper(unittest.TestCase):
    """
    Tests the `pkpd.FixedEtaLogLikelihoodWrapper`.
    """

    @classmethod
    def setUpClass(cls):
        # Create single output model
        model = pints.toy.ConstantModel(1)

        # Generate data
        times = np.array([1, 2, 3, 4])
        data = np.array([1, 2, 3, 4]) / 5.0

        # Create problem
        problem = pints.SingleOutputProblem(model, times, data)

        # Create likelihoods
        cls.multplicative_log_likelihood = \
            pints.MultiplicativeGaussianLogLikelihood(problem)
        cls.c_and_m_log_likelihood = \
            pkpd.ConstantAndMultiplicativeGaussianLogLikelihood(problem)

    def test_bad_likelihood(self):
        # Create single output model
        model = pints.toy.ConstantModel(1)

        # Generate data
        times = np.array([1, 2, 3, 4])
        data = np.array([1, 2, 3, 4]) / 5.0

        # Create problem
        problem = pints.SingleOutputProblem(model, times, data)

        # Create "bad" likelihood
        log_likelihood = pints.GaussianLogLikelihood(problem)

        # Check that error is thown when we attempt to fix eta
        eta = 1
        self.assertRaisesRegex(
            ValueError, 'This likelihood wrapper is only defined for a ',
            pkpd.FixedEtaLogLikelihoodWrapper, log_likelihood, eta)

    def test_bad_n_output(self):
        # Create multi output model
        model = pints.toy.ConstantModel(3)

        # Generate data
        times = np.array([1, 2, 3, 4])
        data = np.array([
            [10.7, 3.5, 3.8],
            [1.1, 3.2, -1.4],
            [9.3, 0.0, 4.5],
            [1.2, -3, -10]])

        # Create problem
        problem = pints.MultiOutputProblem(model, times, data)

        # Create "bad" likelihood
        log_likelihood = pints.MultiplicativeGaussianLogLikelihood(problem)

        # Check that error is thown when we attempt to fix eta
        eta = 1
        self.assertRaisesRegex(
            ValueError, 'This likelihood wrapper is only defined for a ',
            pkpd.FixedEtaLogLikelihoodWrapper, log_likelihood, eta)

    def test_call_multiplicative_log_likelihood(self):

        # Fix eta
        eta = 1.1
        wrapped_log_likelihood = pkpd.FixedEtaLogLikelihoodWrapper(
            self.multplicative_log_likelihood, eta)

        # Check that computed likelihoods agree
        params = [1, 2]
        params_plus_eta = [1] + [eta] + [2]
        self.assertEqual(
            wrapped_log_likelihood(params),
            self.multplicative_log_likelihood(params_plus_eta))

    def test_call_constant_and_multiplicative_log_likelihood(self):

        # Fix eta
        eta = 1.1
        wrapped_log_likelihood = pkpd.FixedEtaLogLikelihoodWrapper(
            self.c_and_m_log_likelihood, eta)

        # Check that computed likelihoods agree
        params = [1, 2, 1]
        params_plus_eta = [1, 2] + [eta] + [1]
        self.assertEqual(
            wrapped_log_likelihood(params),
            self.c_and_m_log_likelihood(params_plus_eta))

    def test_evaluateS1(self):

        # Fix eta
        eta = 1.1
        wrapped_log_likelihood = pkpd.FixedEtaLogLikelihoodWrapper(
            self.multplicative_log_likelihood, eta)

        # Check that computed likelihoods agree
        params = [1, 2]

        self.assertRaisesRegex(
            NotImplementedError, 'Method has not been implemented.',
            wrapped_log_likelihood.evaluateS1, params)

    def test_n_parameters(self):

        # Fix eta
        eta = 1.1
        wrapped_log_likelihood = pkpd.FixedEtaLogLikelihoodWrapper(
            self.multplicative_log_likelihood, eta)

        self.assertEqual(wrapped_log_likelihood.n_parameters(), 2)


if __name__ == '__main__':
    unittest.main()

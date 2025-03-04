#testing script for SimplifiedThreePL
import unittest
import numpy as np
from scipy.special import expit
from src.SignalDetection import SignalDetection
from src.Experiment import Experiment
from src.SimplifiedThreePL import SimplifiedThreePL

class TestSimplifiedThreePL(unittest.TestCase):

    def setUp(self):
        # Create a sample experiment for testing
        conditions = [
            SDT(hits=55, misses=45, false_alarms=45, correct_rejections=55),
            SDT(hits=60, misses=40, false_alarms=40, correct_rejections=60),
            SDT(hits=75, misses=25, false_alarms=25, correct_rejections=75),
            SDT(hits=90, misses=10, false_alarms=10, correct_rejections=90),
            SDT(hits=95, misses=5, false_alarms=5, correct_rejections=95)
        ]
        self.experiment = Experiment(conditions)
        self.model = SimplifiedThreePL(self.experiment)

    def test_constructor(self):
        # Test that constructor properly handles valid inputs
        self.assertIsInstance(self.model, SimplifiedThreePL)
        
        # Test that constructor raises appropriate exceptions for invalid inputs
        with self.assertRaises(ValueError):
            SimplifiedThreePL(None)

    def test_predict(self):
        # Test that predict() outputs values between 0 and 1 (inclusive)
        self.model.fit()
        predictions = self.model.predict([self.model.get_discrimination(), self.model.get_logit_base_rate()])
        self.assertTrue(all(0 <= p <= 1 for p in predictions))

        # Test that higher base rate values result in higher probabilities
        low_base_rate = self.model.predict([1.0, -2.0])  # logit(-2) ≈ 0.12
        high_base_rate = self.model.predict([1.0, 2.0])  # logit(2) ≈ 0.88
        self.assertTrue(all(h > l for h, l in zip(high_base_rate, low_base_rate)))

        # Test difficulty and ability effects
        self.model.set_discrimination(1.0)
        self.model.set_logit_base_rate(0)  # c = 0.5
        low_ability = self.model.predict([1.0, 0])
        self.model._person_param = 2  # Set higher ability
        high_ability = self.model.predict([1.0, 0])
        self.assertTrue(all(h > l for h, l in zip(high_ability, low_ability)))

        # Test with negative discrimination
        self.model.set_discrimination(-1.0)
        self.model._person_param = 0
        neg_disc_low_ability = self.model.predict([-1.0, 0])
        self.model._person_param = 2
        neg_disc_high_ability = self.model.predict([-1.0, 0])
        self.assertTrue(all(l > h for l, h in zip(neg_disc_low_ability, neg_disc_high_ability)))

    def test_parameter_estimation(self):
        # Test that negative_log_likelihood improves after fitting
        initial_nll = self.model.negative_log_likelihood([1.0, 0.0])
        self.model.fit()
        fitted_nll = self.model.negative_log_likelihood([self.model.get_discrimination(), self.model.get_logit_base_rate()])
        self.assertLess(fitted_nll, initial_nll)

        # Test that a larger estimate of a is returned for steeper curve
        steep_conditions = [
            SDT(hits=51, misses=49, false_alarms=49, correct_rejections=51),
            SDT(hits=60, misses=40, false_alarms=40, correct_rejections=60),
            SDT(hits=80, misses=20, false_alarms=20, correct_rejections=80),
            SDT(hits=95, misses=5, false_alarms=5, correct_rejections=95),
            SDT(hits=99, misses=1, false_alarms=1, correct_rejections=99)
        ]
        steep_experiment = Experiment(steep_conditions)
        steep_model = SimplifiedThreePL(steep_experiment)
        steep_model.fit()
        self.assertGreater(steep_model.get_discrimination(), self.model.get_discrimination())

        # Verify that the user cannot request parameter estimates before the model is fit
        unfit_model = SimplifiedThreePL(self.experiment)
        with self.assertRaises(ValueError):
            unfit_model.get_discrimination()
        with self.assertRaises(ValueError):
            unfit_model.get_base_rate()

    def test_multiple_fits(self):
        # Test that parameters remain approximately stable when fitting multiple times
        self.model.fit()
        initial_discrimination = self.model.get_discrimination()
        initial_base_rate = self.model.get_base_rate()
        
        for _ in range(5):
            self.model.fit()
            self.assertAlmostEqual(initial_discrimination, self.model.get_discrimination(), places=4)
            self.assertAlmostEqual(initial_base_rate, self.model.get_base_rate(), places=4)

    def test_integration(self):
        # Integration test
        conditions = [
            SDT(hits=55, misses=45, false_alarms=45, correct_rejections=55),
            SDT(hits=60, misses=40, false_alarms=40, correct_rejections=60),
            SDT(hits=75, misses=25, false_alarms=25, correct_rejections=75),
            SDT(hits=90, misses=10, false_alarms=10, correct_rejections=90),
            SDT(hits=95, misses=5, false_alarms=5, correct_rejections=95)
        ]
        experiment = Experiment(conditions)
        model = SimplifiedThreePL(experiment)
        model.fit()
        
        predictions = model.predict([model.get_discrimination(), model.get_logit_base_rate()])
        observed = [0.55, 0.60, 0.75, 0.90, 0.95]
        
        for pred, obs in zip(predictions, observed):
            self.assertAlmostEqual(pred, obs, places=2)

    def test_corruption(self):
        # Corruption tests
        with self.assertRaises(ValueError):
            self.model.set_discrimination(-1.0)
        
        with self.assertRaises(ValueError):
            self.model.set_base_rate(1.5)
        
        with self.assertRaises(ValueError):
            self.model.set_base_rate(0)

if __name__ == '__main__':
    unittest.main()

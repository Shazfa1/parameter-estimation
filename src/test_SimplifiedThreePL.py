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
        self.experiment = Experiment()
        conditions = [
            SignalDetection(hits=55, misses=45, falseAlarms=45, correctRejections=55),
            SignalDetection(hits=60, misses=40, falseAlarms=40, correctRejections=60),
            SignalDetection(hits=75, misses=25, falseAlarms=25, correctRejections=75),
            SignalDetection(hits=90, misses=10, falseAlarms=10, correctRejections=90),
            SignalDetection(hits=95, misses=5, falseAlarms=5, correctRejections=95)
        ]
        for i, condition in enumerate(conditions):
            self.experiment.add_condition(condition, f"Condition {i+1}")
        
        self.model = SimplifiedThreePL(self.experiment)
    
    def test_constructor(self):
        # Test that constructor properly handles valid inputs
        valid_experiment = Experiment()
        valid_conditions = [
            SignalDetection(hits=55, misses=45, falseAlarms=45, correctRejections=55),
            SignalDetection(hits=60, misses=40, falseAlarms=40, correctRejections=60),
            SignalDetection(hits=75, misses=25, falseAlarms=25, correctRejections=75)
        ]
        for i, condition in enumerate(valid_conditions):
            valid_experiment.add_condition(condition, f"Condition {i+1}")
        
        valid_model = SimplifiedThreePL(valid_experiment)
        self.assertIsInstance(valid_model, SimplifiedThreePL)
        
        # Test that constructor raises appropriate exceptions for invalid inputs
        
        # Test with None
        with self.assertRaises(ValueError):
            SimplifiedThreePL(None)
        
        # Test with an empty experiment
        empty_experiment = Experiment()
        with self.assertRaises(ValueError):
            SimplifiedThreePL(empty_experiment)
        
        # Test trying to access parameter estimates that aren't determined yet
        untrained_model = SimplifiedThreePL(valid_experiment)
        with self.assertRaises(ValueError):
            untrained_model.get_discrimination()
        with self.assertRaises(ValueError):
            untrained_model.get_base_rate()
        with self.assertRaises(ValueError):
            untrained_model.get_logit_base_rate()

    def test_predict(self):
        # Test that predict() outputs values between 0 and 1 (inclusive)
        self.model.fit()
        predictions = self.model.predict([self.model.get_discrimination(), self.model.get_logit_base_rate()])
        self.assertTrue(all(0 <= p <= 1 for p in predictions))
    
        # Test that higher base rate values result in higher probabilities
        low_base_rate = self.model.predict([1.0, -2.0])  # logit(-2) ≈ 0.12
        high_base_rate = self.model.predict([1.0, 2.0])  # logit(2) ≈ 0.88
        self.assertTrue(all(h > l for h, l in zip(high_base_rate, low_base_rate)))
    
        # Test difficulty effects
        self.model.set_discrimination(1.0)
        self.model.set_logit_base_rate(0)  # c = 0.5
        easy_difficulty = self.model.predict([1.0, 0])
        self.model._difficulty_params = np.array([4, 3, 2, 1, 0])  # Increase difficulty
        hard_difficulty = self.model.predict([1.0, 0])
        self.assertTrue(all(e > h for e, h in zip(easy_difficulty, hard_difficulty)))
    
        # Test ability effects
        self.model._difficulty_params = np.array([2, 1, 0, -1, -2])  # Reset difficulty
        low_ability = self.model.predict([1.0, 0])
        self.model._person_param = 2  # Set higher ability
        high_ability = self.model.predict([1.0, 0])
        self.assertTrue(all(h > l for h, l in zip(high_ability, low_ability)))
    
        # Test with very low (near-zero) discrimination
        self.model.set_discrimination(0.0001)  # Very low, positive discrimination
        self.model._person_param = 0
        low_disc_low_ability = self.model.predict([0.0001, 0])
        self.model._person_param = 2
        low_disc_high_ability = self.model.predict([0.0001, 0])
        self.assertTrue(all(abs(l - h) < 1e-6 for l, h in zip(low_disc_low_ability, low_disc_high_ability)))

        # Test with known parameter values
        self.model.set_discrimination(1.0)
        self.model.set_logit_base_rate(0)  # c = 0.5
        self.model._person_param = 0
        expected_output = 0.5 + 0.5 / (1 + np.exp(-1 * (0 - np.array([2, 1, 0, -1, -2]))))
        actual_output = self.model.predict([1.0, 0])
        np.testing.assert_almost_equal(actual_output, expected_output)

    def test_parameter_estimation(self):
        # Test that negative_log_likelihood improves after fitting
        initial_nll = self.model.negative_log_likelihood([1.0, 0.0])
        self.model.fit()
        fitted_nll = self.model.negative_log_likelihood([self.model.get_discrimination(), self.model.get_logit_base_rate()])
        self.assertLess(fitted_nll, initial_nll)

        # Test that a larger estimate of a is returned for steeper curve
        steep_conditions = [
            SignalDetection(hits=51, misses=49, falseAlarms=49, correctRejections=51),
            SignalDetection(hits=60, misses=40, falseAlarms=40, correctRejections=60),
            SignalDetection(hits=80, misses=20, falseAlarms=20, correctRejections=80),
            SignalDetection(hits=95, misses=5, falseAlarms=5, correctRejections=95),
            SignalDetection(hits=99, misses=1, falseAlarms=1, correctRejections=99)
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
            SignalDetection(hits=55, misses=45, falseAlarms=45, correctRejections=55),
            SignalDetection(hits=60, misses=40, falseAlarms=40, correctRejections=60),
            SignalDetection(hits=75, misses=25, falseAlarms=25, correctRejections=75),
            SignalDetection(hits=90, misses=10, falseAlarms=10, correctRejections=90),
            SignalDetection(hits=95, misses=5, falseAlarms=5, correctRejections=95)
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

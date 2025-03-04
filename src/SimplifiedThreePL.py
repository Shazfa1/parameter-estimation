import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from src.Experiment import Experiment

class SimplifiedThreePL:
    def __init__(self, experiment: Experiment):
        if experiment is None:
            raise ValueError("Experiment cannot be None")
        self.experiment = experiment
        self.experiment = experiment
        self._base_rate = None  # c
        self._logit_base_rate = None  # q (logit of c)
        self._discrimination = None  # alpha
        self._is_fitted = False
        self._difficulty_params = np.array([2, 1, 0, -1, -2])  # bi
        self._person_param = 0  # theta

    def summary(self):
        n_total = sum(sdt.hits + sdt.misses + sdt.falseAlarms + sdt.correctRejections 
                      for sdt in self.experiment.conditions)
        n_correct = sum(sdt.hits + sdt.correctRejections for sdt in self.experiment.conditions)
        n_incorrect = n_total - n_correct
        n_conditions = len(self.experiment.conditions)

        return {
            "n_total": n_total,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_conditions": n_conditions
        }

    def predict(self, parameters):
        alpha, q = parameters
        c = expit(q)  # Transform q back to c using inverse logit
        probabilities = c + (1 - c) / (1 + np.exp(-alpha * (self._person_param - self._difficulty_params)))
        return probabilities

    def negative_log_likelihood(self, parameters):
        probabilities = self.predict(parameters)
        log_likelihood = 0
        for p, sdt in zip(probabilities, self.experiment.conditions):
            n_correct = sdt.hits + sdt.correctRejections
            n_total = sdt.hits + sdt.misses + sdt.falseAlarms + sdt.correctRejections
            log_likelihood += n_correct * np.log(p) + (n_total - n_correct) * np.log(1 - p)
        return -log_likelihood

    def fit(self):
        initial_guess = [1.0, 0.0]  # Initial guess for alpha and q
        bounds = [(0, None), (None, None)]  # alpha > 0, q unbounded
        result = minimize(self.negative_log_likelihood, initial_guess, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            self._discrimination, self._logit_base_rate = result.x
            self._base_rate = expit(self._logit_base_rate)
            self._is_fitted = True
            return result
        else:
            raise ValueError("Optimization failed to converge.")

    def get_discrimination(self):
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self._discrimination

    def get_base_rate(self):
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self._base_rate

    def get_logit_base_rate(self):
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self._logit_base_rate

    def set_discrimination(self, value):
        if value < 0:
            raise ValueError("Discrimination parameter (alpha) must be non-negative.")
        self._discrimination = value
        self._is_fitted = False

    def set_base_rate(self, value):
        if not 0 < value < 1:
            raise ValueError("Base rate (c) must be between 0 and 1 (exclusive).")
        self._base_rate = value
        self._logit_base_rate = logit(value)
        self._is_fitted = False

    def set_logit_base_rate(self, value):
        self._logit_base_rate = value
        self._base_rate = expit(value)
        self._is_fitted = False

    def is_fitted(self):
        return self._is_fitted

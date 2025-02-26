# code base from UCI ZotGPT - prompted with original  equations and asked for code for each
# changes made by group to handle edge cases
import numpy as np
from scipy.stats import norm

class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        # Handles non-negative integers and decimals
        if not all(isinstance(x, (int, float)) and x >= 0 and np.isfinite(x)
                   for x in [hits, misses, falseAlarms, correctRejections]):
            raise ValueError("Inputs must be non-negative finite integers.")
        
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections

    def hit_rate(self):
        return self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.5

    def false_alarm_rate(self):
        return self.falseAlarms / (self.falseAlarms + self.correctRejections) if (self.falseAlarms + self.correctRejections) > 0 else 0.5

    def _adjusted_rate(self, rate):
        return max(0.01, min(rate, 0.99))

    def d_prime(self):
        hit_rate = self._adjusted_rate(self.hit_rate())
        fa_rate = self._adjusted_rate(self.false_alarm_rate())
        
        z_hit = norm.ppf(hit_rate)
        z_fa = norm.ppf(fa_rate)
        
        return z_hit - z_fa

    def criterion(self):
        hit_rate = self._adjusted_rate(self.hit_rate())
        fa_rate = self._adjusted_rate(self.false_alarm_rate())
        
        z_hit = norm.ppf(hit_rate)
        z_fa = norm.ppf(fa_rate)
        
        return -0.5 * (z_hit + z_fa)

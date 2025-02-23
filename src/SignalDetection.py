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

    def d_prime(self):
        # Handles hit+miss=0 or falseAlarm+correctRejections=0... Assume 0.5 chance rate
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.5
        fa_rate = self.falseAlarms / (self.falseAlarms + self.correctRejections) if (self.falseAlarms + self.correctRejections) > 0 else 0.5
        
        # Adjust rates if they are 0 or 1
        hit_rate = max(0.01, min(hit_rate, 0.99))
        fa_rate = max(0.01, min(fa_rate, 0.99))
        
        z_hit = norm.ppf(hit_rate)
        z_fa = norm.ppf(fa_rate)
        
        return z_hit - z_fa

    def criterion(self):
        # Handles hit+miss=0 or falseAlarm+correctRejections=0... Assume 0.5 chance rate
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.5
        fa_rate = self.falseAlarms / (self.falseAlarms + self.correctRejections) if (self.falseAlarms + self.correctRejections) > 0 else 0.5
        
        # Adjust rates if they are 0 or 1
        hit_rate = max(0.01, min(hit_rate, 0.99))
        fa_rate = max(0.01, min(fa_rate, 0.99))
        
        z_hit = norm.ppf(hit_rate)
        z_fa = norm.ppf(fa_rate)
        
        return -0.5 * (z_hit + z_fa)

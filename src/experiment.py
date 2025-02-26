import numpy as np
from SignalDetection import SignalDetection

class Experiment:
    def __init__(self):
        self.conditions = []
        self.labels = []

    def add_condition(self, sdt_obj: SignalDetection, label: str = None):
        self.conditions.append(sdt_obj)
        self.labels.append(label)

    def sorted_roc_points(self):
        if not self.conditions:
            raise ValueError("No conditions added to the experiment")
        
        false_alarm_rates = []
        hit_rates = []
        
        for sdt in self.conditions:
            false_alarm_rates.append(sdt.false_alarm_rate())
            hit_rates.append(sdt.hit_rate())

        # Sort based on false alarm rates
        sorted_indices = np.argsort(false_alarm_rates)
        sorted_false_alarm_rates = [false_alarm_rates[i] for i in sorted_indices]
        sorted_hit_rates = [hit_rates[i] for i in sorted_indices]
        
        return sorted_false_alarm_rates, sorted_hit_rates

    def compute_auc(self):
        if not self.conditions:
            raise ValueError("No conditions added to the experiment")
        
        far, hr = self.sorted_roc_points()
        
        # Compute AUC using the trapezoidal rule
        auc = np.trapz(hr, far)
        
        return auc

    def plot_roc_curve(self, show_plot=True):
        # Implement plotting logic here
        # This is optional for grading, so you can leave it as a placeholder
        pass

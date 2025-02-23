#new
#code generated from ChatGPT, still work in progress
import unittest
from src.experiment import experiment
from src.SignalDetection import SignalDetection

class TestExperiment(unittest.TestCase):
    
    def setUp(self):
        self.exp = Experiment()

    def test_add_condition(self):
        sdt = SignalDetection(40, 10, 20, 30)
        self.exp.add_condition(sdt, label="Condition A")
        self.assertEqual(len(self.exp.conditions), 1)
        self.assertEqual(self.exp.labels[0], "Condition A")
    
    def test_sorted_roc_points(self):
        self.exp.add_condition(SignalDetection(50, 50, 10, 90), "A")
        self.exp.add_condition(SignalDetection(90, 10, 30, 70), "B")
        
        false_alarm_rates, hit_rates = self.exp.sorted_roc_points()
        
        self.assertEqual(len(false_alarm_rates), 2)
        self.assertEqual(len(hit_rates), 2)
        self.assertTrue(all(x <= y for x, y in zip(false_alarm_rates, false_alarm_rates[1:])))
    
    def test_compute_auc(self):
        self.exp.add_condition(SignalDetection(0, 100, 0, 100), "Low")
        self.exp.add_condition(SignalDetection(100, 0, 100, 0), "High")
        auc = self.exp.compute_auc()
        self.assertAlmostEqual(auc, 0.5, places=2)
    
    def test_compute_auc_perfect(self):
        self.exp.add_condition(SignalDetection(0, 100, 0, 100), "Low")
        self.exp.add_condition(SignalDetection(100, 0, 0, 100), "Mid")
        self.exp.add_condition(SignalDetection(100, 0, 100, 0), "High")
        auc = self.exp.compute_auc()
        self.assertAlmostEqual(auc, 1.0, places=2)
    
    def test_sorted_roc_empty(self):
        with self.assertRaises(ValueError):
            self.exp.sorted_roc_points()
    
    def test_compute_auc_empty(self):
        with self.assertRaises(ValueError):
            self.exp.compute_auc()

if __name__ == "__main__":
    unittest.main()


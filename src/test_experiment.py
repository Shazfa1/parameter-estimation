#new
#code generated from ChatGPT, still work in progress
import unittest
from experiment import Experiment
from SignalDetection import SignalDetection

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
#additional tests for more realistic values 
    def test_multiple_conditions(self):
        self.exp.add_condition(SignalDetection(40, 10, 20, 30), "Easy")
        self.exp.add_condition(SignalDetection(30, 20, 20, 30), "Medium")
        self.exp.add_condition(SignalDetection(25, 25, 20, 30), "Hard")
        auc = self.exp.compute_auc()
        self.assertTrue(0 < auc < 1)
        self.assertEqual(len(self.exp.conditions), 3)

    def test_roc_points_ordering(self):
        self.exp.add_condition(SignalDetection(90, 10, 30, 70), "High")
        self.exp.add_condition(SignalDetection(50, 50, 10, 90), "Low")
        far, hr = self.exp.sorted_roc_points()
        self.assertTrue(far[0] < far[1])
        self.assertTrue(hr[0] < hr[1])

    def test_auc_realistic_data(self):
        self.exp.add_condition(SignalDetection(80, 20, 30, 70), "Condition 1")
        self.exp.add_condition(SignalDetection(60, 40, 40, 60), "Condition 2")
        self.exp.add_condition(SignalDetection(40, 60, 50, 50), "Condition 3")
        auc = self.exp.compute_auc()
        self.assertTrue(0.5 < auc < 1)

    def test_add_condition_no_label(self):
        sdt = SignalDetection(40, 10, 20, 30)
        self.exp.add_condition(sdt)
        self.assertEqual(len(self.exp.conditions), 1)
        self.assertIsNone(self.exp.labels[0])
        
if __name__ == "__main__":
    unittest.main()


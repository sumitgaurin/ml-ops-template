import unittest
import os
import json
from src.components.classification.model_selector import compare_models

class TestModelSelector(unittest.TestCase):

    def setUp(self):
        # Create temporary metrics files for two models
        self.metrics_file_1 = 'metrics_model_1.json'
        self.metrics_file_2 = 'metrics_model_2.json'
        self.output_dir = 'output'

        # Metrics for the first model
        metrics_1 = {
            "model_name": "diabetes_logreg_model",
            "model_version": "1",
            "accuracy": 0.85
        }
        with open(self.metrics_file_1, 'w') as f:
            json.dump(metrics_1, f)

        # Metrics for the second model
        metrics_2 = {
            "model_name": "diabetes_gboost_model",
            "model_version": "1",
            "accuracy": 0.90
        }
        with open(self.metrics_file_2, 'w') as f:
            json.dump(metrics_2, f)

    def test_compare_models(self):
        # Call the model comparison function
        report_file = os.path.join(self.output_dir, 'comparison_report.txt')
        compare_models([self.metrics_file_1, self.metrics_file_2], report_file)

        # Verify that the comparison report is saved
        self.assertTrue(os.path.exists(report_file))

        # Check the best model is gboost (since its accuracy is higher)
        with open(report_file, 'r') as f:
            best_model = f.read().strip()
        print(best_model)
        self.assertIn('diabetes_gboost_model', best_model)

    def tearDown(self):
        # Clean up the test environment
        if os.path.exists(self.metrics_file_1):
            os.remove(self.metrics_file_1)
        if os.path.exists(self.metrics_file_2):
            os.remove(self.metrics_file_2)
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))
            os.rmdir(self.output_dir)

if __name__ == '__main__':
    unittest.main()

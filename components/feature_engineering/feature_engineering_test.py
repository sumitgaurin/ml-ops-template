import unittest
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from feature_engineering import feature_engineering

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        # Create a temporary dataset for testing
        self.dataset_name = 'test_dataset.csv'
        self.output_dir = 'output'
        df = pd.DataFrame({
            'numeric_col': [1.0, 2.0, 3.0],
            'categorical_col': ['A', 'B', 'A']
        })
        df.to_csv(self.dataset_name, index=False)

    def test_feature_engineering(self):
        # Call the feature engineering function
        feature_engineering(self.dataset_name, self.output_dir)

        # Verify that the transformed dataset is saved
        # Write your custom code here

    def tearDown(self):
        # Clean up the test environment
        if os.path.exists(self.dataset_name):
            os.remove(self.dataset_name)
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))
            os.rmdir(self.output_dir)

if __name__ == '__main__':
    unittest.main()

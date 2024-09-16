import unittest
import os
import pandas as pd
from split_data import split_dataset

class TestDatasetSplitting(unittest.TestCase):

    def setUp(self):
        # Create a temporary dataset for testing
        self.dataset_path = 'test_data.csv'
        self.train_output_dir = 'train_output'
        self.test_output_dir = 'test_output'
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'label': [0, 1, 0, 1, 0]
        })
        df.to_csv(self.dataset_path, index=False)

    def test_split_dataset(self):
        # Call the dataset splitting function
        split_dataset(self.dataset_path, self.train_output_dir, self.test_output_dir, split_ratio=0.6)

        # Verify that train and test datasets are saved
        train_file = os.path.join(self.train_output_dir, 'train_data.csv')
        test_file = os.path.join(self.test_output_dir, 'test_data.csv')
        self.assertTrue(os.path.exists(train_file))
        self.assertTrue(os.path.exists(test_file))

        # Verify the content of the output files
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        self.assertEqual(train_df.shape[0], 3)  # 60% of 5 rows
        self.assertEqual(test_df.shape[0], 2)   # 40% of 5 rows

    def tearDown(self):
        # Clean up the test environment
        if os.path.exists(self.dataset_path):
            os.remove(self.dataset_path)
        if os.path.exists(self.train_output_dir):
            for file in os.listdir(self.train_output_dir):
                os.remove(os.path.join(self.train_output_dir, file))
            os.rmdir(self.train_output_dir)
        if os.path.exists(self.test_output_dir):
            for file in os.listdir(self.test_output_dir):
                os.remove(os.path.join(self.test_output_dir, file))
            os.rmdir(self.test_output_dir)

if __name__ == '__main__':
    unittest.main()

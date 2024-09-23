import unittest
import os
import pandas as pd
from unittest import mock
from src.components.training.train_model import train_model_paynet

class TestTrainModelPaynet(unittest.TestCase):

    @mock.patch('src.components.training.train_model.mlflow.start_run')
    @mock.patch('src.components.training.train_model.mlflow.sklearn.autolog')
    @mock.patch('src.components.training.train_model.mlflow.sklearn.save_model')
    @mock.patch('src.components.training.train_model.glob.glob')
    @mock.patch('src.components.training.train_model.pd.read_csv')
    def test_train_model_paynet(self, mock_read_csv, mock_glob, mock_save_model, mock_autolog, mock_start_run):
        # Mock the CSV files
        mock_glob.return_value = ['file1.csv', 'file2.csv']
        
        # Mock the dataframes returned by pd.read_csv
        mock_read_csv.side_effect = [
            pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'outcome': [0, 1]}),
            pd.DataFrame({'feature1': [5, 6], 'feature2': [7, 8], 'outcome': [1, 0]})
        ]
        
        # Define the arguments
        training_data_path = 'dummy_path'
        outcome_label = 'outcome'
        output_model_path = 'dummy_output_path'
        n_estimators = 100
        learning_rate = 0.1
        
        # Call the function
        train_model_paynet(training_data_path, outcome_label, output_model_path, n_estimators, learning_rate)
        
        # Assertions
        mock_glob.assert_called_once_with(os.path.join(training_data_path, '*.csv'))
        # assert that mock_read_csv is called twice
        self.assertEqual(mock_read_csv.call_count, 2)
        # assert that mock_autolog is called once
        mock_autolog.assert_called_once()
        # assert that mock_save_model is called once
        mock_save_model.assert_called_once()
        # assert that mock_start_run is called more than once
        self.assertGreater(mock_start_run.call_count, 1)

    @mock.patch('src.components.training.train_model.mlflow.start_run')
    @mock.patch('src.components.training.train_model.mlflow.sklearn.autolog')
    @mock.patch('src.components.training.train_model.mlflow.sklearn.save_model')
    @mock.patch('src.components.training.train_model.glob.glob')
    @mock.patch('src.components.training.train_model.pd.read_csv')
    def test_train_model_paynet_no_csv_files(self, mock_read_csv, mock_glob, mock_save_model, mock_autolog, mock_start_run):
        # Mock no CSV files
        mock_glob.return_value = []
        
        # Define the arguments
        training_data_path = 'dummy_path'
        outcome_label = 'outcome'
        output_model_path = 'dummy_output_path'
        n_estimators = 100
        learning_rate = 0.1
        
        # Call the function
        with self.assertRaises(ValueError):
            train_model_paynet(training_data_path, outcome_label, output_model_path, n_estimators, learning_rate)
        
        # Assertions
        mock_glob.assert_called_once_with(os.path.join(training_data_path, '*.csv'))
        # assert that mock_read_csv is not called
        mock_read_csv.assert_not_called()
        # assert that mock_autolog is not called
        mock_autolog.assert_called_once()
        # assert that mock_save_model is not called
        mock_save_model.assert_not_called()
        # assert that mock_start_run is not called
        mock_start_run.assert_called_once()

if __name__ == '__main__':
    unittest.main()

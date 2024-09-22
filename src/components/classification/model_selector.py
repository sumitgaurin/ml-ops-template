import argparse
import json
import os
import mlflow
import pandas as pd

def compare_models(metrics_file_paths, constraint, output_path):
    # Start Logging with mlflow using context manager
    with mlflow.start_run():
        # Load metrics from the provided JSON files
        metrics = {}
        for file_path in metrics_file_paths:
            with open(file_path, 'r') as f:
                metrics[file_path] = json.load(f)

        # Convert the metrics dictionary to a DataFrame for comparison
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.reset_index(level=0, inplace=True)
        metrics_df['f1_score'] = metrics_df['f1_score'].astype(float)
        metrics_df['fpr'] = metrics_df['fpr'].astype(float)
        metrics_df['fnr'] = metrics_df['fnr'].astype(float)

        print("Metrics DataFrame:\n", metrics_df.info())
        
        # Find the best model by constraint
        if constraint == 'minimize_fp':
            best_model_row = metrics_df.loc[metrics_df['fpr'].idxmin()]
        elif constraint == 'minimize_fn':
            best_model_row = metrics_df.loc[metrics_df['fnr'].idxmin()]
        else:
            best_model_row = metrics_df.loc[metrics_df['f1_score'].idxmax()]

        # Get the model name
        best_model = best_model_row['model_id']

        # Generate a comparison report
        # Convert the dataframe to a list of dictionaries for the models
        models_list = metrics_df.to_dict(orient='records')

        # Prepare the final dictionary
        final_output = {
            "models": models_list,
            "best_model_id": best_model
        }

        # Dump the final dictionary to a JSON string (or to a file)
        json_output = json.dumps(final_output, indent=4)

        # Save the comparison report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as report_file:
            report_file.write(json_output)
        print(f"Comparison report saved at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1_report_path', type=str, help='Path to the evaluation result file of first model')
    parser.add_argument('--model2_report_path', type=str, help='Path to the evaluation result file of second model')
    parser.add_argument('--constraint', type=str, help='The criteria on which the best model selection is done')
    parser.add_argument('--comparison_report', type=str, help='File to save the comparison report and best model')

    args = parser.parse_args()
    print('Printing received arguments...')
    for arg_name in vars(args):
        print(f"{arg_name}: {getattr(args, arg_name)}")

    report_files = [args.model1_report_path, args.model2_report_path]
    compare_models(report_files, args.constraint, args.comparison_report)

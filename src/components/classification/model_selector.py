import argparse
import json
import os
import pandas as pd

def compare_models(metrics_file_paths, output_path):
    # Load metrics from the provided JSON files
    metrics = {}
    for file_path in metrics_file_paths:
        with open(file_path, 'r') as f:
            metrics[file_path] = json.load(f)

    # Convert the metrics dictionary to a DataFrame for comparison
    metrics_df = pd.DataFrame(metrics).T
    print("Metrics DataFrame:\n", metrics_df)

    # Find the model with the highest accuracy
    best_model_row = metrics_df.loc[metrics_df['accuracy'].idxmax()]
    # Get the model name
    best_model = best_model_row['model_name']

    # Generate a comparison report
    # Convert the dataframe to a list of dictionaries for the models
    models_list = metrics_df.to_dict(orient='records')

    # Prepare the final dictionary
    final_output = {
        "models": models_list,
        "best_model": best_model
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
    parser.add_argument('--metrics-files', type=str, nargs='+', help='List of paths to the metrics JSON files for comparison')
    parser.add_argument('--output-path', type=str, help='File to save the comparison report and best model')

    args = parser.parse_args()
    compare_models(args.metrics_files, args.output_path)

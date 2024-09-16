import argparse
import json
import os
import pandas as pd

def compare_models(metrics_file_paths, output_dir):
    # Load metrics from the provided JSON files
    metrics = {}
    for file_path in metrics_file_paths:
        with open(file_path, 'r') as f:
            metrics[file_path] = json.load(f)

    # Convert the metrics dictionary to a DataFrame for comparison
    metrics_df = pd.DataFrame(metrics).T
    print("Metrics DataFrame:\n", metrics_df)

    # Find the model with the highest accuracy
    best_model = metrics_df['accuracy'].idxmax()

    # Generate a comparison report
    comparison_report = f"Model Comparison Report:\n\n{metrics_df.to_string()}\n\nBest Model: {best_model}"

    # Save the comparison report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    with open(report_path, 'w') as report_file:
        report_file.write(comparison_report)
    print(f"Comparison report saved at {report_path}")

    # Return the best model as an output
    best_model_output_path = os.path.join(output_dir, 'best_model.txt')
    with open(best_model_output_path, 'w') as best_model_file:
        best_model_file.write(best_model)
    print(f"Best model saved at {best_model_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics-files', type=str, nargs='+', help='List of paths to the metrics JSON files for comparison')
    parser.add_argument('--output-dir', type=str, help='Directory to save the comparison report and best model')

    args = parser.parse_args()
    compare_models(args.metrics_files, args.output_dir)

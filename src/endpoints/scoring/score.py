import json
import os
import numpy as np
import joblib

# Global variables
model = None
model_name = None
model_version = None

# This init() function is called once when the container is started
def init():
    global model, model_name, model_version

    # Retrieve the model directory from the environment variable
    model_dir = os.environ.get("AZUREML_MODEL_DIR")

    # Construct the path to the model file
    model_path = os.path.join(model_dir, "model", "model.pkl")

    # Extract the model name and version from the directory structure
    # Format of AZUREML_MODEL_DIR: ./azureml-models/$MODEL_NAME/$VERSION
    if model_dir:
        path_parts = model_dir.split(os.sep)
        model_name = path_parts[-2]  # Second last folder is the model name
        model_version = path_parts[-1]  # Last folder is the model version

    # Load the model
    model = joblib.load(model_path)

# This run() function is called each time a request is made to the scoring endpoint
def run(raw_data):
    try:
        # Parse the input JSON
        data = json.loads(raw_data)

        # Define the expected feature order based on the model signature
        feature_order = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

        # Extract features from the input JSON and convert to the correct order
        features = [data[feature_name] for feature_name in feature_order]

        # Convert to a NumPy array and reshape for model input
        features_array = np.array([features], dtype=float)

        # Predict probabilities for the input features
        probabilities = model.predict_proba(features_array)[0]

        # Determine the class with the highest probability
        class_idx = np.argmax(probabilities)
        class_probability = probabilities[class_idx]

        # Define class labels (adjust based on your actual class labels)
        class_labels = [
            "No Diabetes", "Diabetes"
        ]

        # Create response JSON with the highest probability class, its probability, model name, and version
        result = {
            "predicted_class": class_labels[class_idx],
            "probability": class_probability,
            "model_name": model_name,
            "model_version": model_version
        }

        return json.dumps(result)

    except KeyError as e:
        # Handle missing input fields
        return json.dumps({"error": f"Missing input field: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

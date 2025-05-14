import tensorflow as tf
import numpy as np
import os
from nforest_model import NeuralForest, custom_objects_for_loading # Import the model and custom objects
from sklearn.datasets import make_classification # For generating dummy prediction data

# Set random seed for reproducibility (optional, for predictable dummy data)
np.random.seed(43) # Use a different seed than training if you like

# --- Configuration ---
# Need the save path to load the model
config = {
    'model_save_path': './saved_neural_forest',
    # You might need other config details here depending on your prediction logic
    # e.g., the type of task (sigmoid, softmax, linear)
}

# --- Load the Trained Forest ---
print(f"--- Loading Neural Forest from {config['model_save_path']} ---")

try:
    loaded_forest = NeuralForest.load(config['model_save_path'])
    print("Forest loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading forest: {e}")
    print("Please ensure the training script has been run and the model saved.")
    exit() # Exit if model loading fails
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    exit()


# --- Generate Dummy Data for Prediction (Replace with your new data) ---
# Assuming the model was trained on data with the same number of features
# In a real scenario, load your new data here.
print("\n--- Generating Dummy Data for Prediction ---")
# To generate dummy data with the correct number of features, we might need
# to know the input_dim from the loaded model.
# Accessing input_dim from the loaded forest:
input_dim_for_prediction = loaded_forest.input_dim

# Generate some dummy data with the correct input dimension
X_new, _ = make_classification(n_samples=10, n_features=input_dim_for_prediction, random_state=43)
print(f"Generated dummy data with shape: {X_new.shape}")


# --- Make Predictions ---
print("\n--- Making Predictions ---")

# The task type is determined by the loaded forest's leaf_nn_config
output_activation = loaded_forest.leaf_nn_params['output_activation']

if output_activation in ['sigmoid', 'softmax']:
    # For classification, get probabilities and potentially labels
    y_pred_proba = loaded_forest.predict_proba(X_new)
    print("\nPredicted Probabilities/Values:")
    print(y_pred_proba)

    y_pred_labels = loaded_forest.predict(X_new)
    print("\nPredicted Labels:")
    print(y_pred_labels)

elif output_activation == 'linear':
    # For regression, predict returns the values directly
    y_pred_values = loaded_forest.predict(X_new)
    print("\nPredicted Values:")
    print(y_pred_values)

else:
    print(f"Unsupported output activation for prediction display: {output_activation}")


print("\nPrediction complete.")
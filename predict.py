import tensorflow as tf
import numpy as np
import os
from sklearn.datasets import make_classification, make_regression # For dummy data

# Import from the installed or local package
from neural_forest.model import NeuralForest
# If needed for direct loading (though NeuralForest.load handles it):
# from neural_forest.model import custom_objects_for_keras_load_model

# Set random seed for reproducibility of dummy data
PREDICT_SEED = 123
np.random.seed(PREDICT_SEED)
tf.random.set_seed(PREDICT_SEED)

# --- Configuration for Prediction ---
# This should match the configuration used during training for model path
# For this example, let's assume the regression model from train_example.py was saved
# You'll need to adjust 'model_artifact_version' and 'leaf_nn_output_activation'
# based on which trained model you want to load.
predict_config = {
    'base_model_save_dir': './saved_neural_forest_models',
    'model_artifact_version': '1.0.0',
    'leaf_nn_output_activation_of_saved_model': 'linear', # MUST MATCH THE SAVED MODEL
     # 'leaf_nn_output_activation_of_saved_model': 'sigmoid', # Example if loading a sigmoid model
}

# Construct the path to the saved model artifact
model_to_load_path = os.path.join(
    predict_config['base_model_save_dir'],
    f"forest_v{predict_config['model_artifact_version']}_{predict_config['leaf_nn_output_activation_of_saved_model']}"
)

# --- Load the Trained Forest ---
print(f"--- Loading Neural Forest from {model_to_load_path} ---")

if not os.path.exists(model_to_load_path):
    print(f"Error: Model path not found: {model_to_load_path}")
    print("Please ensure the training script (train_example.py) has been run and the model saved,")
    print("or that the 'predict_config' variables match a saved model.")
    exit()

try:
    # The NeuralForest.load method handles custom objects internally
    loaded_neural_forest = NeuralForest.load(model_to_load_path)
    print("Neural Forest loaded successfully.")
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    print("Ensure that the model was saved correctly and TensorFlow versions are compatible.")
    exit()


# --- Generate Dummy Data for Prediction (Replace with your new data) ---
# The number of features for dummy data should match the model's input_dim.
# This is stored in the loaded forest's config.
input_dim_from_loaded_model = loaded_neural_forest.input_dim
n_samples_predict = 10
print(f"\n--- Generating {n_samples_predict} Dummy Samples for Prediction (Features: {input_dim_from_loaded_model}) ---")

# Generate data appropriate for the model's task type (inferred from its output activation)
model_output_activation = loaded_neural_forest.leaf_nn_params.get('output_activation', '').lower()

if model_output_activation == 'linear': # Regression model
    X_new, _ = make_regression(
        n_samples=n_samples_predict, n_features=input_dim_from_loaded_model,
        n_informative=max(5, input_dim_from_loaded_model // 2), # Ensure n_informative <= n_features
        noise=0.1, random_state=PREDICT_SEED
    )
    print("Generated regression-like dummy data.")
else: # Classification model (sigmoid or softmax)
    # For make_classification, n_classes is needed if inferring from softmax output_dim
    # This is a simplified example; real data would match the original training distribution.
    n_classes_dummy = 2
    if model_output_activation == 'softmax':
        # Try to get output_dim from loaded model if it's softmax
        output_dim_dummy = loaded_neural_forest.leaf_nn_params.get('output_dim', 2)
        n_classes_dummy = output_dim_dummy if output_dim_dummy > 1 else 2

    X_new, _ = make_classification(
        n_samples=n_samples_predict, n_features=input_dim_from_loaded_model,
        n_informative=max(5, input_dim_from_loaded_model // 2), # Ensure n_informative <= n_features
        n_redundant=0, n_classes=n_classes_dummy, random_state=PREDICT_SEED
    )
    print("Generated classification-like dummy data.")

print(f"Dummy data X_new shape: {X_new.shape}")


# --- Make Predictions ---
print("\n--- Making Predictions ---")

# For classification, predict_proba gives probabilities, predict gives class labels.
# For regression, predict_proba and predict both give the predicted values.

if model_output_activation in ['sigmoid', 'softmax']:
    # Get probabilities
    y_pred_proba = loaded_neural_forest.predict_proba(X_new)
    print("\nPredicted Probabilities:")
    print(y_pred_proba)

    # Get final class labels
    y_pred_labels = loaded_neural_forest.predict(X_new)
    print("\nPredicted Labels:")
    print(y_pred_labels)

elif model_output_activation == 'linear':
    # For regression, predict() or predict_proba() will give the values
    y_pred_values = loaded_neural_forest.predict(X_new) # or .predict_proba(X_new)
    print("\nPredicted Values (Regression):")
    print(y_pred_values)
else:
    print(f"Warning: Model has an unrecognized output_activation '{model_output_activation}'. Displaying raw predict_proba output.")
    y_pred_raw = loaded_neural_forest.predict_proba(X_new)
    print("\nRaw Predictions (unknown task type):")
    print(y_pred_raw)

print("\nPrediction script complete.")
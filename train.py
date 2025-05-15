import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression # Added make_regression
from tensorflow import keras # For keras.utils.to_categorical
import os

# Import from the installed or local package
from neural_forest.model import NeuralForest
# from neural_forest import __version__ # Example of importing version

# print(f"Using Neural Forest version: {__version__}") # Optional

# Set random seed for reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

# --- Configuration ---
config = {
    'model_artifact_version': '1.0.0', # Version for the saved model artifact

    # Tree Parameters
    'tree_depth': 3, # Depth of each tree
    'leaf_nn_hidden_layers': [16, 8], # Hidden layers for leaf networks
    'leaf_nn_activation': 'relu', # Activation for hidden layers in leaf NNs
    # 'leaf_nn_output_activation': 'sigmoid', # 'sigmoid'(binary), 'softmax'(multi-class), 'linear'(regression)
    'leaf_nn_output_activation': 'linear', # CHANGED FOR REGRESSION EXAMPLE

    # Forest Parameters
    'num_trees_in_forest': 5, # Number of trees in the forest
    'epochs_per_tree': 15, # Epochs to train each tree
    'batch_size_per_tree': 32, # Batch size for training each tree

    # Training Control
    'validation_split_per_tree': 0.15,
    'early_stopping_patience': 3, # Set to 0 to disable early stopping
    'random_state_for_bagging': GLOBAL_SEED, # Base seed for bagging in forest.fit

    # Verbosity
    'forest_verbose': 1, # 0 for silent, 1 for progress
    'tree_training_verbose': 0, # Keras verbosity (0=silent, 1=progress bar, 2=one line per epoch)

    # Saving
    'base_model_save_dir': './saved_neural_forest_models', # Base directory for saved models
    'task_type': 'regression' # 'classification' or 'regression' - for data generation
}

# Construct the versioned model save path for this specific trained artifact
versioned_model_save_path = os.path.join(
    config['base_model_save_dir'],
    f"forest_v{config['model_artifact_version']}_{config['leaf_nn_output_activation']}"
)
config['model_save_path'] = versioned_model_save_path


# --- Data Generation ---
print("--- Generating Example Data ---")
n_samples = 1000
n_features = 15
input_dim = n_features # Will be set from X.shape[1] later

if config['task_type'] == 'classification':
    n_classes_data = 2
    if config['leaf_nn_output_activation'] == 'softmax':
        n_classes_data = 3 # Example for multi-class
    X, y_original = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=10,
        n_redundant=2, n_classes=n_classes_data, random_state=GLOBAL_SEED
    )
    if config['leaf_nn_output_activation'] == 'sigmoid':
        y = y_original.reshape(-1, 1).astype(np.float32)
        leaf_nn_output_dim = 1
    elif config['leaf_nn_output_activation'] == 'softmax':
        y = keras.utils.to_categorical(y_original, num_classes=n_classes_data).astype(np.float32)
        leaf_nn_output_dim = n_classes_data
    else: # linear for classification (not typical)
        print("Warning: Linear output for classification task type. Treating as regression for y-prep.")
        y = y_original.astype(np.float32).reshape(-1, 1)
        leaf_nn_output_dim = 1

elif config['task_type'] == 'regression':
    if config['leaf_nn_output_activation'] != 'linear':
        print(f"Warning: Task type is regression, but output activation is {config['leaf_nn_output_activation']}. Consider 'linear'.")
    X, y_original = make_regression(
        n_samples=n_samples, n_features=n_features, n_informative=10,
        noise=0.5, random_state=GLOBAL_SEED
    )
    y = y_original.astype(np.float32).reshape(-1, 1)
    leaf_nn_output_dim = 1
else:
    raise ValueError(f"Unsupported task_type: {config['task_type']}")

input_dim = X.shape[1] # Correctly set input_dim from data

print(f"Data generated: X shape {X.shape}, y shape {y.shape}")
print(f"Output activation: {config['leaf_nn_output_activation']}, Leaf NN output_dim: {leaf_nn_output_dim}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=GLOBAL_SEED)

# Define parameters for each tree's leaf networks
leaf_nn_parameters_config = {
    'hidden_layers': config['leaf_nn_hidden_layers'],
    'activation': config['leaf_nn_activation'],
    'output_dim': leaf_nn_output_dim,
    'output_activation': config['leaf_nn_output_activation']
}

# --- Create and Train the Neural Forest ---
print("\n--- Neural Forest Initialization ---")
neural_forest_model = NeuralForest(
    num_trees=config['num_trees_in_forest'],
    tree_depth=config['tree_depth'],
    input_dim=input_dim,
    leaf_nn_config=leaf_nn_parameters_config
)

print("\n--- Neural Forest Training ---")
neural_forest_model.fit(
    X_train,
    y_train,
    epochs=config['epochs_per_tree'],
    batch_size_per_tree=config['batch_size_per_tree'],
    validation_split_per_tree=config['validation_split_per_tree'],
    early_stopping_patience=config['early_stopping_patience'],
    forest_verbose=config['forest_verbose'],
    tree_training_verbose=config['tree_training_verbose'],
    random_state_base=config['random_state_for_bagging']
)

# --- Save the Trained Forest ---
print(f"\n--- Saving Neural Forest to {config['model_save_path']} ---")
os.makedirs(config['model_save_path'], exist_ok=True)
neural_forest_model.save(config['model_save_path'])

print("\nTraining and saving complete.")

# --- Optional: Evaluate on Test Set ---
print("\n--- Evaluation on Test Set ---")
# For classification, predict_proba gives probabilities, predict gives labels
# For regression, predict_proba (and predict) gives the predicted values
y_pred_values_or_proba = neural_forest_model.predict_proba(X_test)

if config['leaf_nn_output_activation'] == 'sigmoid':
    from sklearn.metrics import accuracy_score, roc_auc_score
    y_pred_labels = (y_pred_values_or_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_labels)
    roc_auc = roc_auc_score(y_test, y_pred_values_or_proba) # Use probabilities for AUC
    print(f"Forest Test Accuracy: {accuracy:.4f}")
    print(f"Forest Test ROC AUC: {roc_auc:.4f}")

elif config['leaf_nn_output_activation'] == 'softmax':
    from sklearn.metrics import accuracy_score # roc_auc_score needs careful setup for multi-class
    y_pred_labels = np.argmax(y_pred_values_or_proba, axis=1)
    y_test_labels = np.argmax(y_test, axis=1) # Assuming y_test is one-hot
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    print(f"Forest Test Accuracy: {accuracy:.4f}")
    # For multi-class AUC, consider metrics like One-vs-Rest or One-vs-One if needed

elif config['leaf_nn_output_activation'] == 'linear': # Regression
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    mse = mean_squared_error(y_test, y_pred_values_or_proba)
    mae = mean_absolute_error(y_test, y_pred_values_or_proba)
    r2 = r2_score(y_test, y_pred_values_or_proba)
    print(f"Forest Test Mean Squared Error (MSE): {mse:.4f}")
    print(f"Forest Test Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Forest Test R-squared (R2): {r2:.4f}")
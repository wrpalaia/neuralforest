import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from nforest_model import NeuralForest # Import your model
import os # Import os for path manipulation

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
# Use a dictionary for configuration
config = {
    # Model Versioning
    'model_version': '1.0.0', # <-- Add a semantic version here

    # Common Tree Parameters
    'tree_depth': 3,
    'leaf_nn_hidden_layers': [8],
    'leaf_nn_activation': 'relu',
    'leaf_nn_output_activation': 'sigmoid', # 'sigmoid', 'softmax', or 'linear'

    # Forest Parameters
    'num_trees_in_forest': 10,
    'epochs_per_tree': 20,
    'batch_size': 64,

    # Training Parameters
    'validation_split_per_tree': 0.1,
    'early_stopping_patience': 5,
    'forest_verbose': 1, # 1 for progress, 0 for silent
    'tree_training_verbose': 0, # Keras verbosity (0 for silent, 1 for progress bar)

    # Base Model Saving Directory (Version will be added to this)
    'base_model_save_dir': './saved_neural_forest' # <-- Change to a base directory
}

# Construct the versioned model save path
# Example: ./saved_neural_forest/1.0.0
versioned_model_save_path = os.path.join(
    config['base_model_save_dir'],
    config['model_version']
)
config['model_save_path'] = versioned_model_save_path # Update config with the full path


# --- Data Generation (Example - Replace with your data loading) ---
print("--- Generating Example Data ---")
# Adjust n_classes based on leaf_nn_output_activation if needed
n_classes = 2
if config['leaf_nn_output_activation'] == 'softmax':
    n_classes = 3 # Example for multi-class
    output_dim = n_classes
elif config['leaf_nn_output_activation'] == 'sigmoid':
     output_dim = 1
elif config['leaf_nn_output_activation'] == 'linear':
     # For regression, make_classification generates integer classes,
     # you might want a different data source or transform y.
     # For demonstration, we'll treat it as a binary classification dataset initially.
     # If you intended regression, replace make_classification with a regression dataset generator.
     print("Warning: Using make_classification for linear output. Consider a regression dataset.")
     output_dim = 1


X, y_original = make_classification(n_samples=2000, n_features=20, n_informative=15, n_redundant=5, n_classes=n_classes, random_state=42)
input_dim = X.shape[1]

# Prepare y based on the task
if config['leaf_nn_output_activation'] == 'sigmoid': # Binary classification
    y = y_original.reshape(-1, 1)
elif config['leaf_nn_output_activation'] == 'softmax': # Multi-class
    y = keras.utils.to_categorical(y_original, num_classes=n_classes)
    # Ensure leaf_nn_config['output_dim'] matches number of classes
    # This override happens outside the config dict for clarity before passing
    leaf_nn_output_dim = n_classes
elif config['leaf_nn_output_activation'] == 'linear': # Regression
    # If using make_classification for linear, y is 0 or 1.
    # You'd typically load or generate continuous y values for regression.
    y = y_original.astype(np.float32).reshape(-1, 1) # Convert to float for regression
    leaf_nn_output_dim = 1
else:
     raise ValueError(f"Unsupported leaf_nn_output_activation: {config['leaf_nn_output_activation']}")

# Determine leaf network output dimension based on task
if config['leaf_nn_output_activation'] in ['sigmoid', 'linear']:
    leaf_nn_output_dim = 1
elif config['leaf_nn_output_activation'] == 'softmax':
    leaf_nn_output_dim = y.shape[1] # Number of columns in one-hot encoded y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Define parameters for each tree's leaf networks based on config and data
leaf_nn_parameters_config = {
    'hidden_layers': config['leaf_nn_hidden_layers'],
    'activation': config['leaf_nn_activation'],
    'output_dim': leaf_nn_output_dim, # Use calculated output_dim
    'output_activation': config['leaf_nn_output_activation']
}

# --- Create and Train the Neural Forest ---
print("\n--- Neural Forest Initialization ---")
neural_forest = NeuralForest(
    num_trees=config['num_trees_in_forest'],
    tree_depth_val=config['tree_depth'],
    input_dim=input_dim, # Input dimension from your data
    leaf_nn_config=leaf_nn_parameters_config
)

print("\n--- Neural Forest Training ---")
neural_forest.fit(
    X_train,
    y_train,
    epochs=config['epochs_per_tree'],
    tree_batch_size=config['batch_size'],
    validation_split_per_tree=config['validation_split_per_tree'],
    early_stopping_patience=config['early_stopping_patience'],
    forest_verbose=config['forest_verbose'],
    tree_training_verbose=config['tree_training_verbose']
)

# --- Save the Trained Forest ---
print(f"\n--- Saving Neural Forest to {config['model_save_path']} ---")
# Ensure the versioned directory exists before saving
os.makedirs(config['model_save_path'], exist_ok=True) # <-- Create the versioned directory
neural_forest.save(config['model_save_path'])

print("\nTraining and saving complete.")

# Optional: Evaluate on test set after training (for verification)
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

print("\n--- Evaluation on Test Set ---")
y_pred_proba_forest = neural_forest.predict_proba(X_test)

if config['leaf_nn_output_activation'] == 'sigmoid': # Binary classification
    y_pred_labels_forest = (y_pred_proba_forest > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_labels_forest)
    try:
        # Ensure y_test is the correct shape for roc_auc_score if it's not already
        if y_test.ndim > 1 and y_test.shape[1] == 1:
             y_test_roc_auc = y_test.flatten()
        else:
             y_test_roc_auc = y_test # Assume it's already 1D for binary

        # Ensure y_pred_proba_forest is also 1D for binary ROC AUC
        if y_pred_proba_forest.ndim > 1 and y_pred_proba_forest.shape[1] == 1:
            y_pred_proba_forest_roc_auc = y_pred_proba_forest.flatten()
        else:
            y_pred_proba_forest_roc_auc = y_pred_proba_forest # Assume it's already 1D

        roc_auc = roc_auc_score(y_test_roc_auc, y_pred_proba_forest_roc_auc)
        print(f"Forest Test Accuracy: {accuracy:.4f}")
        print(f"Forest Test ROC AUC: {roc_auc:.4f}")
    except ValueError as e:
        print(f"Forest Test Accuracy: {accuracy:.4f}")
        print(f"Could not calculate ROC AUC: {e}")

elif config['leaf_nn_output_activation'] == 'softmax': # Multi-class classification
    y_pred_labels_forest = np.argmax(y_pred_proba_forest, axis=1)
    y_test_labels = np.argmax(y_test, axis=1) # Assuming y_test is one-hot encoded
    accuracy = accuracy_score(y_test_labels, y_pred_labels_forest)
    print(f"Forest Test Accuracy: {accuracy:.4f}")

elif config['leaf_nn_output_activation'] == 'linear': # Regression
    mse = mean_squared_error(y_test, y_pred_proba_forest)
    r2 = r2_score(y_test, y_pred_proba_forest)
    print(f"Forest Test Mean Squared Error: {mse:.4f}")
    print(f"Forest Test R-squared: {r2:.4f}")
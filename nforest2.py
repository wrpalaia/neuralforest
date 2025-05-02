import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
# Define the number of neural trees in the forest
n_trees = 10

# Define the parameters for the Shallow MLP neural trees
input_dim = None # To be determined from the data
hidden_layers = [32, 16] # Number of neurons in each hidden layer
activation = 'relu'
output_dim = 1 # For binary classification or regression (adjust for multi-class)
output_activation = 'sigmoid' # Use 'linear' for regression, 'softmax' for multi-class
loss_function = 'binary_crossentropy' # Use 'mse' for regression, 'categorical_crossentropy' for multi-class
optimizer = 'adam'
epochs_per_tree = 20
batch_size_per_tree = 32

# Bagging and Random Subspace parameters
bootstrap_sample_size = 1.0 # Fraction of original dataset size for bootstrap samples
feature_subset_fraction = 0.8 # Fraction of total features for random subspace

# --- Data Generation (Example) ---
# Create a synthetic dataset for demonstration
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
input_dim = X.shape[1] # Set input dimension based on generated data

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to pandas DataFrame for easier feature subsetting
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)

# --- Neural Tree Model Definition (Shallow MLP) ---
def build_shallow_mlp(input_dim, hidden_layers, output_dim, activation, output_activation):
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))
    for layer_size in hidden_layers:
        model.add(keras.layers.Dense(layer_size, activation=activation))
    model.add(keras.layers.Dense(output_dim, activation=output_activation))
    return model

# --- Neural Forest Implementation ---
class NeuralForest:
    def __init__(self, n_trees, tree_params, bagging_params, feature_params):
        self.n_trees = n_trees
        self.tree_params = tree_params
        self.bagging_params = bagging_params
        self.feature_params = feature_params
        self.forest = []
        self.feature_subsets = []

    def train(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        n_bagging_samples = int(self.bagging_params['bootstrap_sample_size'] * n_samples)
        n_feature_subset = int(self.feature_params['feature_subset_fraction'] * n_features)

        for i in range(self.n_trees):
            print(f"Training Tree {i+1}/{self.n_trees}")

            # Step 1: Data Sampling (Bootstrap Aggregating)
            X_sample, y_sample = resample(X_train, y_train, n_samples=n_bagging_samples, replace=True, random_state=i)
            X_sample_df = pd.DataFrame(X_sample)

            # Step 2: Feature Selection (Random Subspace Method)
            selected_features_indices = np.random.choice(n_features, n_feature_subset, replace=False)
            self.feature_subsets.append(selected_features_indices)
            X_sample_subset = X_sample_df.iloc[:, selected_features_indices]

            # Step 3: Neural Tree Model Definition and Training
            tree_input_dim = X_sample_subset.shape[1]
            neural_tree = build_shallow_mlp(
                input_dim=tree_input_dim,
                hidden_layers=self.tree_params['hidden_layers'],
                output_dim=self.tree_params['output_dim'],
                activation=self.tree_params['activation'],
                output_activation=self.tree_params['output_activation']
            )

            neural_tree.compile(
                optimizer=self.tree_params['optimizer'],
                loss=self.tree_params['loss_function'],
                metrics=['accuracy'] # Add other metrics as needed
            )

            # Early stopping to prevent overfitting of individual trees
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            # Use a portion of the bootstrap sample for validation during tree training
            history = neural_tree.fit(
                X_sample_subset,
                y_sample,
                epochs=self.tree_params['epochs_per_tree'],
                batch_size=self.tree_params['batch_size_per_tree'],
                validation_split=0.2, # Using 20% of bootstrap sample for validation
                callbacks=[early_stopping],
                verbose=0 # Set to 1 to see training progress per tree
            )
            self.forest.append(neural_tree)

    def predict_proba(self, X_test):
        n_samples, n_features = X_test.shape
        X_test_df = pd.DataFrame(X_test)
        predictions = []
        for i, neural_tree in enumerate(self.forest):
            # Select the same feature subset used for training this tree
            selected_features_indices = self.feature_subsets[i]
            X_test_subset = X_test_df.iloc[:, selected_features_indices]
            tree_predictions = neural_tree.predict(X_test_subset)
            predictions.append(tree_predictions)

        # Aggregate predictions (averaging probabilities)
        # Stack predictions along a new axis and calculate the mean
        averaged_predictions = np.mean(np.stack(predictions, axis=-1), axis=-1)
        return averaged_predictions

    def predict(self, X_test):
        # For classification, convert probabilities to class labels
        probabilities = self.predict_proba(X_test)
        if self.tree_params['output_activation'] == 'sigmoid': # Binary classification
            return (probabilities > 0.5).astype(int)
        elif self.tree_params['output_activation'] == 'softmax': # Multi-class classification
            return np.argmax(probabilities, axis=1)
        else: # Regression
            return probabilities

# --- Empirical Evaluation ---
if __name__ == "__main__":
    tree_parameters = {
        'hidden_layers': hidden_layers,
        'activation': activation,
        'output_dim': output_dim,
        'output_activation': output_activation,
        'loss_function': loss_function,
        'optimizer': optimizer,
        'epochs_per_tree': epochs_per_tree,
        'batch_size_per_tree': batch_size_per_tree
    }

    bagging_parameters = {
        'bootstrap_sample_size': bootstrap_sample_size
    }

    feature_parameters = {
        'feature_subset_fraction': feature_subset_fraction
    }

    # Create and train the Neural Forest
    neural_forest = NeuralForest(
        n_trees=n_trees,
        tree_params=tree_parameters,
        bagging_params=bagging_parameters,
        feature_params=feature_parameters
    )
    neural_forest.train(X_train, y_train)

    # Make predictions on the test set
    y_pred_proba = neural_forest.predict_proba(X_test)
    y_pred = neural_forest.predict(X_test)

    # Evaluate the Neural Forest (Example for binary classification)
    from sklearn.metrics import accuracy_score, roc_auc_score

    print("\n--- Empirical Evaluation Results ---")
    if output_activation == 'sigmoid':
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
    elif output_activation == 'linear': # Regression
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared: {r2:.4f}")
    # Add evaluation for multi-class classification as needed
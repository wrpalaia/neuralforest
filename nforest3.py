import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
# Define the number of Neural Decision Trees in the forest
n_trees = 5 # Keeping it small for demonstration

# Define the parameters for the Neural Decision Trees
tree_depth = 3 # Depth of each tree (number of splits before reaching leaves)
num_leaves = 2**tree_depth

# Parameters for the small neural networks at the leaves
leaf_nn_hidden_layers = [8] # Number of neurons in hidden layers for leaf NNs
leaf_nn_activation = 'relu'
output_dim = 1 # For binary classification or regression (adjust for multi-class)
leaf_nn_output_activation = 'sigmoid' # Use 'linear' for regression, 'softmax' for multi-class
loss_function = 'binary_crossentropy' # Use 'mse' for regression, 'categorical_crossentropy' for multi-class
optimizer = 'adam'
epochs_per_tree = 50 # More epochs might be needed for end-to-end training
batch_size = 64 # Batch size for training the entire tree

# Bagging and Random Subspace parameters for the forest
bootstrap_sample_size = 0.8 # Fraction of original dataset size for bootstrap samples
feature_subset_fraction = 0.7 # Fraction of total features for random subspace

# --- Data Generation (Example) ---
# Create a synthetic dataset for demonstration
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
input_dim = X.shape[1] # Set input dimension based on generated data

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to pandas DataFrame for easier feature subsetting (optional but good practice)
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)

# --- Neural Decision Tree Components ---

class LeafNetwork(keras.Model):
    """A small neural network at the leaf node."""
    def __init__(self, hidden_layers, output_dim, activation, output_activation):
        super(LeafNetwork, self).__init__()
        self.hidden_layers = []
        for size in hidden_layers:
            self.hidden_layers.append(layers.Dense(size, activation=activation))
        self.output_layer = layers.Dense(output_dim, activation=output_activation)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

class DecisionNode(layers.Layer):
    """A differentiable decision node for routing."""
    def __init__(self, **kwargs):
        super(DecisionNode, self).__init__(**kwargs)
        # This dense layer will learn a linear combination of features
        # followed by a sigmoid to get a routing probability (e.g., probability of going left)
        self.router = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # The output is the probability of going left
        return self.router(inputs)

class NeuralDecisionTree(keras.Model):
    """A single Neural Decision Tree with differentiable routing and leaf networks."""
    def __init__(self, depth, input_dim, leaf_nn_params, **kwargs):
        super(NeuralDecisionTree, self).__init__(**kwargs)
        self.depth = depth
        self.num_leaves = 2**depth
        self.input_dim = input_dim
        self.leaf_nn_params = leaf_nn_params

        # Create decision nodes for each level except the last (leaves)
        self.decision_nodes = []
        for _ in range(self.num_leaves // 2): # A binary tree with N leaves has N/2 internal nodes
            self.decision_nodes.append(DecisionNode())

        # Create leaf networks
        self.leaf_networks = []
        for _ in range(self.num_leaves):
            self.leaf_networks.append(LeafNetwork(
                hidden_layers=self.leaf_nn_params['hidden_layers'],
                output_dim=self.leaf_nn_params['output_dim'],
                activation=self.leaf_nn_params['activation'],
                output_activation=self.leaf_nn_params['output_activation']
            ))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Initialize probabilities of reaching each node
        # Start with probability 1.0 at the root
        node_probabilities = tf.ones((batch_size, 1))
        leaf_probabilities = []

        node_index = 0
        # Traverse down the tree level by level
        for level in range(self.depth):
            new_node_probabilities = []
            num_nodes_in_level = 2**level
            for i in range(num_nodes_in_level):
                # Get the routing probability from the decision node
                # Need to select the appropriate decision node based on the path
                # This simplified implementation uses nodes sequentially, a real tree would route
                # This is a conceptual simplification of the routing logic
                if level < self.depth:
                    routing_prob_left = self.decision_nodes[node_index + i](inputs) # Simplified: uses full input
                    routing_prob_right = 1.0 - routing_prob_left

                    # Probability of reaching the left child node
                    prob_to_left_child = node_probabilities[:, i:i+1] * routing_prob_left
                    # Probability of reaching the right child node
                    prob_to_right_child = node_probabilities[:, i:i+1] * routing_prob_right

                    new_node_probabilities.append(prob_to_left_child)
                    new_node_probabilities.append(prob_to_right_child)
                else:
                    # If at the leaf level, the node probability is the leaf probability
                    leaf_probabilities.append(node_probabilities[:, i:i+1])

            node_probabilities = tf.concat(new_node_probabilities, axis=1)
            node_index += num_nodes_in_level # Move to the next set of decision nodes

        # After traversing all levels, node_probabilities contains probabilities of reaching each leaf
        leaf_probabilities = node_probabilities

        # Get predictions from each leaf network
        leaf_predictions = [leaf_nn(inputs) for leaf_nn in self.leaf_networks] # Simplified: leaf NNs use full input

        # Combine leaf predictions weighted by the probability of reaching each leaf
        # Stack leaf probabilities and predictions
        leaf_probabilities = tf.stack(leaf_probabilities, axis=1)
        leaf_predictions = tf.stack(leaf_predictions, axis=1)

        # Weighted sum of leaf predictions
        # For classification, average probabilities; for regression, average values
        if self.leaf_nn_params['output_activation'] == 'sigmoid' or self.leaf_nn_params['output_activation'] == 'softmax':
             # Weighted average of probabilities
            averaged_predictions = tf.reduce_sum(leaf_probabilities * leaf_predictions, axis=1)
        else:
            # Weighted average of regression values
            averaged_predictions = tf.reduce_sum(leaf_probabilities * leaf_predictions, axis=1)

        return averaged_predictions

# --- Neural Forest (Ensemble of Neural Decision Trees) ---
class NeuralDecisionForest:
    def __init__(self, n_trees, tree_params, bagging_params, feature_params):
        self.n_trees = n_trees
        self.tree_params = tree_params
        self.bagging_params = bagging_params
        self.feature_params = feature_params
        self.forest = []
        self.feature_subsets = [] # Store feature indices for each tree

    def train(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        n_bagging_samples = int(self.bagging_params['bootstrap_sample_size'] * n_samples)
        # Determine the number of features to use for each tree
        n_feature_subset = max(1, int(self.feature_params['feature_subset_fraction'] * n_features)) # Ensure at least 1 feature

        for i in range(self.n_trees):
            print(f"Training Neural Decision Tree {i+1}/{self.n_trees}")

            # Step 1: Data Sampling (Bootstrap Aggregating)
            X_sample, y_sample = resample(X_train, y_train, n_samples=n_bagging_samples, replace=True, random_state=i)
            X_sample_df = pd.DataFrame(X_sample)

            # Step 2: Feature Selection (Random Subspace Method)
            # Select a random subset of feature indices for this tree
            selected_features_indices = np.random.choice(n_features, n_feature_subset, replace=False)
            self.feature_subsets.append(selected_features_indices)

            # Subset the data for training this tree
            X_sample_subset = X_sample_df.iloc[:, selected_features_indices]
            tree_input_dim = X_sample_subset.shape[1]

            # Step 3: Neural Decision Tree Model Definition and Training
            # Create a Neural Decision Tree with the appropriate input dimension
            neural_decision_tree = NeuralDecisionTree(
                depth=self.tree_params['tree_depth'],
                input_dim=tree_input_dim,
                leaf_nn_params={
                    'hidden_layers': self.tree_params['leaf_nn_hidden_layers'],
                    'activation': self.tree_params['leaf_nn_activation'],
                    'output_dim': self.tree_params['output_dim'],
                    'output_activation': self.tree_params['leaf_nn_output_activation']
                }
            )

            # Build the model with a dummy input to initialize weights
            neural_decision_tree.build(input_shape=(None, tree_input_dim))
            # Print model summary to see the structure (optional)
            # neural_decision_tree.summary()

            neural_decision_tree.compile(
                optimizer=self.tree_params['optimizer'],
                loss=self.tree_params['loss_function'],
                metrics=['accuracy'] # Add other metrics as needed
            )

            # Early stopping
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            # Train the Neural Decision Tree
            # Pass the subsetted data for training
            history = neural_decision_tree.fit(
                X_sample_subset,
                y_sample,
                epochs=self.tree_params['epochs_per_tree'],
                batch_size=self.tree_params['batch_size'],
                validation_split=0.2, # Use a portion of the bootstrap sample for validation
                callbacks=[early_stopping],
                verbose=0 # Set to 1 to see training progress per tree
            )
            self.forest.append(neural_decision_tree)

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

        # Reshape for binary classification output if needed
        if self.tree_params['output_dim'] == 1 and self.tree_params['leaf_nn_output_activation'] == 'sigmoid':
             return averaged_predictions.reshape(-1, 1)
        else:
             return averaged_predictions

    def predict(self, X_test):
        # For classification, convert probabilities to class labels
        probabilities = self.predict_proba(X_test)
        if self.tree_params['leaf_nn_output_activation'] == 'sigmoid': # Binary classification
            return (probabilities > 0.5).astype(int)
        elif self.tree_params['leaf_nn_output_activation'] == 'softmax': # Multi-class classification
            return np.argmax(probabilities, axis=1)
        else: # Regression
            return probabilities

# --- Empirical Evaluation ---
if __name__ == "__main__":
    tree_parameters = {
        'tree_depth': tree_depth,
        'leaf_nn_hidden_layers': leaf_nn_hidden_layers,
        'leaf_nn_activation': leaf_nn_activation,
        'output_dim': output_dim,
        'leaf_nn_output_activation': leaf_nn_output_activation,
        'loss_function': loss_function,
        'optimizer': optimizer,
        'epochs_per_tree': epochs_per_tree,
        'batch_size': batch_size
    }

    bagging_parameters = {
        'bootstrap_sample_size': bootstrap_sample_size
    }

    feature_parameters = {
        'feature_subset_fraction': feature_subset_fraction
    }

    # Create and train the Neural Forest (of Neural Decision Trees)
    neural_decision_forest = NeuralDecisionForest(
        n_trees=n_trees,
        tree_params=tree_parameters,
        bagging_params=bagging_parameters,
        feature_params=feature_parameters
    )
    # Pass the original training data to the forest's train method
    neural_decision_forest.train(X_train, y_train)

    # Make predictions on the test set
    y_pred_proba = neural_decision_forest.predict_proba(X_test)
    y_pred = neural_decision_forest.predict(X_test)

    # Evaluate the Neural Forest (Example for binary classification)
    from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

    print("\n--- Empirical Evaluation Results ---")
    if leaf_nn_output_activation == 'sigmoid':
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
    elif leaf_nn_output_activation == 'linear': # Regression
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared: {r2:.4f}")
    # Add evaluation for multi-class classification as needed
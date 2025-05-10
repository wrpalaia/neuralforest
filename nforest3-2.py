import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd # Though not explicitly used in this script, kept for potential future data loading
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration for a Single Neural Decision Tree ---
tree_depth = 3  # Depth of the tree
# num_leaves will be 2**tree_depth
leaf_nn_hidden_layers = [8]  # Hidden layer sizes for each leaf's neural network
leaf_nn_activation = 'relu'  # Activation function for hidden layers in leaf NNs
output_dim = 1               # Output dimension (1 for binary classification/regression)
leaf_nn_output_activation = 'sigmoid'  # Activation for the output layer of leaf NNs ('sigmoid' for binary classification, 'linear' for regression, 'softmax' for multi-class)
loss_function = 'binary_crossentropy'  # Loss function (e.g., 'binary_crossentropy', 'mean_squared_error', 'categorical_crossentropy')
optimizer = 'adam'           # Optimizer for training
epochs_for_tree = 50         # Number of epochs to train the tree
batch_size = 64              # Batch size for training

# --- Data Generation (Example) ---
# Using make_classification for a reproducible example
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
input_dim = X.shape[1]  # Input dimension for the tree

# Reshape y to be (n_samples, 1) if it's a binary classification task and output_dim is 1
if output_dim == 1 and len(y.shape) == 1:
    y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Neural Decision Tree Components ---

class LeafNetwork(keras.Model):
    """A small neural network at each leaf node of the decision tree."""
    def __init__(self, hidden_layers, output_dim, activation, output_activation, name=None):
        super(LeafNetwork, self).__init__(name=name)
        self.hidden_network_layers = []
        for i, size in enumerate(hidden_layers):
            self.hidden_network_layers.append(layers.Dense(size, activation=activation, name=f"leaf_hidden_{i}"))
        self.output_leaf_layer = layers.Dense(output_dim, activation=output_activation, name="leaf_output")

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_network_layers:
            x = layer(x)
        return self.output_leaf_layer(x)

class DecisionNode(layers.Layer):
    """A differentiable decision node that routes inputs using a sigmoid activation."""
    def __init__(self, **kwargs):
        super(DecisionNode, self).__init__(**kwargs)
        # Dense layer with 1 unit and sigmoid activation to produce a routing probability ( Entscheidung )
        self.router = layers.Dense(1, activation='sigmoid', name="router_dense")

    def call(self, inputs):
        return self.router(inputs) # Returns a probability [0, 1] for routing to the left child

class NeuralDecisionTree(keras.Model):
    """
    A single Neural Decision Tree with differentiable routing and leaf networks.
    The tree structure is fixed by 'depth'.
    Routing probabilities are learned by DecisionNode instances.
    Leaf nodes use LeafNetwork instances to make predictions.
    """
    def __init__(self, depth, input_dim, leaf_nn_params, name="neural_decision_tree", **kwargs):
        super(NeuralDecisionTree, self).__init__(name=name, **kwargs)
        self.depth = depth
        self.num_leaves = 2**depth
        self.input_dim = input_dim # Stored for reference, though not directly used in this __init__ after passing to leaves/nodes
        self.leaf_nn_params = leaf_nn_params

        # A complete binary tree of depth 'd' has 2^d leaves and 2^d - 1 internal (decision) nodes.
        self.num_decision_nodes = self.num_leaves - 1
        self.decision_nodes = []
        for i in range(self.num_decision_nodes):
            self.decision_nodes.append(DecisionNode(name=f"decision_node_{i}"))

        self.leaf_networks = []
        for i in range(self.num_leaves):
            self.leaf_networks.append(LeafNetwork(
                hidden_layers=self.leaf_nn_params['hidden_layers'],
                output_dim=self.leaf_nn_params['output_dim'],
                activation=self.leaf_nn_params['activation'],
                output_activation=self.leaf_nn_params['output_activation'],
                name=f"leaf_network_{i}"
            ))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # path_probabilities will store the probability of each sample reaching each node at the current level.
        # Initially, all samples reach the root node with probability 1.
        # Shape: (batch_size, num_nodes_at_current_level)
        current_level_path_probabilities = tf.ones((batch_size, 1))
        
        decision_node_idx_offset = 0 # Keeps track of which decision node to use from the flat list

        for level in range(self.depth):
            num_nodes_in_current_level = 2**level
            next_level_path_probabilities_list = []

            for i in range(num_nodes_in_current_level):
                # Probability of reaching the current node (which is about to split)
                prob_reaching_parent_node = current_level_path_probabilities[:, i:i+1]
                
                # Get routing probabilities from the corresponding decision node
                # The decision nodes are indexed breadth-first.
                decision_node = self.decision_nodes[decision_node_idx_offset + i]
                routing_prob_to_left = decision_node(inputs)  # Shape: (batch_size, 1)
                routing_prob_to_right = 1.0 - routing_prob_to_left

                # Calculate probabilities of reaching the left and right children
                prob_to_left_child = prob_reaching_parent_node * routing_prob_to_left
                prob_to_right_child = prob_reaching_parent_node * routing_prob_to_right

                next_level_path_probabilities_list.append(prob_to_left_child)
                next_level_path_probabilities_list.append(prob_to_right_child)
            
            current_level_path_probabilities = tf.concat(next_level_path_probabilities_list, axis=1)
            decision_node_idx_offset += num_nodes_in_current_level
            
        # After the loop, current_level_path_probabilities contains the probabilities of reaching each leaf node.
        # Shape: (batch_size, num_leaves)
        path_probabilities_to_leaves = current_level_path_probabilities

        # Get predictions from all leaf networks
        # Each leaf network takes the original input features.
        leaf_predictions_list = [leaf_nn(inputs) for leaf_nn in self.leaf_networks]
        # Stack predictions: list of num_leaves tensors of shape (batch_size, output_dim)
        # becomes a single tensor of shape (batch_size, num_leaves, output_dim).
        leaf_predictions_tensor = tf.stack(leaf_predictions_list, axis=1)

        # Weight leaf predictions by path probabilities
        # Path probabilities shape: (batch_size, num_leaves)
        # Leaf predictions shape: (batch_size, num_leaves, output_dim)
        # Expand dims of path_probabilities_to_leaves to (batch_size, num_leaves, 1) for broadcasting.
        path_probs_expanded = tf.expand_dims(path_probabilities_to_leaves, axis=2)

        # Element-wise multiplication and sum over leaves
        # weighted_predictions shape: (batch_size, num_leaves, output_dim)
        weighted_predictions = path_probs_expanded * leaf_predictions_tensor
        
        # Averaged predictions shape: (batch_size, output_dim)
        final_predictions = tf.reduce_sum(weighted_predictions, axis=1)

        return final_predictions

    def get_config(self):
        config = super(NeuralDecisionTree, self).get_config()
        config.update({
            "depth": self.depth,
            "input_dim": self.input_dim,
            "leaf_nn_params": self.leaf_nn_params,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Ensure custom objects are passed if needed, or handle them here
        # For this simple case, direct instantiation might be fine if LeafNetwork and DecisionNode are registered or simple enough
        return cls(**config)

# --- Main Execution for a Single Neural Decision Tree ---
if __name__ == "__main__":
    # Define parameters for the single tree's leaf networks
    leaf_nn_parameters = {
        'hidden_layers': leaf_nn_hidden_layers,
        'activation': leaf_nn_activation,
        'output_dim': output_dim,
        'output_activation': leaf_nn_output_activation
    }

    # Create the Neural Decision Tree
    print("Creating a single Neural Decision Tree...")
    single_neural_tree = NeuralDecisionTree(
        depth=tree_depth,
        input_dim=input_dim,
        leaf_nn_params=leaf_nn_parameters
    )

    # Build the model by calling it on a sample input shape (or use model.build)
    # This is necessary to create the weights of the layers.
    # Using (None, input_dim) allows for variable batch sizes.
    single_neural_tree.build(input_shape=(None, input_dim))
    single_neural_tree.summary() # Display model structure

    # Compile the tree
    single_neural_tree.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy'] # Add other metrics as needed (e.g., tf.keras.metrics.AUC())
    )

    # Early stopping callback to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )

    # Train the Neural Decision Tree
    print("\nTraining the Neural Decision Tree...")
    history = single_neural_tree.fit(
        X_train,
        y_train,
        epochs=epochs_for_tree,
        batch_size=batch_size,
        validation_split=0.2, # Use a portion of the training data for validation
        callbacks=[early_stopping],
        verbose=1 # Set to 1 to see training progress, 0 for silent, 2 for per-epoch
    )

    # Evaluate the model on the test set
    print("\nEvaluating the model on the test set...")
    loss, accuracy = single_neural_tree.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Make predictions on the test set
    print("\nMaking predictions on the test set...")
    y_pred_proba_single_tree = single_neural_tree.predict(X_test)

    # Convert probabilities to class labels for classification
    if leaf_nn_output_activation == 'sigmoid': # Binary classification
        y_pred_single_tree = (y_pred_proba_single_tree > 0.5).astype(int)
        # Additional metrics for binary classification
        roc_auc = roc_auc_score(y_test, y_pred_proba_single_tree)
        print(f"Test ROC AUC: {roc_auc:.4f}")
    elif leaf_nn_output_activation == 'softmax': # Multi-class classification
        y_pred_single_tree = np.argmax(y_pred_proba_single_tree, axis=1)
        # For multi-class, ensure y_test is appropriately shaped for accuracy_score (e.g., 1D array of class indices)
        # accuracy_eval = accuracy_score(np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test, y_pred_single_tree)
        # print(f"Test Accuracy (from sklearn): {accuracy_eval:.4f}")
    elif leaf_nn_output_activation == 'linear': # Regression
        y_pred_single_tree = y_pred_proba_single_tree
        mse = mean_squared_error(y_test, y_pred_single_tree)
        r2 = r2_score(y_test, y_pred_single_tree)
        print(f"Test Mean Squared Error: {mse:.4f}")
        print(f"Test R-squared: {r2:.4f}")
    else: # Fallback for other activation types
        y_pred_single_tree = y_pred_proba_single_tree


    print("\n--- Single Neural Tree Evaluation Summary ---")
    # The evaluation is already printed above during model.evaluate and prediction steps.
    # This section can be used for more detailed reports or visualizations if needed.

    # --- Deployment Considerations ---
    # To prepare for deployment, you would typically save the trained model:
    # model_save_path = "single_neural_tree_model"
    # single_neural_tree.save(model_save_path)
    # print(f"\nSingle Neural Tree model saved to {model_save_path}")

    # When deploying, you would load the model and use it for predictions:
    # loaded_model = keras.models.load_model(model_save_path)
    # example_prediction = loaded_model.predict(X_test[:5]) # Example with first 5 test samples
    # print("\nPrediction with loaded model (first 5 test samples):")
    # print(example_prediction)
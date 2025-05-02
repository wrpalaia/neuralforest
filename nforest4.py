import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
# Tree parameters
tree_depth = 2 # Depth of the binary tree (e.g., depth 2 means 4 leaf nodes)
n_leaf_networks = 2**tree_depth

# Parameters for the small neural networks at the leaves
leaf_nn_hidden_layers = [8] # Number of neurons in hidden layers of leaf NNs
leaf_nn_activation = 'relu'
leaf_nn_output_dim = 1 # For binary classification
leaf_nn_output_activation = 'sigmoid'

# Training parameters
loss_function = 'binary_crossentropy'
optimizer = 'adam'
epochs = 50
batch_size = 32

# --- Data Generation (Example) ---
# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=7, random_state=42)
input_dim = X.shape[1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Neural Decision Tree Components ---

class DifferentiableDecisionNode(layers.Layer):
    """
    A layer representing a differentiable decision node in the tree.
    Learns a linear combination of inputs and applies a sigmoid to route.
    """
    def __init__(self, **kwargs):
        super(DifferentiableDecisionNode, self).__init__(**kwargs)
        self.dense = layers.Dense(1, activation='sigmoid') # Sigmoid output for routing probability

    def call(self, inputs):
        # Output is the probability of going to the right child
        return self.dense(inputs)

class LeafNetwork(keras.Model):
    """
    A small neural network residing at a leaf node.
    """
    def __init__(self, hidden_layers, output_dim, activation, output_activation, **kwargs):
        super(LeafNetwork, self).__init__(**kwargs)
        self.hidden_layers = [layers.Dense(size, activation=activation) for size in hidden_layers]
        self.output_layer = layers.Dense(output_dim, activation=output_activation)

    def call(self, inputs):
        x = inputs
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return self.output_layer(x)

class NeuralDecisionTree(keras.Model):
    """
    A single Neural Decision Tree with differentiable routing and leaf networks.
    """
    def __init__(self, depth, input_dim, leaf_nn_params, **kwargs):
        super(NeuralDecisionTree, self).__init__(**kwargs)
        self.depth = depth
        self.input_dim = input_dim
        self.leaf_nn_params = leaf_nn_params
        self.n_leaf_nodes = 2**self.depth

        # Create decision nodes (internal nodes)
        # A binary tree of depth D has 2^D - 1 internal nodes
        self.decision_nodes = [DifferentiableDecisionNode() for _ in range(2**self.depth - 1)]

        # Create leaf networks
        self.leaf_networks = [LeafNetwork(**self.leaf_nn_params) for _ in range(self.n_leaf_nodes)]

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Path probabilities for each instance to reach each leaf node
        # Initially, each instance is at the root with probability 1
        path_probabilities = tf.ones((batch_size, 1))

        # Propagate through the tree
        node_index = 0
        for level in range(self.depth):
            new_path_probabilities = []
            for i in range(2**level):
                # For each node at the current level
                current_node_probability = path_probabilities[:, i:i+1]

                # Get routing probability from the decision node
                routing_prob_right = self.decision_nodes[node_index](inputs)
                routing_prob_left = 1.0 - routing_prob_right

                # Calculate probabilities of reaching the children nodes
                prob_to_left_child = current_node_probability * routing_prob_left
                prob_to_right_child = current_node_probability * routing_prob_right

                new_path_probabilities.append(prob_to_left_child)
                new_path_probabilities.append(prob_to_right_child)

                node_index += 1 # Move to the next decision node for the next iteration

            path_probabilities = tf.concat(new_path_probabilities, axis=1)

        # At this point, path_probabilities has shape (batch_size, n_leaf_nodes)
        # where each element is the probability of an instance reaching a specific leaf

        # Get predictions from each leaf network
        leaf_predictions = [leaf_nn(inputs) for leaf_nn in self.leaf_networks]

        # Stack leaf predictions: shape (batch_size, n_leaf_nodes, output_dim)
        leaf_predictions_stacked = tf.stack(leaf_predictions, axis=1)

        # Weighted sum of leaf predictions based on path probabilities
        # Ensure dimensions match for element-wise multiplication and sum
        # path_probabilities shape: (batch_size, n_leaf_nodes)
        # leaf_predictions_stacked shape: (batch_size, n_leaf_nodes, output_dim)
        # We need to expand path_probabilities to (batch_size, n_leaf_nodes, 1)
        weighted_predictions = tf.expand_dims(path_probabilities, axis=-1) * leaf_predictions_stacked

        # Sum over the leaf node dimension to get the final prediction for each instance
        final_prediction = tf.reduce_sum(weighted_predictions, axis=1)

        return final_prediction

# --- Empirical Evaluation ---
if __name__ == "__main__":
    leaf_nn_parameters = {
        'hidden_layers': leaf_nn_hidden_layers,
        'activation': leaf_nn_activation,
        'output_dim': leaf_nn_output_dim,
        'output_activation': leaf_nn_output_activation
    }

    # Create a single Neural Decision Tree model
    neural_decision_tree = NeuralDecisionTree(
        depth=tree_depth,
        input_dim=input_dim,
        leaf_nn_params=leaf_nn_parameters
    )

    # Compile the model
    neural_decision_tree.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    print("Training a single Neural Decision Tree...")

    # Train the model end-to-end
    history = neural_decision_tree.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2 # Use a validation split
    )

    print("\nEvaluating the single Neural Decision Tree...")

    # Evaluate the model on the test set
    loss, accuracy = neural_decision_tree.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # To build a Neural Forest using this method:
    # You would create an ensemble of `n_trees` NeuralDecisionTree instances.
    # Each tree would ideally be trained on a bootstrap sample of the data
    # and potentially a random subset of features (random subspace).
    # Prediction would involve averaging the predictions from all the trees in the forest.

    # Example (Conceptual) of Forest Prediction:
    # forest_predictions = []
    # for tree in neural_forest_of_these: # Assuming neural_forest_of_these is a list of trained NeuralDecisionTree
    #    tree_pred = tree.predict(X_test)
    #    forest_predictions.append(tree_pred)
    # averaged_forest_predictions = np.mean(np.stack(forest_predictions, axis=-1), axis=-1)
    # final_forest_output = (averaged_forest_predictions > 0.5).astype(int) # For binary classification
    # print("\nConceptual Forest Prediction (averaging single tree outputs)")
    # print(f"Conceptual Forest Accuracy: {accuracy_score(y_test, final_forest_output):.4f}")
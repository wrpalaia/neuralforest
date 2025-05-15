import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.utils import resample # For bagging
import os # To manage saving directories
import json # For saving/loading config

# --- Neural Decision Tree Components ---

class LeafNetwork(keras.Model):
    """
    A small neural network at each leaf node of the decision tree.
    This network processes the input features and makes a prediction if the input
    is routed to this leaf.
    """
    def __init__(self, hidden_layers, output_dim, activation, output_activation, name=None, **kwargs):
        super(LeafNetwork, self).__init__(name=name, **kwargs)
        self.hidden_network_layers = []
        for i, size in enumerate(hidden_layers):
            self.hidden_network_layers.append(layers.Dense(size, activation=activation, name=f"leaf_hidden_{i}"))
        self.output_leaf_layer = layers.Dense(output_dim, activation=output_activation, name="leaf_output")

        # Store configs for get_config
        self._hidden_layers_config = hidden_layers
        self._output_dim_config = output_dim
        self._activation_config = activation
        self._output_activation_config = output_activation

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_network_layers:
            x = layer(x)
        return self.output_leaf_layer(x)

    def get_config(self):
        config = super(LeafNetwork, self).get_config()
        config.update({
            "hidden_layers": self._hidden_layers_config,
            "output_dim": self._output_dim_config,
            "activation": self._activation_config,
            "output_activation": self._output_activation_config,
            # Name is handled by super class
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DecisionNode(layers.Layer):
    """
    A differentiable decision node that routes inputs using a sigmoid activation.
    The node learns a linear combination of input features to make a soft routing decision.
    """
    def __init__(self, **kwargs):
        super(DecisionNode, self).__init__(**kwargs)
        # Dense layer for routing will be created in build method

    def build(self, input_shape):
        # The router dense layer learns to project the input features to a single value,
        # which is then passed through a sigmoid to get a routing probability.
        self.router = layers.Dense(1, activation='sigmoid', name="router_dense")
        super(DecisionNode, self).build(input_shape) # Important: call super().build()

    def call(self, inputs):
        """Returns the probability of routing to the left child."""
        return self.router(inputs)

    def get_config(self):
        config = super(DecisionNode, self).get_config()
        # No specific parameters to add beyond default layer config for this version
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NeuralDecisionTree(keras.Model):
    """
    A single Neural Decision Tree with differentiable routing and leaf networks.
    The tree structure is fixed by its depth. Inputs are routed through decision
    nodes, and the final prediction is a weighted average of the predictions from
    all leaf networks, weighted by the path probabilities to those leaves.
    """
    def __init__(self, depth, leaf_nn_params_dict, name="neural_decision_tree", **kwargs):
        super(NeuralDecisionTree, self).__init__(name=name, **kwargs)
        if depth <= 0:
            raise ValueError("Tree depth must be greater than 0.")
        self.tree_depth = depth # Renamed from 'depth' for clarity in get_config
        self.num_leaves = 2**self.tree_depth
        self.leaf_nn_params = leaf_nn_params_dict

        self.num_decision_nodes = self.num_leaves - 1
        self.decision_nodes_list = [DecisionNode(name=f"decision_node_{i}") for i in range(self.num_decision_nodes)] # Renamed for clarity

        self.leaf_networks_list = [LeafNetwork( # Renamed for clarity
                hidden_layers=self.leaf_nn_params['hidden_layers'],
                output_dim=self.leaf_nn_params['output_dim'],
                activation=self.leaf_nn_params['activation'],
                output_activation=self.leaf_nn_params['output_activation'],
                name=f"leaf_network_{i}"
            ) for i in range(self.num_leaves)]

    def build(self, input_shape):
         # input_shape is (batch_size, input_dim)
         # Build decision nodes - they receive the original input features
         for node in self.decision_nodes_list:
             if not node.built: # Build only if not already built
                node.build(input_shape)
         # Build leaf networks - they also receive original input features
         for leaf_nn in self.leaf_networks_list:
              if not leaf_nn.built: # Build only if not already built
                leaf_nn.build(input_shape)
         super(NeuralDecisionTree, self).build(input_shape)


    def call(self, inputs, training=False): # Added training flag
        batch_size = tf.shape(inputs)[0]
        # Initialize path probabilities for the root (level 0 has one node, the conceptual root)
        # current_level_path_probabilities shape: (batch_size, nodes_in_level)
        current_level_path_probabilities = tf.ones((batch_size, 1), dtype=tf.float32)
        decision_node_idx_offset = 0

        # Traverse the tree level by level
        for level in range(self.tree_depth):
            num_nodes_in_current_level = 2**level
            next_level_path_probabilities_list = []

            for i in range(num_nodes_in_current_level):
                # Probability of reaching the current parent node (or pseudo-node at level i)
                prob_reaching_parent_node = current_level_path_probabilities[:, i:i+1]

                # Get the actual decision node for this split
                decision_node = self.decision_nodes_list[decision_node_idx_offset + i]

                # Decision nodes receive the full input features
                routing_prob_to_left = decision_node(inputs, training=training) # Pass training flag
                routing_prob_to_right = 1.0 - routing_prob_to_left

                # Calculate probabilities of reaching the left and right children
                prob_to_left_child = prob_reaching_parent_node * routing_prob_to_left
                prob_to_right_child = prob_reaching_parent_node * routing_prob_to_right

                next_level_path_probabilities_list.extend([prob_to_left_child, prob_to_right_child])

            # Concatenate probabilities for the next level
            current_level_path_probabilities = tf.concat(next_level_path_probabilities_list, axis=1)
            decision_node_idx_offset += num_nodes_in_current_level

        # At the end, current_level_path_probabilities contains path probabilities to each leaf
        path_probabilities_to_leaves = current_level_path_probabilities # Shape: (batch_size, num_leaves)

        # Leaf networks receive the full input features
        leaf_predictions_list = [leaf_nn(inputs, training=training) for leaf_nn in self.leaf_networks_list] # Pass training flag

        # Stack predictions from all leaves: (num_leaves, batch_size, output_dim)
        # Then transpose to: (batch_size, num_leaves, output_dim)
        leaf_predictions_tensor = tf.stack(leaf_predictions_list, axis=1)

        # Expand path_probabilities_to_leaves for broadcasting: (batch_size, num_leaves, 1)
        path_probs_expanded = tf.expand_dims(path_probabilities_to_leaves, axis=2)

        # Weight leaf predictions by their path probabilities
        # (batch_size, num_leaves, output_dim) * (batch_size, num_leaves, 1)
        weighted_predictions = path_probs_expanded * leaf_predictions_tensor

        # Sum weighted predictions across leaves: (batch_size, output_dim)
        final_predictions = tf.reduce_sum(weighted_predictions, axis=1)
        return final_predictions

    def get_config(self):
        config = super(NeuralDecisionTree, self).get_config()
        config.update({
            "depth": self.tree_depth, # Use the renamed attribute
            "leaf_nn_params_dict": self.leaf_nn_params,
            # Name is handled by the super class
            # Decision nodes and leaf networks are attributes and will be saved
            # by Keras if they are layers/models themselves.
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Keras's load_model will handle recreating the layers/sub-models
        # provided they are attributes of the main model and have get_config/from_config
        return cls(**config)


# --- Neural Forest Class ---
class NeuralForest:
    """
    Manages an ensemble of NeuralDecisionTree models.
    Each tree is trained on a bootstrap sample of the data (bagging).
    Predictions are typically averaged across all trees.
    """
    def __init__(self, num_trees, tree_depth, input_dim, leaf_nn_config):
        if num_trees <= 0:
            raise ValueError("Number of trees must be greater than 0.")
        self.num_trees = num_trees
        self.tree_depth = tree_depth # Renamed from tree_depth_val for consistency
        self.input_dim = input_dim
        self.leaf_nn_params = leaf_nn_config # leaf_nn_config is a dict
        self.trees = []
        self.history_per_tree = [] # Optional: to store training history of each tree

        # Determine task type and loss function based on output activation
        self.loss_function = None
        self.metrics_list = []
        output_activation = self.leaf_nn_params.get('output_activation', '').lower()

        if output_activation == 'sigmoid':
            self.loss_function = 'binary_crossentropy'
            self.metrics_list = ['accuracy', tf.keras.metrics.AUC(name='auc')]
        elif output_activation == 'softmax':
            self.loss_function = 'categorical_crossentropy'
            self.metrics_list = ['accuracy', tf.keras.metrics.AUC(name='auc')]
        elif output_activation == 'linear':
            self.loss_function = 'mean_squared_error'
            self.metrics_list = ['mean_absolute_error']
        else:
             raise ValueError(
                 f"Unsupported leaf_nn_output_activation: {output_activation}. "
                 "Choose from 'sigmoid', 'softmax', or 'linear'."
            )

    def fit(self, X_data, y_data, epochs, batch_size_per_tree, # Renamed for clarity
            validation_split_per_tree=0.1, early_stopping_patience=5,
            forest_verbose=1, tree_training_verbose=0, random_state_base=None):
        """
        Trains the Neural Forest.

        Args:
            X_data: Training features.
            y_data: Training labels.
            epochs: Number of epochs to train each tree.
            batch_size_per_tree: Batch size for training each tree.
            validation_split_per_tree: Fraction of training data for validation for each tree.
            early_stopping_patience: Patience for early stopping for each tree.
            forest_verbose: Verbosity level for forest training (0 or 1).
            tree_training_verbose: Verbosity level for individual tree Keras fit (0, 1, or 2).
            random_state_base: Base seed for bagging. If None, uses np.random.randint.
        """
        self.trees = [] # Reset trees for a new fit
        self.history_per_tree = []

        if not isinstance(X_data, tf.Tensor):
            X_data = tf.convert_to_tensor(X_data, dtype=tf.float32)
        if not isinstance(y_data, tf.Tensor):
            y_data = tf.convert_to_tensor(y_data, dtype=tf.float32) # Ensure correct dtype

        for i in range(self.num_trees):
            if forest_verbose > 0:
                print(f"\n--- Training Tree {i+1}/{self.num_trees} ---")

            # Bagging: Create a bootstrap sample
            # Use a different random state for each bag for diversity
            current_random_state = None
            if random_state_base is not None:
                current_random_state = random_state_base + i
            else: # If no base seed, use fully random seeds
                current_random_state = np.random.randint(0, 2**32 - 1)

            X_sample, y_sample = resample(X_data.numpy(), y_data.numpy(), random_state=current_random_state)
            X_sample = tf.convert_to_tensor(X_sample, dtype=tf.float32)
            y_sample = tf.convert_to_tensor(y_sample, dtype=y_data.dtype) # Match original y_data dtype

            tree = NeuralDecisionTree(
                depth=self.tree_depth,
                leaf_nn_params_dict=self.leaf_nn_params,
                name=f"neural_tree_{i}"
            )

            # Build and compile the tree
            # Input shape is (None, num_features)
            tree.build(input_shape=(None, self.input_dim))
            tree.compile(
                optimizer=tf.keras.optimizers.Adam(), # New optimizer instance per tree
                loss=self.loss_function,
                metrics=self.metrics_list
            )

            callbacks_for_tree = []
            early_stop_cb = None # Define for access after fit
            if early_stopping_patience > 0 and validation_split_per_tree > 0:
                early_stop_cb = keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    verbose=tree_training_verbose
                )
                callbacks_for_tree.append(early_stop_cb)

            history = tree.fit(
                X_sample,
                y_sample,
                epochs=epochs,
                batch_size=batch_size_per_tree,
                validation_split=validation_split_per_tree if validation_split_per_tree > 0 else None,
                callbacks=callbacks_for_tree,
                verbose=tree_training_verbose
            )

            self.trees.append(tree)
            self.history_per_tree.append(history)

            if forest_verbose > 0 and early_stop_cb:
                 if early_stop_cb.stopped_epoch > 0 :
                    print(f"Tree {i+1} stopped early at epoch {early_stop_cb.stopped_epoch + 1}.")
                 elif epochs > 0 : # Only print if epochs were actually run
                     print(f"Tree {i+1} finished {epochs} epochs.")
            elif forest_verbose > 0 and epochs > 0:
                 print(f"Tree {i+1} finished {epochs} epochs.")


        if forest_verbose > 0:
            print("\nNeural Forest training complete.")


    def predict_proba(self, X_data, batch_size=32):
        """
        Predicts probabilities (or raw values for regression) for the input data.

        Args:
            X_data: Input features.
            batch_size: Batch size for prediction.

        Returns:
            Numpy array of averaged predictions from all trees.
        """
        if not self.trees:
            raise ValueError("The forest has not been trained yet or no trees were trained.")

        if not isinstance(X_data, tf.Tensor):
            X_data = tf.convert_to_tensor(X_data, dtype=tf.float32)

        all_tree_predictions = []
        for tree in self.trees:
            # Each tree.predict returns a tf.Tensor
            all_tree_predictions.append(tree.predict(X_data, batch_size=batch_size, verbose=0))

        # Stack predictions: list of num_trees tensors of shape (batch_size, output_dim)
        # becomes a single tensor of shape (num_trees, batch_size, output_dim).
        if not all_tree_predictions: # Should not happen if self.trees is not empty
            return np.array([])

        predictions_stacked = tf.stack(all_tree_predictions, axis=0)

        # Average probabilities across trees
        # forest_predictions shape: (batch_size, output_dim)
        forest_mean_predictions = tf.reduce_mean(predictions_stacked, axis=0)
        return forest_mean_predictions.numpy() # Return as numpy array


    def predict(self, X_data, batch_size=32):
        """
        Makes final class predictions for classification tasks or returns values for regression.

        Args:
            X_data: Input features.
            batch_size: Batch size for prediction.

        Returns:
            Numpy array of final predictions.
        """
        probabilities_or_values = self.predict_proba(X_data, batch_size=batch_size)
        output_activation = self.leaf_nn_params.get('output_activation', '').lower()

        if output_activation == 'sigmoid': # Binary classification
            return (probabilities_or_values > 0.5).astype(int)
        elif output_activation == 'softmax': # Multi-class classification
            return np.argmax(probabilities_or_values, axis=1)
        # For 'linear' (regression), predict_proba already returns the values
        return probabilities_or_values

    def save(self, filepath):
        """
        Saves the entire neural forest, including configuration and all individual trees.
        Models are saved in TensorFlow's SavedModel format.

        Args:
            filepath: Directory path to save the forest.
        """
        if not self.trees:
             raise ValueError("The forest has not been trained or is empty. Cannot save.")

        os.makedirs(filepath, exist_ok=True)
        # Save forest configuration (number of trees, depth, leaf params, input_dim)
        forest_config = {
            'num_trees': self.num_trees,
            'tree_depth': self.tree_depth,
            'input_dim': self.input_dim,
            'leaf_nn_config': self.leaf_nn_params, # This is already a dict
            'loss_function': self.loss_function, # Save determined loss/metrics
            'metrics_list_names': [m.name if hasattr(m, 'name') else str(m) for m in self.metrics_list]
        }

        with open(os.path.join(filepath, 'forest_config.json'), 'w') as f:
            json.dump(forest_config, f, indent=4)

        # Save each tree individually
        tree_save_dir = os.path.join(filepath, 'trees_savedmodel') # More specific name
        os.makedirs(tree_save_dir, exist_ok=True)
        for i, tree in enumerate(self.trees):
            tree_path = os.path.join(tree_save_dir, f'tree_{i}')
            tree.save(tree_path, save_format='tf')

        print(f"Neural Forest successfully saved to {filepath}")


    @classmethod
    def load(cls, filepath):
        """
        Loads a neural forest from a saved directory.

        Args:
            filepath: Directory path from which to load the forest.

        Returns:
            An instance of NeuralForest with loaded trees and configuration.
        """
        config_path = os.path.join(filepath, 'forest_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Forest configuration file not found at {config_path}")

        with open(config_path, 'r') as f:
            forest_config = json.load(f)

        # Create an instance of the NeuralForest using the loaded config
        loaded_forest = cls(
            num_trees=forest_config['num_trees'],
            tree_depth=forest_config['tree_depth'],
            input_dim=forest_config['input_dim'],
            leaf_nn_config=forest_config['leaf_nn_config']
        )
        # Optionally restore loss and metrics if needed for further compilation/training
        # loaded_forest.loss_function = forest_config.get('loss_function')
        # loaded_forest.metrics_list_names = forest_config.get('metrics_list_names')


        # Load each tree individually
        tree_load_dir = os.path.join(filepath, 'trees_savedmodel')
        if not os.path.isdir(tree_load_dir): # Check if it's a directory
             raise FileNotFoundError(f"Saved trees directory not found or is not a directory at {tree_load_dir}")

        loaded_trees = []
        # List directories inside tree_load_dir, assuming each is a saved tree
        # Filter for actual directories that might represent saved models
        tree_subdirs = sorted([
            d for d in os.listdir(tree_load_dir)
            if os.path.isdir(os.path.join(tree_load_dir, d)) and d.startswith('tree_')
        ])


        if len(tree_subdirs) != loaded_forest.num_trees:
             print(
                 f"Warning: Expected {loaded_forest.num_trees} trees based on config, "
                 f"but found {len(tree_subdirs)} saved tree directories in {tree_load_dir}. "
                 "Loading found trees and updating num_trees."
            )
             # Update num_trees to reflect actually loaded trees, or handle as an error
             # For robustness, we can try to load what's there.
             # loaded_forest.num_trees = len(tree_subdirs) # This could be problematic if some are missing

        custom_objects_to_pass = {
            'LeafNetwork': LeafNetwork,
            'DecisionNode': DecisionNode,
            'NeuralDecisionTree': NeuralDecisionTree
            # Add other custom objects if you introduce more that are part of NeuralDecisionTree
        }

        for i in range(forest_config['num_trees']): # Iterate based on original config
            tree_folder_name = f'tree_{i}'
            tree_path = os.path.join(tree_load_dir, tree_folder_name)
            if not os.path.isdir(tree_path):
                print(f"Warning: Saved model for tree_{i} not found at {tree_path}. Skipping.")
                continue # Or raise an error if all trees must be present

            try:
                # Load the tree, providing custom objects
                loaded_tree = keras.models.load_model(tree_path, custom_objects=custom_objects_to_pass)
                loaded_trees.append(loaded_tree)
            except Exception as e:
                print(f"Error loading tree from {tree_path}: {e}. Skipping this tree.")


        if not loaded_trees:
            raise ValueError(f"No trees were successfully loaded from {tree_load_dir}.")

        loaded_forest.trees = loaded_trees
        # Update num_trees to actual number of loaded trees
        loaded_forest.num_trees = len(loaded_trees)

        print(f"Neural Forest with {loaded_forest.num_trees} tree(s) successfully loaded from {filepath}")
        return loaded_forest

# It's good practice to have custom objects defined or importable for loading,
# The NeuralForest.load method handles passing them, so a global dict isn't strictly
# necessary for the user of NeuralForest.load but can be useful for direct Keras loading.
custom_objects_for_keras_load_model = {
    'LeafNetwork': LeafNetwork,
    'DecisionNode': DecisionNode,
    'NeuralDecisionTree': NeuralDecisionTree
}
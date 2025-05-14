import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.utils import resample # For bagging
import os # To manage saving directories

# --- Neural Decision Tree Components (Identical to refined single tree) ---

class LeafNetwork(keras.Model):
    """A small neural network at each leaf node of the decision tree."""
    def __init__(self, hidden_layers, output_dim, activation, output_activation, name=None, **kwargs):
        super(LeafNetwork, self).__init__(name=name, **kwargs)
        self.hidden_network_layers = []
        for i, size in enumerate(hidden_layers):
            self.hidden_network_layers.append(layers.Dense(size, activation=activation, name=f"leaf_hidden_{i}"))
        self.output_leaf_layer = layers.Dense(output_dim, activation=output_activation, name="leaf_output")
        # Store configs for get_config
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation = activation
        self.output_activation = output_activation


    def call(self, inputs):
        x = inputs
        for layer in self.hidden_network_layers:
            x = layer(x)
        return self.output_leaf_layer(x)

    def get_config(self):
        config = super(LeafNetwork, self).get_config()
        config.update({
            "hidden_layers": self.hidden_layers,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "name": self.name # Include name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DecisionNode(layers.Layer):
    """A differentiable decision node that routes inputs using a sigmoid activation."""
    def __init__(self, **kwargs):
        super(DecisionNode, self).__init__(**kwargs)
        # Use build to create weights, it's safer with Layer subclassing
        # self.router = layers.Dense(1, activation='sigmoid', name="router_dense")

    def build(self, input_shape):
        self.router = layers.Dense(1, activation='sigmoid', name="router_dense")
        super(DecisionNode, self).build(input_shape) # Important: call super().build()

    def call(self, inputs):
        return self.router(inputs)

    def get_config(self):
        config = super(DecisionNode, self).get_config()
        # No specific parameters to add beyond default layer config
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NeuralDecisionTree(keras.Model):
    """A single Neural Decision Tree with differentiable routing and leaf networks."""
    def __init__(self, depth, leaf_nn_params_dict, name="neural_decision_tree", **kwargs):
        super(NeuralDecisionTree, self).__init__(name=name, **kwargs)
        self.depth = depth
        self.num_leaves = 2**depth
        self.leaf_nn_params = leaf_nn_params_dict

        self.num_decision_nodes = self.num_leaves - 1
        self.decision_nodes = [DecisionNode(name=f"decision_node_{i}") for i in range(self.num_decision_nodes)]

        self.leaf_networks = [LeafNetwork(
                hidden_layers=self.leaf_nn_params['hidden_layers'],
                output_dim=self.leaf_nn_params['output_dim'],
                activation=self.leaf_nn_params['activation'],
                output_activation=self.leaf_nn_params['output_activation'],
                name=f"leaf_network_{i}"
            ) for i in range(self.num_leaves)]

    # It's often good practice to define build for Models as well
    def build(self, input_shape):
         # Assuming input_shape is (batch_size, input_dim)
         input_dim = input_shape[-1]
         # Build decision nodes - they receive the original input features
         for node in self.decision_nodes:
             node.build((None, input_dim)) # Use None for batch size
         # Build leaf networks - they also receive original input features
         for leaf_nn in self.leaf_networks:
              leaf_nn.build((None, input_dim)) # Use None for batch size
         super(NeuralDecisionTree, self).build(input_shape)


    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        current_level_path_probabilities = tf.ones((batch_size, 1))
        decision_node_idx_offset = 0

        for level in range(self.depth):
            num_nodes_in_current_level = 2**level
            next_level_path_probabilities_list = []
            for i in range(num_nodes_in_current_level):
                prob_reaching_parent_node = current_level_path_probabilities[:, i:i+1]
                decision_node = self.decision_nodes[decision_node_idx_offset + i]
                # Decision nodes receive the full input features
                routing_prob_to_left = decision_node(inputs)
                routing_prob_to_right = 1.0 - routing_prob_to_left
                prob_to_left_child = prob_reaching_parent_node * routing_prob_to_left
                prob_to_right_child = prob_reaching_parent_node * routing_prob_to_right
                next_level_path_probabilities_list.extend([prob_to_left_child, prob_to_right_child])
            current_level_path_probabilities = tf.concat(next_level_path_probabilities_list, axis=1)
            decision_node_idx_offset += num_nodes_in_current_level

        path_probabilities_to_leaves = current_level_path_probabilities
        
        # Leaf networks receive the full input features
        leaf_predictions_list = [leaf_nn(inputs) for leaf_nn in self.leaf_networks]
        
        leaf_predictions_tensor = tf.stack(leaf_predictions_list, axis=1)
        path_probs_expanded = tf.expand_dims(path_probabilities_to_leaves, axis=2)
        weighted_predictions = path_probs_expanded * leaf_predictions_tensor
        final_predictions = tf.reduce_sum(weighted_predictions, axis=1)
        return final_predictions

    def get_config(self):
        config = super(NeuralDecisionTree, self).get_config()
        config.update({
            "depth": self.depth,
            "leaf_nn_params_dict": self.leaf_nn_params,
            "name": self.name # Include name
            # Note: Decision nodes and leaf networks are not explicitly added here,
            # Keras handles saving/loading them as they are model/layer attributes.
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Keras's load_model will handle recreating the layers/sub-models
        # provided they are attributes of the main model and have get_config/from_config
        return cls(**config)


# --- Neural Forest Class ---
class NeuralForest:
    """Manages an ensemble of NeuralDecisionTree models."""
    def __init__(self, num_trees, tree_depth_val, input_dim, leaf_nn_config):
        self.num_trees = num_trees
        self.tree_depth = tree_depth_val
        self.input_dim = input_dim
        self.leaf_nn_params = leaf_nn_config
        self.trees = []
        self.history_per_tree = [] # Optional: to store training history of each tree

        # Determine task type and loss function based on output activation
        self.loss_function = None
        self.metrics_list = None
        if self.leaf_nn_params['output_activation'] == 'sigmoid':
            self.loss_function = 'binary_crossentropy'
            self.metrics_list = ['accuracy', tf.keras.metrics.AUC(name='auc')]
        elif self.leaf_nn_params['output_activation'] == 'softmax':
            self.loss_function = 'categorical_crossentropy'
            self.metrics_list = ['accuracy', tf.keras.metrics.AUC(name='auc')] # AUC might need one-hot y_true
        elif self.leaf_nn_params['output_activation'] == 'linear':
            self.loss_function = 'mean_squared_error'
            self.metrics_list = ['mean_absolute_error']
        else:
             raise ValueError(f"Unsupported leaf_nn_output_activation: {self.leaf_nn_params['output_activation']}")


    def fit(self, X_data, y_data, epochs, tree_batch_size, validation_split_per_tree=0.1, early_stopping_patience=5, forest_verbose=1, tree_training_verbose=0):
        self.trees = [] # Reset trees for a new fit
        self.history_per_tree = []

        for i in range(self.num_trees):
            if forest_verbose > 0:
                print(f"\nTraining Tree {i+1}/{self.num_trees}...")

            # Bagging: Create a bootstrap sample
            # Use a different random state for each bag for diversity
            X_sample, y_sample = resample(X_data, y_data, random_state=np.random.randint(10000)) 

            tree = NeuralDecisionTree(
                depth=self.tree_depth,
                leaf_nn_params_dict=self.leaf_nn_params,
                name=f"neural_tree_{i}"
            )

            # Build and compile the tree
            tree.build(input_shape=(None, self.input_dim))
            tree.compile(
                optimizer=tf.keras.optimizers.Adam(), # New optimizer instance per tree
                loss=self.loss_function,
                metrics=self.metrics_list
            )

            callbacks_for_tree = []
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
                batch_size=tree_batch_size,
                validation_split=validation_split_per_tree if validation_split_per_tree > 0 else None,
                callbacks=callbacks_for_tree,
                verbose=tree_training_verbose
            )

            self.trees.append(tree)
            self.history_per_tree.append(history)

            if forest_verbose > 0 and callbacks_for_tree: # Check if early stopping was used
                 if early_stop_cb.stopped_epoch > 0 :
                    print(f"Tree {i+1} stopped early at epoch {early_stop_cb.stopped_epoch + 1}.")
                 else:
                     print(f"Tree {i+1} finished {epochs} epochs.")


        if forest_verbose > 0:
            print("\nNeural Forest training complete.")


    def predict_proba(self, X_data):
        if not self.trees:
            raise ValueError("The forest has not been trained yet or no trees were trained.")

        # Convert to tensor if not already
        X_data = tf.convert_to_tensor(X_data, dtype=tf.float32)

        all_tree_predictions = []
        for tree in self.trees:
            all_tree_predictions.append(tree.predict(X_data, verbose=0)) # Suppress tree prediction progress

        # Stack predictions: list of num_trees tensors of shape (batch_size, output_dim)
        # becomes a single tensor of shape (num_trees, batch_size, output_dim).
        predictions_stacked = tf.stack(all_tree_predictions, axis=0)

        # Average probabilities across trees
        # forest_predictions shape: (batch_size, output_dim)
        forest_mean_predictions = tf.reduce_mean(predictions_stacked, axis=0)
        return forest_mean_predictions.numpy() # Return as numpy array


    def predict(self, X_data):
        """Makes final class predictions for classification tasks."""
        probabilities = self.predict_proba(X_data)

        if self.leaf_nn_params['output_activation'] == 'sigmoid': # Binary classification
            return (probabilities > 0.5).astype(int)
        elif self.leaf_nn_params['output_activation'] == 'softmax': # Multi-class classification
            return np.argmax(probabilities, axis=1)
        else: # Regression or other cases where probabilities are the final output
            return probabilities

    def save(self, filepath):
        """Saves the entire neural forest."""
        if not self.trees:
             raise ValueError("The forest has not been trained yet. Cannot save an untrained forest.")

        os.makedirs(filepath, exist_ok=True)
        # Save forest configuration (number of trees, depth, leaf params)
        forest_config = {
            'num_trees': self.num_trees,
            'tree_depth': self.tree_depth,
            'input_dim': self.input_dim,
            'leaf_nn_config': self.leaf_nn_params
        }
        import json
        with open(os.path.join(filepath, 'forest_config.json'), 'w') as f:
            json.dump(forest_config, f)

        # Save each tree individually
        tree_save_dir = os.path.join(filepath, 'trees')
        os.makedirs(tree_save_dir, exist_ok=True)
        for i, tree in enumerate(self.trees):
            tree_path = os.path.join(tree_save_dir, f'tree_{i}')
            tree.save(tree_path, save_format='tf') # Use TensorFlow SavedModel format

        print(f"Neural Forest successfully saved to {filepath}")


    @classmethod
    def load(cls, filepath):
        """Loads a neural forest from a saved directory."""
        import json
        config_path = os.path.join(filepath, 'forest_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Forest configuration file not found at {config_path}")

        with open(config_path, 'r') as f:
            forest_config = json.load(f)

        # Create an instance of the NeuralForest using the loaded config
        # We pass dummy values for num_trees and input_dim initially as trees list will be populated later
        # The actual number of trees will be determined by loaded trees.
        loaded_forest = cls(
            num_trees=forest_config['num_trees'], # Initialize with expected num_trees
            tree_depth_val=forest_config['tree_depth'],
            input_dim=forest_config['input_dim'],
            leaf_nn_config=forest_config['leaf_nn_config']
        )

        # Load each tree individually
        tree_load_dir = os.path.join(filepath, 'trees')
        if not os.path.exists(tree_load_dir):
             raise FileNotFoundError(f"Tree directory not found at {tree_load_dir}")

        loaded_trees = []
        # List directories inside tree_load_dir, assuming each is a saved tree
        tree_dirs = sorted([d for d in os.listdir(tree_load_dir) if os.path.isdir(os.path.join(tree_load_dir, d))])

        if len(tree_dirs) != loaded_forest.num_trees:
             print(f"Warning: Expected {loaded_forest.num_trees} trees based on config, but found {len(tree_dirs)} directories in {tree_load_dir}. Loading found trees.")
             loaded_forest.num_trees = len(tree_dirs) # Adjust num_trees if mismatch

        custom_objects = {
            'LeafNetwork': LeafNetwork,
            'DecisionNode': DecisionNode,
            'NeuralDecisionTree': NeuralDecisionTree
            # Add other custom objects if you introduce more
        }

        for tree_dir in tree_dirs:
            tree_path = os.path.join(tree_load_dir, tree_dir)
            # Load the tree, providing custom objects
            loaded_tree = keras.models.load_model(tree_path, custom_objects=custom_objects)
            loaded_trees.append(loaded_tree)

        loaded_forest.trees = loaded_trees
        print(f"Neural Forest successfully loaded from {filepath}")

        return loaded_forest

# Define custom objects for Keras loading
custom_objects_for_loading = {
    'LeafNetwork': LeafNetwork,
    'DecisionNode': DecisionNode,
    'NeuralDecisionTree': NeuralDecisionTree
}
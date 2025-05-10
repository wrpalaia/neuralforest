import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
# import pandas as pd # Kept if needed for more complex data loading in future
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.utils import resample # For bagging

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
# Common Tree Parameters (will be used by each tree in the forest)
tree_depth = 3
leaf_nn_hidden_layers = [8]
leaf_nn_activation = 'relu'
output_dim = 1  # Output dimension (1 for binary classification/regression)
# Determine task type based on output_activation
leaf_nn_output_activation = 'sigmoid'  # 'sigmoid' for binary, 'softmax' for multi-class, 'linear' for regression

# Forest Parameters
num_trees_in_forest = 10
epochs_per_tree = 20 # Epochs for training each individual tree in the forest
batch_size = 64

# Determine loss function based on task
if leaf_nn_output_activation == 'sigmoid':
    loss_function = 'binary_crossentropy'
    metrics_list = ['accuracy', tf.keras.metrics.AUC(name='auc')]
elif leaf_nn_output_activation == 'softmax':
    loss_function = 'categorical_crossentropy'
    metrics_list = ['accuracy', tf.keras.metrics.AUC(name='auc')] # AUC might need one-hot y_true for categorical
elif leaf_nn_output_activation == 'linear':
    loss_function = 'mean_squared_error'
    metrics_list = ['mean_absolute_error']
else:
    raise ValueError(f"Unsupported leaf_nn_output_activation: {leaf_nn_output_activation}")

optimizer_name = 'adam' # Optimizer for each tree

# --- Data Generation (Example) ---
X, y_original = make_classification(n_samples=2000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
input_dim = X.shape[1]

# Reshape y and prepare for the task
if leaf_nn_output_activation == 'sigmoid': # Binary classification
    y = y_original.reshape(-1, 1)
elif leaf_nn_output_activation == 'softmax': # Multi-class
    num_classes = len(np.unique(y_original))
    output_dim = num_classes # Override output_dim for softmax
    y = keras.utils.to_categorical(y_original, num_classes=num_classes)
elif leaf_nn_output_activation == 'linear': # Regression (if y_original was regression data)
    y = y_original.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- Neural Decision Tree Components (Identical to refined single tree) ---

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
        self.router = layers.Dense(1, activation='sigmoid', name="router_dense")

    def call(self, inputs):
        return self.router(inputs)

class NeuralDecisionTree(keras.Model):
    """A single Neural Decision Tree with differentiable routing and leaf networks."""
    def __init__(self, depth, current_input_dim, leaf_nn_params_dict, name="neural_decision_tree", **kwargs): # Renamed input_dim to current_input_dim
        super(NeuralDecisionTree, self).__init__(name=name, **kwargs)
        self.depth = depth
        self.num_leaves = 2**depth
        self.current_input_dim = current_input_dim # Store the input dim this tree expects
        self.leaf_nn_params = leaf_nn_params_dict

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
        current_level_path_probabilities = tf.ones((batch_size, 1))
        decision_node_idx_offset = 0

        for level in range(self.depth):
            num_nodes_in_current_level = 2**level
            next_level_path_probabilities_list = []
            for i in range(num_nodes_in_current_level):
                prob_reaching_parent_node = current_level_path_probabilities[:, i:i+1]
                decision_node = self.decision_nodes[decision_node_idx_offset + i]
                routing_prob_to_left = decision_node(inputs)
                routing_prob_to_right = 1.0 - routing_prob_to_left
                prob_to_left_child = prob_reaching_parent_node * routing_prob_to_left
                prob_to_right_child = prob_reaching_parent_node * routing_prob_to_right
                next_level_path_probabilities_list.extend([prob_to_left_child, prob_to_right_child])
            current_level_path_probabilities = tf.concat(next_level_path_probabilities_list, axis=1)
            decision_node_idx_offset += num_nodes_in_current_level
            
        path_probabilities_to_leaves = current_level_path_probabilities
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
            "current_input_dim": self.current_input_dim,
            "leaf_nn_params_dict": self.leaf_nn_params, # Ensure this is serializable
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Leaf_nn_params might need careful handling if it contains custom objects not default-Keras serializable
        # For this structure, it should be fine as it's a dict of basic types.
        return cls(**config)

# --- Neural Forest Class ---
class NeuralForest:
    """Manages an ensemble of NeuralDecisionTree models."""
    def __init__(self, num_trees, tree_depth_val, overall_input_dim, leaf_nn_config): # Renamed params for clarity
        self.num_trees = num_trees
        self.tree_depth = tree_depth_val
        self.input_dim = overall_input_dim # Input dimension for the dataset
        self.leaf_nn_params = leaf_nn_config
        self.trees = []
        self.history_per_tree = [] # Optional: to store training history of each tree

    def fit(self, X_data, y_data, epochs, tree_batch_size, validation_split_per_tree=0.1, early_stopping_patience=5, forest_verbose=1, tree_training_verbose=0):
        self.trees = []
        self.history_per_tree = []

        for i in range(self.num_trees):
            if forest_verbose > 0:
                print(f"\nTraining Tree {i+1}/{self.num_trees}...")

            # Bagging: Create a bootstrap sample
            X_sample, y_sample = resample(X_data, y_data, random_state=np.random.randint(10000)) # Different random state for each bag

            tree = NeuralDecisionTree(
                depth=self.tree_depth,
                current_input_dim=self.input_dim, # Each tree sees all features of the bagged sample
                leaf_nn_params_dict=self.leaf_nn_params,
                name=f"neural_tree_{i}"
            )

            # Build and compile the tree
            tree.build(input_shape=(None, self.input_dim))
            tree.compile(
                optimizer=tf.keras.optimizers.Adam(), # Create a new optimizer instance for each tree
                loss=loss_function, # Use global loss_function
                metrics=metrics_list # Use global metrics_list
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
            if forest_verbose > 0 and early_stopping_patience > 0 and validation_split_per_tree > 0 and early_stop_cb.stopped_epoch > 0 :
                 print(f"Tree {i+1} stopped early at epoch {early_stop_cb.stopped_epoch + 1}.")
        if forest_verbose > 0:
            print("\nNeural Forest training complete.")

    def predict_proba(self, X_data):
        if not self.trees:
            raise ValueError("The forest has not been trained yet or no trees were trained.")
        
        all_tree_predictions = []
        for tree in self.trees:
            all_tree_predictions.append(tree.predict(X_data))
        
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


# --- Main Execution for Neural Forest ---
if __name__ == "__main__":
    # Define parameters for each tree's leaf networks
    leaf_nn_parameters_config = {
        'hidden_layers': leaf_nn_hidden_layers,
        'activation': leaf_nn_activation,
        'output_dim': output_dim, # This might be overridden for softmax
        'output_activation': leaf_nn_output_activation
    }
    # Adjust output_dim for softmax if it was changed during data prep
    if leaf_nn_output_activation == 'softmax':
        leaf_nn_parameters_config['output_dim'] = y_train.shape[1]


    # Create the Neural Forest
    print("--- Neural Forest Initialization ---")
    neural_forest = NeuralForest(
        num_trees=num_trees_in_forest,
        tree_depth_val=tree_depth,
        overall_input_dim=input_dim,
        leaf_nn_config=leaf_nn_parameters_config
    )

    # Train the Neural Forest
    print("\n--- Neural Forest Training ---")
    neural_forest.fit(
        X_train,
        y_train,
        epochs=epochs_per_tree,
        tree_batch_size=batch_size,
        validation_split_per_tree=0.1, # Use 10% of each bootstrap sample for validation
        early_stopping_patience=5,     # Patience for early stopping for each tree
        forest_verbose=1,              # Print progress for the forest
        tree_training_verbose=1        # Keras verbosity for individual tree training (0 for silent, 1 for progress bar)
    )

    # Make predictions on the test set
    print("\n--- Neural Forest Prediction & Evaluation ---")
    y_pred_proba_forest = neural_forest.predict_proba(X_test)

    if leaf_nn_output_activation == 'sigmoid': # Binary classification
        y_pred_labels_forest = (y_pred_proba_forest > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred_labels_forest)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba_forest)
            print(f"Forest Test Accuracy: {accuracy:.4f}")
            print(f"Forest Test ROC AUC: {roc_auc:.4f}")
        except ValueError as e: # Handle cases like y_test having only one class after split
            print(f"Forest Test Accuracy: {accuracy:.4f}")
            print(f"Could not calculate ROC AUC: {e}")

    elif leaf_nn_output_activation == 'softmax': # Multi-class classification
        y_pred_labels_forest = np.argmax(y_pred_proba_forest, axis=1)
        y_test_labels = np.argmax(y_test, axis=1) # Assuming y_test is one-hot encoded
        accuracy = accuracy_score(y_test_labels, y_pred_labels_forest)
        # ROC AUC for multi-class is more complex (e.g., OvR or OvO)
        # For simplicity, just reporting accuracy here.
        print(f"Forest Test Accuracy: {accuracy:.4f}")
        # Example for multi-class AUC if needed (requires y_test to be probabilities or one-hot)
        # from sklearn.preprocessing import LabelBinarizer
        # lb = LabelBinarizer()
        # lb.fit(y_test_labels)
        # y_test_binarized = lb.transform(y_test_labels)
        # if y_test_binarized.shape[1] > 1: # Check if truly multi-class
        #     roc_auc = roc_auc_score(y_test_binarized, y_pred_proba_forest, multi_class='ovr')
        #     print(f"Forest Test ROC AUC (OvR): {roc_auc:.4f}")

    elif leaf_nn_output_activation == 'linear': # Regression
        # y_pred_proba_forest already contains the regression values
        mse = mean_squared_error(y_test, y_pred_proba_forest)
        r2 = r2_score(y_test, y_pred_proba_forest)
        print(f"Forest Test Mean Squared Error: {mse:.4f}")
        print(f"Forest Test R-squared: {r2:.4f}")

    # print("\nExample Forest Predictions (probabilities/values):")
    # print(y_pred_proba_forest[:5])
    # if leaf_nn_output_activation != 'linear':
    #     print("\nExample Forest Predictions (labels):")
    #     print(y_pred_labels_forest[:5])
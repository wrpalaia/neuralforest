import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Removed: from sklearn.utils import resample (not needed for a single tree without bagging)
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration for a Single Neural Decision Tree ---
tree_depth = 3
num_leaves = 2**tree_depth
leaf_nn_hidden_layers = [8]
leaf_nn_activation = 'relu'
output_dim = 1 # For binary classification
leaf_nn_output_activation = 'sigmoid' # For binary classification
loss_function = 'binary_crossentropy'
optimizer = 'adam'
epochs_for_tree = 50 # Renamed from epochs_per_tree
batch_size = 64

# --- Data Generation (Example from your script) ---
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
input_dim = X.shape[1] # Input dimension for the single tree

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Neural Decision Tree Components (LeafNetwork, DecisionNode, NeuralDecisionTree classes remain as they are) ---
# (Your LeafNetwork, DecisionNode, and NeuralDecisionTree class definitions go here - unchanged from nforest3.py)

class LeafNetwork(keras.Model): #
    """A small neural network at the leaf node.""" #
    def __init__(self, hidden_layers, output_dim, activation, output_activation): #
        super(LeafNetwork, self).__init__() #
        self.hidden_layers = [] #
        for size in hidden_layers: #
            self.hidden_layers.append(layers.Dense(size, activation=activation)) #
        self.output_layer = layers.Dense(output_dim, activation=output_activation) #

    def call(self, inputs): #
        x = inputs #
        for layer in self.hidden_layers: #
            x = layer(x) #
        return self.output_layer(x) #

class DecisionNode(layers.Layer): #
    """A differentiable decision node for routing.""" #
    def __init__(self, **kwargs): #
        super(DecisionNode, self).__init__(**kwargs) #
        self.router = layers.Dense(1, activation='sigmoid') #

    def call(self, inputs): #
        return self.router(inputs) #

class NeuralDecisionTree(keras.Model): #
    """A single Neural Decision Tree with differentiable routing and leaf networks.""" #
    def __init__(self, depth, input_dim, leaf_nn_params, **kwargs): #
        super(NeuralDecisionTree, self).__init__(**kwargs) #
        self.depth = depth #
        self.num_leaves = 2**depth #
        self.input_dim = input_dim #
        self.leaf_nn_params = leaf_nn_params #

        self.decision_nodes = [] #
        for _ in range(self.num_leaves // 2): #
            self.decision_nodes.append(DecisionNode()) #

        self.leaf_networks = [] #
        for _ in range(self.num_leaves): #
            self.leaf_networks.append(LeafNetwork( #
                hidden_layers=self.leaf_nn_params['hidden_layers'], #
                output_dim=self.leaf_nn_params['output_dim'], #
                activation=self.leaf_nn_params['activation'], #
                output_activation=self.leaf_nn_params['output_activation'] #
            )) #

    def call(self, inputs): #
        batch_size = tf.shape(inputs)[0] #
        node_probabilities = tf.ones((batch_size, 1)) #
        leaf_probabilities = [] #

        node_index = 0 #
        for level in range(self.depth): #
            new_node_probabilities = [] #
            num_nodes_in_level = 2**level #
            for i in range(num_nodes_in_level): #
                if level < self.depth: #
                    routing_prob_left = self.decision_nodes[node_index + i](inputs) #
                    routing_prob_right = 1.0 - routing_prob_left #

                    prob_to_left_child = node_probabilities[:, i:i+1] * routing_prob_left #
                    prob_to_right_child = node_probabilities[:, i:i+1] * routing_prob_right #

                    new_node_probabilities.append(prob_to_left_child) #
                    new_node_probabilities.append(prob_to_right_child) #
                else: #
                    leaf_probabilities.append(node_probabilities[:, i:i+1]) #

            node_probabilities = tf.concat(new_node_probabilities, axis=1) #
            node_index += num_nodes_in_level #
        
        leaf_probabilities = node_probabilities #

        leaf_predictions = [leaf_nn(inputs) for leaf_nn in self.leaf_networks] #

        leaf_probabilities = tf.stack(leaf_probabilities, axis=1) #
        leaf_predictions = tf.stack(leaf_predictions, axis=1) #
        
        if self.leaf_nn_params['output_activation'] == 'sigmoid' or self.leaf_nn_params['output_activation'] == 'softmax': #
            averaged_predictions = tf.reduce_sum(leaf_probabilities * leaf_predictions, axis=1) #
        else: #
            averaged_predictions = tf.reduce_sum(leaf_probabilities * leaf_predictions, axis=1) #

        return averaged_predictions #


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
        input_dim=input_dim, # Use the full input dimension
        leaf_nn_params=leaf_nn_parameters
    )

    # Build the model
    single_neural_tree.build(input_shape=(None, input_dim)) #
    # single_neural_tree.summary() # Optional: to see the structure

    # Compile the tree
    single_neural_tree.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy'] # Add other metrics as needed
    )

    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) #

    # Train the Neural Decision Tree
    print("Training the Neural Decision Tree...")
    history = single_neural_tree.fit(
        X_train, # Use the full training data
        y_train,
        epochs=epochs_for_tree,
        batch_size=batch_size,
        validation_split=0.2, # Use a portion of the training data for validation
        callbacks=[early_stopping],
        verbose=1 # Set to 1 to see training progress
    )

    # Make predictions on the test set
    print("\nMaking predictions...")
    y_pred_proba_single_tree = single_neural_tree.predict(X_test)

    # Convert probabilities to class labels for classification
    if leaf_nn_output_activation == 'sigmoid': # Binary classification
        y_pred_single_tree = (y_pred_proba_single_tree > 0.5).astype(int)
    elif leaf_nn_output_activation == 'softmax': # Multi-class classification
        y_pred_single_tree = np.argmax(y_pred_proba_single_tree, axis=1)
    else: # Regression
        y_pred_single_tree = y_pred_proba_single_tree


    # Evaluate the Neural Tree (Example for binary classification)
    from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

    print("\n--- Single Neural Tree Evaluation Results ---")
    if leaf_nn_output_activation == 'sigmoid': #
        accuracy = accuracy_score(y_test, y_pred_single_tree) #
        roc_auc = roc_auc_score(y_test, y_pred_proba_single_tree) #
        print(f"Accuracy: {accuracy:.4f}") #
        print(f"ROC AUC: {roc_auc:.4f}") #
    elif leaf_nn_output_activation == 'linear': # Regression
        mse = mean_squared_error(y_test, y_pred_single_tree) #
        r2 = r2_score(y_test, y_pred_single_tree) #
        print(f"Mean Squared Error: {mse:.4f}") #
        print(f"R-squared: {r2:.4f}") #
    # Add evaluation for multi-class classification if needed

    # --- Deployment Considerations ---
    # To prepare for deployment, you would typically save the trained model:
    # single_neural_tree.save("single_neural_tree_model")
    # print("\nSingle Neural Tree model saved.")

    # When deploying, you would load the model and use it for predictions:
    # loaded_model = keras.models.load_model("single_neural_tree_model")
    # example_prediction = loaded_model.predict(new_data)
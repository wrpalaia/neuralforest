# neural_forest/__init__.py

# Expose the main classes for easier access
from .model import NeuralForest, NeuralDecisionTree, LeafNetwork, DecisionNode

# You can also define __version__ here
__version__ = "0.1.0" # Or load from a file

# Expose custom objects dictionary if users might need to load NeuralDecisionTree directly
from .model import custom_objects_for_keras_load_model

__all__ = [
    "NeuralForest",
    "NeuralDecisionTree",
    "LeafNetwork",
    "DecisionNode",
    "custom_objects_for_keras_load_model",
    "__version__"
]
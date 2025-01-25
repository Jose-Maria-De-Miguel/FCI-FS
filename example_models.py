# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 13:38:15 2024

@author: chdem
"""

from pgmpy.utils import get_example_model

# List of available example models
example_models = [
   'alarm',
    'asia',
    'barley',
   # 'breast_cancer',
 #   'car',
    'child',
    'sachs',
    'survey',
    'water',
]

print("Available example models:")
for model_name in example_models:
    print(f"- {model_name}")
    model = get_example_model(model_name)  # Load the model by its name
    print("Model nodes:", len(model.nodes()))

# Load a specific example model, for instance 'sachs'


# Display the model's structure (nodes and edges)

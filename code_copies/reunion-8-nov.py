# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 08:27:01 2024

@author: chdem
"""

from graphviz import Digraph

# Initialize a directed graph
graph = Digraph(format='png')

# Add 5 vertices
vertices = ['X3', 'X1', 'X2', 'X4']
for v in vertices:
    graph.node(v)

# Add edges to the graph (customize these as desired)
edges = [
    ('X1', 'X3'),

    ('X1', 'X4'),
    ('X3', 'X5'),
    ('X3', 'X4'),
    ('X2', 'X3'),
    

      # adding a cycle for example purposes
]
for edge in edges:
    graph.edge(*edge)

# Render the graph
graph.render('directed_graph1')

# To visualize in a Jupyter notebook (if using one), use:
# graph.display()

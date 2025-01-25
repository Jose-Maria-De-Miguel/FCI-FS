# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:55:31 2025

@author: chdem
"""

from pgmpy.models import BayesianNetwork
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx
import PDAG

def bnToPDAG(bn_model):
    
    return PDAG.PDAG(v_names =[name for name in bn_model.nodes()], d_edges= bn_model.edges())

def plotPGMpy(bn_model, file_path = f".//test_images//trial_image.png"):
    
    model_graphviz = bn_model.to_graphviz()
    # Set the engine to 'neato' using graph_attr

    model_graphviz.draw(file_path, prog="dot")
    
    img = mpimg.imread(file_path)  # Load the PNG file
    plt.imshow(img)  # Display the image
    plt.axis('off')  # Turn off the axis for better presentation
    plt.show() 

model = BayesianNetwork.get_random(n_nodes=5, edge_prob=2/5)
# Convert model into pygraphviz object


model_PDAG = bnToPDAG(model)

model_PDAG.plot()

plotPGMpy(model)

 



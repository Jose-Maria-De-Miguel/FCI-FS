# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 19:26:47 2025

@author: chdem
"""

from pgmpy.models import BayesianNetwork

from pgmpy.estimators import PC
import PCFS
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PDAG
import random
import csv


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
    
def random_var_order(data):
    """
    

    Parameters
    ----------
    data : TYPE
    
    Change the order or the columns(variables) of the dataset randomly.
    PRE: Two columns can't have the same name.

    Returns
    -------
    None.

    """
    
    
    
    """
    
    
    """
    
    original_columns = list(data.columns)
    new_columns = []
    
    
    while  len(original_columns)>0:
        #Genera un entero aleatorio entre 0 y len-1
        colum = original_columns[int(len(original_columns)*random.random())]
        new_columns.append(colum)
        original_columns.remove(colum)
    
    data = data[new_columns]
    
def extract_random(data, percentage = 0.5):
    """
    

    Parameters
    ----------
    data 
    percentage : float, 
        Defaults 0.1
        
    ----------
    Description
    ----------
    Extract variable names randomly. The number of variables extracted is the lowest integer closer 
    to the percentage of variables given by the parameter percentage . The default is 0.1.
    
    -------
    Returns
    -------
    newVars :
        Names of the randomlyChosen set of variables.
    initialVars : 
        Names of the remaining variables.

    """
    
    initialVars = list(data.columns)
    n = int(percentage*len(initialVars))
    newVars = []
    
    while len(newVars) < n:
        name = initialVars.pop(int(len(initialVars)*random.random()))
        newVars.append(name)
        
        

    return newVars, initialVars
    
    
    
    
INITIAL_P = 0.05

NUM_VARS = 500

NEIGHBORHOOD_SIZE = 2


NUM_INSTANCES = 500


NUM_PVALS = 6

NUM_RANDOM_DAGS = 1

NUM_ORDERS = 1

NUM_PERCENTAGE = 10

for k in range(0, NUM_PVALS):
    p = INITIAL_P/(2**k)
    with open(f"output{k}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        #Write a header row
        colum_names = ['TP_orig', 'FP_orig', 'TN_orig', 'FN_orig', 'TD_orig', 'FD_orig', 'P_orig', 'R_orig', 'F1_orig', 'SHD_orig', 'TP_stable', 'FP_stable', 'TN_stable', 'FN_stable', 'TD_stable', 'FD_stable', 'P_stable', 'R_stable', 'F1_stable', 'SHD_stable']
        FS_cols = [name for percentage in range(0, 3*5, 5)  for name in [f'TP_{percentage}Percent', f'FP_{percentage}Percent', f'TN_{percentage}Percent', f'FN_{percentage}Percent', f'TD_{percentage}Percent', f'FD_{percentage}Percent', f'P_{percentage}Percent', f'R_{percentage}Percent', f'F1_{percentage}Percent', f'SHD_{percentage}Percent']]
        colum_names = colum_names + FS_cols
        writer.writerow(colum_names)
        for j in range(0, NUM_RANDOM_DAGS):
            #Generate random DAG
            model = BayesianNetwork.get_random(n_nodes=NUM_VARS, edge_prob=NEIGHBORHOOD_SIZE/(NUM_VARS-1))
            
            PDAG_base = bnToPDAG(model)
            
            # Simulate data
            data = model.simulate(n_samples=NUM_INSTANCES)
            variables = list(data.columns)
            estFS = PCFS(data) #PC-FS
            
            PDAG_stable =  estFS.estimate(ci_test='chi_square', max_cond_vars=3,  significance_level= p, new_vars = variables)
            
            for i in range(0, NUM_ORDERS):
                random_var_order(data)
                
                estFS = PCFS(data) #PC-FS
                
                estPC = PC(data) # PC from PGMpy
                
                model_orig = estPC.estimate(variant = "orig", ci_test='chi_square', max_cond_vars=2,  significance_level= p )
              
                PDAG_orig = bnToPDAG(model_orig) # TODO: esto se puede hacer??
                
                PDAG_orig.plot()
                
                for n in range(0, NUM_PERCENTAGE):
                
                    percentage = 0.05*n
                    
                    newVars, initialVars = extract_random(data, percentage = percentage)
                    
                    PDAG_initial = estFS.estimate(ci_test='chi_square', max_cond_vars=3,  significance_level=p, new_vars = initialVars)
                    
                    
                    PDAG_FS = estFS.estimate(ci_test='chi_square', max_cond_vars=3, pdag = PDAG_initial,   significance_level=p, new_vars = newVars)
                    
                    
                
                    
                    
                
                
            
        
        
        

    
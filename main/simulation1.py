# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 19:26:47 2025

@author: chdem
"""

from pgmpy.models import BayesianNetwork





from  PCFS import PCFS
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PDAG
import random
import csv
import numpy as np
import time


def bnToPDAG(bn_model):
    """
    This procedure transforms a PGMpy Bayesian network into the class PDAG.
    This method works under the assumption that the bn_model is  is fully directed, meaning it does not contain any undirected or bidirected edges.

    """
    
    return PDAG.PDAG(v_names =[name for name in bn_model.nodes()], d_edges= bn_model.edges())

def plotPGMpy(bn_model, file_path = f".//test_images//trial_image.png"):
    """
    This method plots bn_model a PGMpy's Bayesian network, and stores the result in the indicated file_path.

    """
    model_graphviz = bn_model.to_graphviz()
    # Set the engine to 'neato' using graph_attr

    model_graphviz.draw(file_path, prog="dot")
    
    img = mpimg.imread(file_path)  # Load the PNG file
    plt.imshow(img)  # Display the image
    plt.axis('off')  # Turn off the axis for better presentation
    plt.show() 
    
def random_var_order(data):
    """
    Change the order or the columns(variables) of the dataset randomly.
    PRE: Two columns can't have the same name.

    """
    
    original_columns = list(data.columns)
    new_columns = []
    
    
    while  len(original_columns)>0:
        #Genera un entero aleatorio entre 0 y len-1
        colum = original_columns[int(len(original_columns)*random.random())]
        new_columns.append(colum)
        original_columns.remove(colum)
    
    return data[new_columns]
    
def extract_random(data, percentage = 0.5):
    """
    

    Parameters
    ----------
    data 
    percentage : float, 
        Defaults 0.5
        
    ----------
    Description
    ----------
    Extract variable names randomly. The number of variables extracted is the lowest integer closer 
    to the percentage of variables given by the parameter percentage . The default is 0.5.
    
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

NUM_VARS = 20

NEIGHBORHOOD_SIZE = 2


NUM_INSTANCES = 500

NUM_PVALS = 1

NUM_RANDOM_DAGS = 1

NUM_ORDERS = 1

NUM_PERCENTAGE = 10

PERCENT_STEP = 0.05

percentList = [round(k*PERCENT_STEP, 2) for k in range(NUM_PERCENTAGE) ]

for k in range(0, NUM_PVALS):
    p = INITIAL_P/(2**k)
    with open(f"output{k}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        #Write a header row
        colum_names = ['TP_orig', 'FP_orig', 'TN_orig', 'FN_orig', 'TO_orig', 'FO_orig', 'TD_orig', 'P_orig', 'R_orig', 'F1_orig', 'SHD_orig','numCITest_orig', 'time_orig', 'TP_stable', 'FP_stable', 'TN_stable', 'FN_stable', 'TO_stable', 'FO_stable', 'TD_stable', 'P_stable', 'R_stable', 'F1_stable', 'SHD_stable', 'numCITest_stable', 'time_stable']
        FS_cols = [name for percentage in percentList  for name in [f'TP_{int(percentage*100)}Percent', 
        f'FP_{int(percentage*100)}Percent', f'TN_{int(percentage*100)}Percent', f'FN_{int(percentage*100)}Percent', 
        f'TO_{int(percentage*100)}Percent', f'FO_{int(percentage*100)}Percent', f'TD_{int(percentage*100)}Percent',
        f'P_{int(percentage*100)}Percent', f'R_{int(percentage*100)}Percent', f'F1_{int(percentage*100)}Percent', 
        f'SHD_{int(percentage*100)}Percent', f'numCITestMarginal__{int(percentage*100)}Percent', f'timeMarginal__{int(percentage*100)}Percent', f'numCITestFS__{int(percentage*100)}Percent', f'timeFS__{int(percentage*100)}Percent']]
        colum_names = colum_names + FS_cols
        writer.writerow(colum_names)
        for j in range(0, NUM_RANDOM_DAGS):
            
            n_states = np.random.randint(low=3, high=6, size=NUM_VARS)
            #Generate random DAG
            model = BayesianNetwork.get_random(n_nodes=NUM_VARS, edge_prob=NEIGHBORHOOD_SIZE/(NUM_VARS-1), n_states = n_states)
            
            PDAG_base = bnToPDAG(model)
            PDAG_base.simplifyPDAG()
            PDAG_base.applyMeek()
            
            # Simulate data
            
            
            data = model.simulate(n_samples=NUM_INSTANCES)
            variables = list(data.columns)
            estFS = PCFS(data) #PC-FS
            print("\nMPC-Stable\n")
            print(f"\nSIMULATE DATA: {j}:{NUM_RANDOM_DAGS} - {k}:{NUM_PVALS} \n")

            
            timeStable0 = time.time()
            
            PDAG_stable, nCITestStable =  estFS.estimate(variant = "MPC-stable", ci_test='chi_square', max_cond_vars=3,  significance_level= p, new_vars = variables)
            
            timeStable1 = time.time()
            
            totalTimeStable = timeStable1 - timeStable0
            
            stable_metrics = PDAG_stable.getGraphicalMetrics(PDAG_base)
            
            stable_metrics =  stable_metrics + [nCITestStable, totalTimeStable ]
            
            
            print("Stable learned")
            for i in range(0, NUM_ORDERS):
                data = random_var_order(data) 
                
                estFS = PCFS(data) #PC-FS
                variables = list(data.columns)
                
                timeOrig0 = time.time()
                
                PDAG_orig, numCIOrig =  estFS.estimate(variant = "orig", ci_test='chi_square', max_cond_vars=3,  significance_level= p, new_vars = variables)
                
                timeOrig1 = time.time()
                
                totalTimeOrig = timeOrig1 - timeOrig0
                
                
                orig_metrics = PDAG_orig.getGraphicalMetrics(PDAG_base)
                
                orig_metrics = orig_metrics + [numCIOrig, totalTimeOrig]
                
                fs_metrics = []
                
                for percentage in percentList :
                
                    newVars, initialVars = extract_random(data, percentage = percentage)
                    
                    timeMarginal0 = time.time()
                    
                    PDAG_marginal, numCIMarginal = estFS.estimate(variant = "MPC-stable", ci_test='chi_square', max_cond_vars=3,  significance_level=p, new_vars = initialVars)
                   
                    timeMarginal1 = time.time()
                   
                    totalTimeMarginal = timeMarginal1 - timeMarginal0
                    
                    timeFS0 = time.time()
                    
                    PDAG_FS, numCIFS = estFS.estimate(variant = "MPC-stable", ci_test='chi_square', max_cond_vars=3, pdag = PDAG_marginal,   significance_level=p, new_vars = newVars)
                    
                    timeFS1 = time.time()
                    
                    totalTimeFS = timeFS1 - timeFS0
                    
                    
                    
                    fs_metrics = fs_metrics + PDAG_FS.getGraphicalMetrics(PDAG_base) + [numCIMarginal, totalTimeMarginal, numCIFS, totalTimeFS]
                
                
                newRow = orig_metrics + stable_metrics + fs_metrics
                
                writer.writerow(newRow)

print("Finished")
                    
                
                    
                    
                
                
            
        
        
        

    
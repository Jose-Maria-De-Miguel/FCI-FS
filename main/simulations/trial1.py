import pandas as pd
import random

import time
from  PC import PC

def random_order(data):
    #Change the order or the columns of data randomly.
    #PRE: Two columns can't have the same name.
    
    original_columns = list(data.columns)
    new_columns = []
    
    
    while  len(original_columns)>0:
        #Genera un entero aleatorio entre 0 y len-1
        colum = original_columns[int(len(original_columns)*random.random())]
        new_columns.append(colum)
        original_columns.remove(colum)
    return new_columns

def extract_random(data, percentage = 0.1):
    
    
    initialVars = list(data.columns)
    n = int(percentage*len(initialVars))
    newVars = []
    
    while len(newVars) < n:
        name = initialVars.pop(int(len(initialVars)*random.random()))
        newVars.append(name)
        
        

    return newVars, initialVars



exec1, exec2, exec3 = 0, 0, 0
check = 0




data = pd.read_csv("..//child_model_data50k.csv")

order = random_order(data)
data = data[order]

est = PC(data)

time0 = time.time()

graph_pc_normal = est.estimate(ci_test='chi_square', max_cond_vars=3,  significance_level= .1, new_vars = order)

time1 = time.time()

exec_time_pc_estable_20vars = time1 - time0


for i in range(1, 11):
    order = random_order(data)
    data = data[order]
    
    
    newVars, initialVars = extract_random(data, percentage = 0.20)
    
    est = PC(data)
    
    
    exec1 = exec1 + exec_time_pc_estable_20vars
    time1 = time.time()
   
    
    graph_chi = est.estimate(ci_test='chi_square', max_cond_vars=3,  significance_level=.1, new_vars = initialVars)
    
    
    time2 = time.time()
    
    exec_time_pc_estable_10vars = time2 - time1
    
    exec2 = exec2 + exec_time_pc_estable_10vars
    
  
    
    graph_chi2 = est.estimate(ci_test='chi_square', max_cond_vars=3, pdag = graph_chi,   significance_level=.1, new_vars = newVars)
    
    time3 = time.time() 
    
    exec_time_add_10_vars = time3 - time2
    
    exec3 = exec3 + exec_time_add_10_vars
    
   
    
    
    graph_chi2.plot()
    
    check = check + (graph_pc_normal == graph_chi2)
    
    print(f"\n \nNum. pruebas: {i}. Num. aciertos: {check}. Prob. Acierto: {check/i}. \n t. medio pc_normal: {exec1/i}.  t. medio pc_reducido: {exec2/i}. t. medio pc_aumento: {exec3/i}. \n \n") 
    
    print(f"Variables incrementales. \n Orden: {order} \n \tNuevas: {newVars}. \n \tIniciales{initialVars}.")
    print(f"Tienen el mismo esqueleto: {graph_pc_normal == graph_chi2}. \nDifferences: {graph_chi2.get_differences(graph_pc_normal)}" )



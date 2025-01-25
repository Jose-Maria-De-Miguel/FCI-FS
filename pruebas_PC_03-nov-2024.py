# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:27:53 2024

@author: chdem
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import networkx as nx

import random
#from pgmpy.estimators import PC
from pgmpy.utils import get_example_model
# Load the example model
model = get_example_model('earthquake')


"""
# Add the nodes 'TV' and 'Nap'
model.add_nodes_from(['TV', 'Nap'])

# Add edges from 'TV' to 'JohnCalls' and from 'Nap' to 'MaryCalls'
model.add_edges_from([('TV', 'JohnCalls'), ('Nap', 'MaryCalls')])

# Verify state names of Alarm
alarm_state_names = model.get_cpds('Alarm').state_names
print("State names of Alarm:", alarm_state_names)

# Map states ('True', 'False') to indices (0, 1)
state_mapping = {'True': 0, 'False': 1}

# Define CPDs for the new nodes
cpd_tv = TabularCPD(variable='TV', variable_card=2, values=[[0.7], [0.3]])  # P(TV='True') = 0.7, P(TV='False') = 0.3
cpd_nap = TabularCPD(variable='Nap', variable_card=2, values=[[0.6], [0.4]])  # P(Nap='True') = 0.6, P(Nap='False') = 0.4

# Define CPD for JohnCalls considering TV
cpd_johncalls = TabularCPD(
    variable='JohnCalls',
    variable_card=2,
    values=[
        [0.9, 0.7, 0.6, 0.2],  # P(JohnCalls='True' | Alarm='True', TV='True')
        [0.1, 0.3, 0.4, 0.8],  # P(JohnCalls='False' | Alarm='True', TV='True')
    ],
    evidence=['Alarm', 'TV'],  # Correct parents for JohnCalls
    evidence_card=[2, 2]       # Alarm and TV both have 2 states
)

# Define CPD for MaryCalls considering Nap
cpd_marycalls = TabularCPD(
    variable='MaryCalls',
    variable_card=2,
    values=[
        [0.8, 0.6, 0.4, 0.2],  # P(MaryCalls='True' | Alarm='True', Nap='True')
        [0.2, 0.4, 0.6, 0.8],  # P(MaryCalls='False' | Alarm='True', Nap='True')
    ],
    evidence=['Alarm', 'Nap'],  # Correct parents for MaryCalls
    evidence_card=[2, 2]        # Alarm and Nap both have 2 states
)

# Add the CPDs to the model
model.add_cpds(cpd_tv, cpd_nap, cpd_johncalls, cpd_marycalls)

# Validate the model
assert model.check_model(), "Model validation failed!"
"""
# Simulate data
data = model.simulate(n_samples=10000)

data.to_csv('earthquake10k.csv', index=False)

A = nx.nx_agraph.to_agraph(model)

A.draw('.//imagenes//model_graphviz.png', format='png', prog='dot') 

# Validate the model
assert model.check_model()
def gen_aleatorio(data):
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


#data = pd.read_csv("child_model_data50k.csv")

#original_columns = list(data.columns)

#data = data[gen_aleatorio(data)]





#est = PC(data)

#Wait times per number of instances: 100k ~ 5 min, 50k ~ 2 min 15sec
#model_chi = est.estimate(ci_test='chi_square', max_cond_vars=3,  significance_level=.1)
#print(len(model_chi.edges()))


#Reorder the columns to draw the graph
#M_chi = model_chi

#A = nx.nx_agraph.to_agraph(M_chi)

#A.draw('.//imagenes//model_M_chi_graphviz.png', format='png', prog='dot') 


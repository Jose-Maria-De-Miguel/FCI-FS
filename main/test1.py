from pgmpy.models import BayesianNetwork

from pgmpy.estimators import PC



from  PCFS import PCFS
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PDAG
import random
import csv
import numpy as np

n_states = {
                var: np.random.randint(low=1, high=5, size=1)[0] for var in range(300)
            }
model = BayesianNetwork.get_random(n_nodes=300, edge_prob=2/(300-1), n_states= n_states )

data = model.simulate(n_samples=500)


data.to_csv("./test1.csv")

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:52:02 2024

@author: chdem
"""
from PDAG import PDAG
from base import StructureEstimator
from itertools import chain, combinations
import sys
from pgmpy.estimators.CITests import *
import copy




CI_TESTS = {
    "chi_square": chi_square,
    "independence_match": independence_match,
    "pearsonr": pearsonr,
    "ci_pillai": ci_pillai,
    "g_sq": g_sq,
    "log_likelihood": log_likelihood,
    "modified_log_likelihood": modified_log_likelihood,
    "power_divergence": power_divergence,
}

class PC(StructureEstimator):
    
    
    def __init__(self, data=None, independencies=None, **kwargs):
        super(PC, self).__init__(data=data, independencies=independencies, **kwargs)
        
        
        
    def estimate(
         self,
         #variant="stable",
         ci_test="chi_square",
         max_cond_vars=5,
         
         significance_level=0.01,
         #n_jobs=-1,
         #show_progress=True,
         pdag = None,
         new_vars = None,
         **kwargs,
     ):
         skel, separating_sets = self.build_skeleton(
            ci_test=ci_test,
            max_cond_vars=max_cond_vars,
            significance_level=significance_level,
            # variant=variant,
            # n_jobs=n_jobs,
            # show_progress=show_progress,
            pdag = pdag,
            new_variables = new_vars,
            **kwargs,
         )
         #skel.plot(title = "Resultado esqueleto:")
         
         phase = "v-estructuras"
         border = "=" * 70  # Adjust the width as needed
       

         """
         pdag =  self.calc_v_structures(
             ci_test=ci_test,
             max_cond_vars=max_cond_vars,
             significance_level=significance_level,
             # variant=variant,
             # n_jobs=n_jobs,
             # show_progress=show_progress,
             skeleton = skel,
             
             **kwargs,
             )
         
         """
         
         self.calc_v_structures(
            ci_test=ci_test,
            max_cond_vars=max_cond_vars,
            significance_level=significance_level,
            # variant=variant,
            # n_jobs=n_jobs,
            # show_progress=show_progress,
            skeleton = skel,
            new_variables = new_vars,
            **kwargs,
         )
        
         phase = "Finish"
         border = "=" * 70  # Adjust the width as needed
         
        
         
         return skel
     
        
         
             
         
         
     
    def build_skeleton(
         self,
         ci_test="chi_square",
         max_cond_vars=5,
         significance_level=0.01,
          pdag = None, 
          new_variables = None,
          deep_copy = True,
          **kwargs,
     ):
        # Initialize initial values and structures.
        lim_neighbors = 0
        separating_sets = dict()
        
        if pdag and deep_copy:
            graph = copy.deepcopy(pdag)
        elif pdag and not deep_copy : 
            graph = pdag
        else:
            graph = PDAG()
            
        if not callable(ci_test):
            try:
                ci_test = CI_TESTS[ci_test]
            except KeyError:
                raise ValueError(
                    f"ci_test must either be one of {list(CI_TESTS.keys())}, or a function. Instead, got: {ci_test}"
                    
                )
         # Step 1: Initialize a fully connected undirected graph
       
        graph.simplify_pdag()
       
        graph.initial_skeleton(new_variables)
        
        
        
        # Exit condition: 1. If all the nodes in graph has less than `lim_neighbors` neighbors.
        #             or  2. `lim_neighbors` is greater than `max_conditional_variables`.
        while not all(
            [len(list(graph.neighbors(var))) < lim_neighbors for var in graph.nodes()]
        ):
            # In case of stable, precompute neighbors as this is the stable algorithm.
            neighbors = {node: set(graph[node]) for node in graph.nodes()}
            for u, v, dir in graph.edges():
                for separating_set in chain(
                    tiocombinans(set(neighbors[u]) - set([v]), lim_neighbors),
                    combinations(set(neighbors[v]) - set([u]), lim_neighbors),
                ):
                    # If a conditioning set exists remove the edge, store the
                    # separating set and move on to finding conditioning set for next edge.
                    if ci_test(
                        u,
                        v,
                        separating_set,
                        data=self.data,
                        independencies=self.independencies,
                        significance_level=significance_level,
                        **kwargs,
                    ):
                        
                        #print(f"Edges: {u} -> {v}. Dirección: {dir}, sepset: {separating_set} ")
                        
                        #TODO: Debug
                        #TODO: Modificar para que acepte aristas bidireccionadas
                        separating_sets[frozenset((u, v))] = separating_set
                        if dir > 0:
                           unmoralized_edges = graph.extramarital_affairs(v)
                           if len(unmoralized_edges) <= 2: #u is supposed to be an unmoralized edge
                               for u_edge in unmoralized_edges:
                                   graph.eliminate_orientation(u_edge, v) 
                           if dir == 2 and (unmoralized_edges := graph.extramarital_affairs(v)) == 2: #Bi directed edges
                               for u_edge in unmoralized_edges:
                                   graph.eliminate_orientation(u_edge, u)
                            
                        graph.remove_edge((u, v))
                        
                        
                        break
            if lim_neighbors >= max_cond_vars:
                print(
                    "Reached maximum number of allowed conditional variables. Exiting"
                , file=sys.stderr
                )
                break
            lim_neighbors += 1
            
        return graph, separating_sets
    
    
        
      
    def calc_v_structures(self,
            
                          ci_test="chi_square",
                          max_cond_vars=5,
                          significance_level=0.01,
                       #   variant="stable",
                       #   n_jobs=-1,
                       #   show_progress=True,
                           skeleton = None, 
                           new_variables = None,
                           deep_copy = True,
                           **kwargs,
                      ):
        
        if not callable(ci_test):
            try:
                ci_test = CI_TESTS[ci_test]
            except KeyError:
                raise ValueError(
                    f"ci_test must either be one of {list(CI_TESTS.keys())}, or a function. Instead, got: {ci_test}")
                    
        if not skeleton:
            
            raise ValueError("A skeleton is needed to calculate the v-structures.")
            
        if deep_copy:   
            pdag = copy.deepcopy(skeleton)
        else:
            pdag = skeleton
        
        
        node_pairs = list(combinations(pdag.nodes(), 2))
        
        
        potential_colliders = []
        
        for pair in node_pairs:
           
            X, Y = pair
            if not pdag.has_edge(X, Y):
                
              #  print(f"Valores: X: {X}-Y: {Y}" )
                
                
                potential_colliders =  list(set(pdag.neighbors(X)) & set(pdag.neighbors(Y)))
                
               # print(f"Potential colliders list: {potential_colliders}")
                    
                if potential_colliders and (sepsets := self.calculate_sepsets(X = X, Y = Y, ci_test = ci_test,max_cond_vars = max_cond_vars,significance_level = significance_level, skeleton = skeleton )):
                    #We put this here in order to avoid calculating the sepsets (the most expensive operation) if there aren't any potential colliders 
                    
                    for Z in potential_colliders:
                        
                    #permitimos o no aristas bidireccionadas
                       # if not pdag.has_direction(v1 = X, v2 = Z) and not pdag.has_direction(v1 = Y, v2 = Z):
                        countZ = countElem(Z, sepsets)
                    #    print(f"Potential collider:  {X}-{Z}-{Y}: Sepsets: {sepsets}")
                      #  print(f"Porcentaje Z: {countZ/len(sepsets)}")
                        if countZ/len(sepsets) < 0.5:
                            
                          #  print(f"V-structure: {X} -> {Z} <- {Y}")
                            pdag.add_direction(X, Z)
                            pdag.add_direction(Y, Z)
                        #    pdag.plot(title = f"V-structure: {X} -> {Z} <- {Y}")

        
        return pdag
        
     
        
    
    def calculate_sepsets(
             self,
             X = None,
             Y = None,
             ci_test="chi_square",
             max_cond_vars=5,
             
             significance_level=0.01,
             
             skeleton = None,
             
             **kwargs,
             ):
        if not skeleton:
            
            raise ValueError(f"A PDAG is needed to calculate the sepsets of {X} and {Y}.")
        
        nodes = skeleton.nodes()
        if  X not in nodes or Y not in nodes:
            raise ValueError("Both vertices must belong to the indicated PDAG.")
            
            
        
        
        neighborsX =  skeleton.neighbors(X)
        
        neighborsY =  skeleton.neighbors(Y)
        
        sepsets = []
        
        #If X and Y are adjacent then we assume they can not be separated by max_cond_vars variables
        if (Y in  neighborsX) or (X in neighborsY):
            return sepsets
        
        #Here we know that X, Y are not adjacent
        
        lim_neighbors = 0
        
        max_neighbors = max(len(neighborsX), len(neighborsY))
        
        while lim_neighbors <= max_cond_vars and lim_neighbors <= max_neighbors:
            
            #TODO: Deberia usar combinations o permutations??
                for separating_set in chain(
                    combinations(set(neighborsX) , lim_neighbors),
                    combinations(set(neighborsY) , lim_neighbors),
                   
                ):
                    if ci_test(
                        X,
                        Y,
                        separating_set,
                        data=self.data,
                        independencies=self.independencies,
                        significance_level=significance_level,
                        **kwargs,
                    ):
                        sepsets.append(separating_set)
                lim_neighbors+=1
                
                
        return sepsets
    
        
            
            
            
def countElem(elem, listOfLists):
    
    count = sum(1 for list in listOfLists if elem in list)
    
    return count
    
            
        
        
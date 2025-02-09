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



class PCFS(StructureEstimator):
    
    
    def __init__(self, data=None, independencies=None, **kwargs):
       
        super(PCFS, self).__init__(data=data, independencies=independencies, **kwargs)
        
        
        
    def estimate(
         self,
         variant = "stable",
         ci_test="chi_square",
         max_cond_vars=3,
         significance_level=0.01,
         pdag = None,
         new_vars = None,
         **kwargs,
     ):
        
         if not new_vars: #If there are not any new variables to add to the pdag, return the pdag
            return pdag, 0
        
         if variant not in ["stable", "orig", "MPC", "MPC-stable"] :
            raise ValueError(f"Variant should be either stable, orig, MPC or MPC-stable but received {variant}")
         
        
         if variant in ["stable", "MPC-stable" ]:
            skelVariant = "stable"
         else:
            skelVariant = "orig"
            
            
         skel, separating_sets, numCItest = self.build_skeleton(
            variant = skelVariant,
            ci_test=ci_test,
            max_cond_vars=max_cond_vars,
            significance_level=significance_level,
            pdag = pdag,
            new_variables = new_vars,
            **kwargs,
         )
         
         
         
        
       
       

         if variant in ["MPC", "MPC-stable" ]:
             sepsets  = None
         else: # Orig version
             sepsets = separating_sets
     
            
         
         pdag, numCItest2 =    self.calc_v_structures(
            ci_test=ci_test,
            max_cond_vars=max_cond_vars,
            significance_level=significance_level,
           
            skeleton = skel,
            
            separating_sets = sepsets,
            **kwargs,
         )
         
         numCItest = numCItest + numCItest2
         
         
        
        
         pdag.applyMeek()
        
         
         
         return pdag, numCItest
     
        
         
             
         
         
     
    def build_skeleton(
         self,
         variant = "stable",
         ci_test="chi_square",
         max_cond_vars=3,
         significance_level=0.01,
          pdag = None, 
          new_variables = None,
          deep_copy = True,
          **kwargs,
     ):
        
        if variant not in ["stable", "orig"] :
            raise ValueError(f"Variant should be either stable or orig but received {variant}")
        
        numCITest = 0
       
        
        
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
       
        graph.undirected()
       
        graph.initialSkeleton(new_variables)
        
        
       
        # Exit condition: 1. If all the nodes in graph has less than `lim_neighbors` neighbors.
        #             or  2. `lim_neighbors` is greater than `max_conditional_variables`.
        while not all(
            [len(list(graph.neighbors(var))) < lim_neighbors for var in graph.nodes()]
        ):
            if variant == "stable":
                # In case of stable, precompute neighbors as this is the stable algorithm.
                neighbors = {node: set(graph[node]) for node in graph.nodes()}
            for u, v, dir in graph.edges():
                
                if variant == "stable": #Stable
                    nu, nv = neighbors[u], neighbors[v]
                else: # Orig
                    nu, nv = graph.neighbors(u), graph.neighbors(v)
                    
                
               
                for separating_set in chain(
                    combinations(set(nu) - set([v]), lim_neighbors),
                    combinations(set(nv) - set([u]), lim_neighbors),
                ):
                    # If a conditioning set exists remove the edge, store the
                    # separating set and move on to finding conditioning set for next edge.
                    
                    numCITest = numCITest + 1
                    
                   
                    
                    if ci_test(
                       u,
                       v,
                       separating_set,
                       data=self.data,
                       independencies=self.independencies,
                       significance_level=significance_level,
                       **kwargs,
                   ):
                        
                        separating_sets[frozenset((u, v))] = separating_set
                        graph.elimAdjacency((u, v))
                        
                        break

            if lim_neighbors >= max_cond_vars:
                # print("Reached maximum number of allowed conditional variables. Exiting", file=sys.stderr )
                break
            lim_neighbors += 1
        
        return graph, separating_sets, numCITest
    
    
        
      
    def calc_v_structures(self,
                          
                          skeleton, 
                          ci_test="chi_square",
                          max_cond_vars=3,
                          significance_level=0.01,
                           
                           separating_sets = None,
                           deep_copy = True,
                           **kwargs,
                      ):
        
        if not callable(ci_test):
            try:
                ci_test = CI_TESTS[ci_test]
            except KeyError:
                raise ValueError(
                    f"ci_test must either be one of {list(CI_TESTS.keys())}, or a function. Instead, got: {ci_test}")
                    
            
        if deep_copy:   
            pdag = copy.deepcopy(skeleton)
        else:
            pdag = skeleton
        
        
        numCITest = 0
        
        node_pairs = list(combinations(pdag.nodes(), 2))
        
        
        potential_colliders = []
        
        for pair in node_pairs:
           
            X, Y = pair
            if not pdag.hasEdge(X, Y):
            #We do not get parent nodes as potential colliders to avoid bidirected edges
                potential_colliders =  list(set(pdag.neighbors(X, mode = "unAndChild")) & set(pdag.neighbors(Y, mode = "unAndChild")))       
                
                if potential_colliders :
                    
                    calcSepsets = self.calculate_sepsets(X = X, Y = Y, ci_test = ci_test,max_cond_vars = max_cond_vars,
                                                       significance_level = significance_level, skeleton = skeleton, separating_sets =  separating_sets )
                    sepsets = calcSepsets[0]
                   
                    numCITest = numCITest + calcSepsets[1]
                    #We the calculate_sepsets within the if condition in order to avoid calculating the sepsets (the most expensive operation) if there aren't any potential colliders  
                    for Z in potential_colliders:      
                    
                        countZ = countElem(Z, sepsets)
                        if len(sepsets) == 0 or countZ/len(sepsets) < 0.5:
                            pdag.addDirection(X, Z)
                            pdag.addDirection(Y, Z)
                
                    
        
        return pdag, numCITest
        
   
      
        
    
    def calculate_sepsets(
             self,
             X,
             Y ,
             skeleton,
             variant = "stable",
             ci_test="chi_square",
             max_cond_vars=5,
             significance_level=0.01,
             separating_sets = None,
             **kwargs,
             ):
        if not skeleton:
            
            raise ValueError(f"A PDAG is needed to calculate the sepsets of {X} and {Y}.")
        
        nodes = skeleton.nodes()
        if  X not in nodes or Y not in nodes:
            raise ValueError("Both vertices must belong to the indicated PDAG.")
            
            
        if separating_sets:
            return [separating_sets[frozenset((X, Y))]], 0
            
        
        
        neighborsX =  skeleton.neighbors(X)
        
        neighborsY =  skeleton.neighbors(Y)
        
        sepsets = []
        
        #If X and Y are adjacent then we assume they can not be separated by max_cond_vars variables
        if (Y in  neighborsX) or (X in neighborsY):
            return sepsets, 0
        
        #Here we know that X, Y are not adjacent
        numCITest = 0
        
        lim_neighbors = 0
        
        max_neighbors = max(len(neighborsX), len(neighborsY))
        
        while lim_neighbors <= max_cond_vars and lim_neighbors <= max_neighbors:
            
            
                for separating_set in chain(
                    combinations(set(neighborsX) , lim_neighbors),
                    combinations(set(neighborsY) , lim_neighbors),             
                ):
                    
                    numCITest = numCITest + 1
                    
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
                
                
        return sepsets, numCITest
    
        
            
            
            
def countElem(elem, listOfLists):
    
    
    count = sum(1 for listElem in listOfLists 
                if elem in listElem)
    
    return count
    
            
        
        
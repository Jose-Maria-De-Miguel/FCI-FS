# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:05:37 2024

@author: chdem
"""

import matplotlib.pyplot as plt
from igraph import Graph, plot
import igraph
import copy
from itertools import  combinations

class PDAG():
    
    
            
    """
    TODO: Mejorar mensajes y tipos de error (ValueError, KeyError etc.)
    """
    
    def __init__(self, v_names = None, d_edges = None, u_edges = None):
        """
        Receives a list of identifiers (vertex names) that are hashable (unique), a list of 
        directed edges, a list of undirected edges, a list of graph attributes, `graph_attrs`, 
        `vertex_attrs`, and `edge_attrs`.
        As local variables, the graph, a set of unique identifiers,
        and a dictionary indicate whether an edge is directed or not.
        The dictionary will store the edges as pairs of integers, using the internal representation of the igraph.
        """
        
        # This variable is used to check if there are repeated variables in a dict
        self.n_vertices = 0
        
        #edge_dct to store whether an edge is  undirected (0), directed (1), or bidirected (2)
        
        self.edge_dct = {}
        
        #names_dict to avoid linear search using the graph
        self.names_dict = {}
        
        
        
        
        if v_names:
            
            self.n_vertices = len(v_names)
            
            #update names in the dictionary 
            self.__update_names_dict(v_names, 0, self.n_vertices)
            
                
        # Initialize a directed iGraph 
        self.graph = Graph(directed=1, n =  self.n_vertices, )
        
        self.graph.vs["color"] = "white"
        self.graph.vs["frame_color"] = "black"  # Border color
        
        self.graph.vs["name"] = v_names
       
        if not v_names and (d_edges or u_edges):
            raise ValueError("Can not initialize edges in the PDAG without declaring any vertex ")
  
        if d_edges and v_names:
            
            self.add_edges(d_edges, 1)
            
            
        if u_edges and v_names:
            
            self.add_edges(u_edges, 0)
        
       
        
       
        
    def neighbors(self, edge):
        """
        Get the neighbors of an indicated edge in the  PDAG.

        Parameters
        ----------
        edge: TYPE
            Edge in the PDAG.

        Raises
        ------
        ValueError
            The indicated edge must be included in the PDAG, otherwise, an error is raised.

        Returns
        -------
        list
            list of neighbors of the indicated edge.

        """
        
        if not edge in self.names_dict:
            raise ValueError("The indicated edge is not included in the graph")
            
        neighbor_indices = self.graph.neighbors(self.names_dict[edge])
        
        return [self.graph.vs[i]["name"] for i in neighbor_indices]
        
        
    def nodes(self):
        """
        Returns list of nodes of the PDAG.

        """
        return list(self.names_dict.keys())
    
    
    def edges(self):
        """
        Returns list of edges of the PDAG.

        """
        
        
        
        return [(self.graph.vs["name"][edge[0]], self.graph.vs["name"][edge[1]], self.edge_dct[hashable_edge(edge[0], edge[1])]) for edge in self.graph.get_edgelist()]


    def remove_edge(self, edge, directed = False):
        
        eid = self.get_edge_id(edge, directed)
        
        if eid == -1:
            raise ValueError("The indicated edge does not exist")
        
        #Eliminate edge from the dictionary
        self.edge_dct.pop(self.__to_h_edge(edge))
        #Eliminate edge from the graph
        self.graph.delete_edges(eid)
                                
        
        
        
    def get_graph(self):
        """Returns the iGraph object."""
        return self.graph
    
    def plot(self,vertex_frame_width = 2, title = "PDAG Visualization:", coords = None ):
        """
        Plot the PDAG, kamada_kawai is used for the plot

        """
        

        graphAux =  copy.deepcopy(self.graph)
        
        for edge in graphAux.es:
            key = hashable_edge(edge.source, edge.target)
            
            if self.edge_dct[key] == 2:
                color = "green"
                style = "solid"  
                directed = True
                
                reverse_edge = (edge.target, edge.source)
                if not graphAux.are_connected(*reverse_edge):
                    graphAux.add_edge(edge.target, edge.source,  
                                           color = "green",
                                           style = "solid"  ,
                                           directed = True,
                                           )
                
            elif self.edge_dct[key] == 1 :
                # Directed edge
                color = "blue"
                style = "solid"  
                directed = True
            else:
                # Undirected edge
                color = "black"
                style = "dashed"  
                directed = False
            
            graphAux.es[edge.index]["color"] = color
            graphAux.es[edge.index]["style"] = style
            
            graphAux.es[edge.index]["arrow_size"] = 10 if directed else -1
    
       
    
        # Plot using a layout
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if coords:
            layout = igraph.Layout(coords)
        else:
            layout = self.graph.layout("kamada_kawai")
            
        plot(
            graphAux,
            target=ax,
            layout=layout,
            edge_arrow_size= graphAux.es["arrow_size"],
            vertex_frame_width=vertex_frame_width,  # Ensure thick borders render
            vertex_label= list(self.names_dict.keys())
            )
        ax.set_title(title, fontsize=16)
        plt.show()
        return layout.coords

    """
    def plot(self):
        
        edge_colors = []
        for edge in self.graph.es:
            if self.edge_dct[hashable_edge(edge.source, edge.target)]:
                edge_colors.append('blue')
            
            else:
                edge_colors.append('red')
                
            
        # Set edge colors
        self.graph.es['color'] = edge_colors  
        
        fig, ax = plt.subplots()
        plot(self.graph, target=ax)  # Use target=ax to direct igraph's plot to Matplotlib
        
        plt.show()
    """    
    def initial_skeleton(self, new_variables = None):
        

        if not new_variables:
            return
        
        if any(var in self.names_dict for var in new_variables):
            raise ValueError("All the vertices must be new.")
        
        
        
        n_oldvertices = self.n_vertices
        self.n_vertices += len(new_variables)
        
        #update names in the dictionary 
        self.__update_names_dict(new_variables, n_oldvertices, self.n_vertices)
        
        
        g_aux = (Graph.Full(len(new_variables))).as_directed(mode="arbitrary")
        
        g_aux.vs["name"] = new_variables
        
        #Each e dge between the new variables is undirected
        self.edge_dct.update({hashable_edge(edge[0]+n_oldvertices, edge[1]+n_oldvertices) : 0 for edge in  g_aux.get_edgelist()})
        
        self.graph = self.graph + g_aux
        
         #Connect each old variable with each new variable with an undirected edge
        for i in range(0, n_oldvertices):
            for j in range(0,  len(new_variables)):
                
                self.graph.add_edges([(i, j + n_oldvertices)])
                self.edge_dct.update({hashable_edge(i, j + n_oldvertices ) : 0})
                
    
    
    def add_edges(self, edges, direction):
        
        if direction not in [0, 1, 2]:
            raise ValueError("The type of edge must be indicated with 0, 1 or 2")
            
        #check there are not repeated edges
        if(len(edges) != len(set(edges))):
            raise ValueError("The graph can not have two edges between the same vertices")
            
            
        
        for edge in edges:
            
            if self.__is_in_edge_dct(edge) :
                raise ValueError("Only one edge is allowed between two variables.")
            
            source = self.names_dict[edge[0]]
            target =  self.names_dict[edge[1]]
            
            if source == target:
                raise ValueError("Edges must go from one variable to a different one")
            
            #Add edge to graph
            Graph.add_edge(self.graph, source, target)
            
            #Assign direction in the edge dictionary
            self.edge_dct.update({hashable_edge(source, target) : direction})
            
       
       
           
            
        
            
    def __update_names_dict(self, names, offset, newsize):
        
        # Check that the vertex names are unique and assign a number corresponding to the vertex id in the graph
        self.names_dict.update({ names[i] : i + offset  for i in range(0, len(names))})
        
        # If there are repeated names, raise error 
        if(len(self.names_dict) != newsize):
            raise ValueError("All vertices names must be unique.")
        
        
    def get_edge_id(self, edge, directed = False, error = False):
        """
        If found, it returns the edge's igraph ID; otherwise, it returns -1.
   
        if there is a self loop or the variables are not included in the names dictionary raise an error 
        
        Directed indicates whether the direction of the edge should be considered.

        """
        
        e0 = edge[0]
        e1 = edge[1]
        
        # Rule 1: Check for self-loops
        if e0 == e1:
           raise ValueError("There cannot be any edges from one variable to itself.")
    
        # Rule 2: Ensure both edge endpoints exist in self.names_dict
        if e0 not in self.names_dict or e1 not in self.names_dict:
            raise ValueError("The indicated edge does not consider variables in the graph.")
            
        if not self.__is_in_edge_dct(edge):
            return -1
        if directed and not self.is_directed(e0, e1): #If directed is true and the edge exist but it is not directed then retur -1 
            return -1
        
        res = self.graph.get_eid(v1 = self.names_dict[e0], v2 = self.names_dict[e1], directed = directed, error = error)
        
        if directed:

           return res 
       
        elif res == -1: #directed = False, the edge is in the dictionary but not in the graph
            raise ValueError("Internal error: Edge present in the edge dictionary but not in the graph. \nCheck how you add\\remove edges.")
        
        #directed = False and the edge is found both in the graph and in the dictionary
        return res
                
            
    def __to_h_edge(self, edge):
        return  hashable_edge(self.names_dict[edge[0]], self.names_dict[edge[1]])
    
    def __is_in_edge_dct(self, edge):
        return self.__to_h_edge(edge) in self.edge_dct
    
    #TODO: SPECIFY METHOD

    def __getitem__(self, variable):
        
        if variable in self.names_dict:
            return self.neighbors(variable)
        
        raise KeyError(f"Variable {variable} not found in the graph.")

        

    def __str__(self):
        """Returns a string representation of the graph and the dictionary."""
        return f"Graph: {self.graph.summary()}\nAttributes: {self.attributes}"
    
    def __eq__(self, other):
    
        return self.get_differences(other) == set()
    
    def get_differences(self, other):
        if not isinstance(other, PDAG):
            return NotImplemented
        
        # Check if nodes match
        if set(self.names_dict.keys()) != set(other.names_dict.keys()):
            return False
        
        # Check if edges match, check for the edge and it's direction
        self_edges = {self.get_edge_names(edge) for edge in self.edge_dct} 
        other_edges = {other.get_edge_names(edge) for edge in other.edge_dct}
        
        return (self_edges - other_edges) | (other_edges - self_edges )
    
    def get_confusion_matrix(self, other):
        """
        Return the confusion matrix of the {self} PDAG compared to the {other} PDAG plus tow other parameters; TD and FD
        TP -> number of true edges/arcs present
        FP -> number of false edges/arcs present
        TN -> number of true absent edges
        FN -> number of false absent edges
        TD -> number of true arcs present
        FD -> number of false arcs present
        """
        names =   set(self.names_dict.keys())
        if names != set(other.names_dict.keys()):
            return False
        
        
        
        TP, FP, TN, FN, TD, FD = 0, 0, 0, 0, 0, 0
        
        #TD, FD consider only whether the edge has been directed correctly
        
        
        for name1, name2 in ( combinations(names, 2)):
            
        
                #Check if the edge is in the graph:
                 if (self.get_edge_id(edge = (name1, name2), directed = False ) != -1):
                     if (other.get_edge_id(edge = (name1, name2), directed = False ) != -1):
                         
                         TP = TP + 1 
                     else :
                         FP = FP + 1
                    
                     if (((self.get_edge_id(edge = (name1, name2), directed = True ) != -1 )
                         and (other.get_edge_id(edge = (name1, name2), directed = True )!= -1) ) 
                         or ((self.get_edge_id(edge = (name2, name1), directed = True ) != -1) and (other.get_edge_id(edge = (name2, name1), directed = True ) != -1))):
                        
                         TD = TD + 1
                         TP = TP + 1 
                     elif self.is_directed(name1, name2):
                      #  If the PDAG contains a directed edge that is absent in the other PDAG (neither directed nor undirected), then the FP is counted twice
                         FD = FD +1
                         FP = FP + 1
                        
                         
                    
                 else:
                    if (other.get_edge_id(edge = (name1, name2), directed = False ) != -1):
                        FN = FN + 1
                    else:
                        TN = TN + 1
                     
                    
        return [TP, FP, TN, FN, TD, FD]
        
    
    def __deepcopy__(self, memo):
        new_obj = PDAG()
        new_obj.names_dict = copy.deepcopy(self.names_dict)
        new_obj.edge_dct = copy.deepcopy(self.edge_dct)
        new_obj.n_vertices = self.n_vertices
        new_obj.graph = copy.deepcopy(self.graph)
        
        return new_obj
        
    
    def get_edge_names(self, h_edge):
        """
        Given a hashable_edge return a tuple of 2 elements: 
          1) The pair of names that correspond to the edge, this pair is sorted if the edge is undirected.
          2) An an intger whith possible values 0,1,2 that indicates whether the edge is undirected, directed or bidirected

        """
        if self.edge_dct[h_edge] == 0 or self.edge_dct[h_edge] == 2:
            return (tuple(sorted((self.graph.vs[h_edge.source]["name"], self.graph.vs[h_edge.target]["name"]))), self.edge_dct[h_edge])
        
        elif self.edge_dct[h_edge] == 1:
            eid =  self.graph.get_eid(h_edge.source, h_edge.target, directed = False)
            v0 = self.graph.es[eid].source
            v1 = self.graph.es[eid].target
            return ((self.graph.vs[v0]["name"], self.graph.vs[v1]["name"]), 1)
        else:
            raise ValueError("Internal error: edge_dct can only take 3 values: 0, 1, 2.")
            
            
    def simplify_pdag(self):
        """
        Simplifies the PDAG by resolving edge directions involved in v-structures. 
        Only edges that are part of v-structures retain their direction. 
        All other edges are left undirected.
        """
        for edge in self.edge_dct:
            if self.edge_dct[edge] > 0 and not self.is_from_v_structure(edge):
                self.edge_dct[edge] = 0
                
            #TODO edges bidireccionados
            #elif self.edge_dct[edge] == 1 etc.
    
    def undirected(self):
        for edge in self.edge_dct:
            self.edge_dct[edge] = 0
        
        
        
      
    
        
    
    
    
    def is_from_v_structure(self, edge):
        """
        Returns true if the edge is directed and it's direction is part of an v-structure, returns false otherwise

        """
        if(self.edge_dct[edge] == 0):
            return False
            
        source, target = self.__getOrientation(edge)
        targetFathers =  self.graph.neighbors(target, mode = "in")
        
        return any(
            (node != source) and
            (node not in self.graph.neighbors(source, mode="all")) and
            (self.edge_dct[hashable_edge(node, target)] == 1)
            for node in targetFathers
        )
       
       
            
        
        
       
        
    def __getOrientation(self, edge):
        """
        Given a hashable_edge return the orientation of the edge in self.graph (even if the edge is not oriented in the pdag)

        """
        
        v1, v2 = edge.to_tuple(edge)
        
        eid = self.graph.get_eid(v1 = v1, v2 = v2, directed = False, error = True)

        return self.graph.es[eid].source,  self.graph.es[eid].target
    
    
    
    def extramarital_affairs(self, vertex):
        """
        Given a vertex, return the list of parents that make it a collider (if any)

        """
        if vertex not in self.names_dict:
            raise ValueError("The given vertex is not present in the graph.")
        
        vertex_num = self.names_dict[vertex]
        targetFathers =  self.graph.neighbors(vertex_num, mode = "in")
        
        
        return [self.graph.vs[node]["name"] for node in targetFathers 
                     if self.is_from_v_structure(hashable_edge(node, vertex_num)) ]
        
    def eliminate_orientation(self, v1, v2):
        
        if not self.__is_in_edge_dct((v1, v2)):
            raise ValueError("Internal error: indicated edge does not belong to the graph.")
            
        num1 = self.names_dict[v1]
        num2 = self.names_dict[v2]
        h_edge = hashable_edge(num1, num2)
        
        if (num1, num2) != self.__getOrientation(h_edge):
            raise ValueError("The indicated orientation does not exist.")
            
        self.edge_dct[h_edge]= 0
    
        
        
        
    def add_direction(self, v1, v2):
        if not self.__is_in_edge_dct((v1, v2)):
            raise ValueError(f"The indicated edge, ({v1}, {v2}) does not belong to the graph.")
            
        if v1 not in self.names_dict or v2 not in self.names_dict:
            raise ValueError("Edges must belong to the graph.")
            
        
        num1 = self.names_dict[v1]
        num2 = self.names_dict[v2]
        
        eid = self.graph.get_eid(v1 = num1, v2 = num2, directed = True, error = False)
        
        if self.edge_dct[hashable_edge(num1, num2)] == 2:
            #If the edge is bidirected return
            return
        
        
        
        
        if(eid == -1):
            eid = self.graph.get_eid(v1 = num2, v2 = num1, directed = True, error = False)
            if(eid == -1):
                raise ValueError("Internal error: The indicated edge does not belong to the graph.")
            elif( self.edge_dct[hashable_edge(num1, num2)] == 1): #before: num2 -> num1 now num2 <->num1
                self.edge_dct[hashable_edge(num1, num2)] = 2
            else: 
                self.graph.reverse_edges([eid])
                self.edge_dct[hashable_edge(num1, num2)] = 1
        
        elif( self.edge_dct[hashable_edge(num1, num2)] < 1): #eid != -1
            self.edge_dct[hashable_edge(num1, num2)] = 1
            
        
        
    
    def has_edge(self, v1, v2):
        
        if v1 not in self.names_dict or v2 not in self.names_dict:
            raise ValueError("Both vertices must belong to the graph")
            
        return self.__is_in_edge_dct((v1, v2))
    
    def is_directed(self, v1, v2):
        if not self.__is_in_edge_dct((v1, v2)):
            raise ValueError(f"The indicated edge, ({v1}, {v2}) does not belong to the graph.")
            
        if v1 not in self.names_dict or v2 not in self.names_dict:
            raise ValueError("Edges must belong to the graph.")
            
        
        num1 = self.names_dict[v1]
        num2 = self.names_dict[v2]
        
        return bool(self.edge_dct[hashable_edge(num1, num2)])
        
        
        
        
        
class hashable_edge():
    """
    
    In the PDAG we introduce the constraint that we can only have one edge between  to do this we identify edges as equal regardless of their direcction.
    
    i.e. (0,1) is equal to (1, 0)
    
    To check the direction of an edge it is needed to query self.graph

    """
    def __init__(self, source, target):
        self.source = source
        self.target = target
        
    def to_tuple(self,reversed = False):
        
      return (self.target, self.source) if reversed else (self.source, self.target)

       
        
        
    def __str__(self):
        return f" h_edge: ({self.source}, {self.target})."
    
    def __repr__(self):
        return f"hashable_edge({self.source}, {self.target})"
       

    def __eq__(self, other):
        if isinstance(other, hashable_edge):
            return (self.source == other.source and self.target == other.target) or (self.source == other.target and self.target == other.source)
        else:
            return NotImplemented

    def __hash__(self):
        # Combine the hash of name and value to create a unique hash for the object
        return hash(tuple(sorted((self.source, self.target))))
        
    
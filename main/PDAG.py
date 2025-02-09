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


def isIterableEmpty(iterable):
    """
    Return True if the iterable has an element, False otherwise.

    """
    return True if any(True for _ in iterable) else False

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
        
        self.edge_dict = {}
        
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
            
            self.__add_underlying_edges(d_edges, 1)
            
            
        if u_edges and v_names:
            
            self.__add_underlying_edges(u_edges, 0)
        
       
        
       
        
    def neighbors(self, vertex, mode = "all"):
        """
        Get the neighbors of an indicated vertex in the  PDAG acording to the indicated mode.
        
        all: returns all its neighbors (default mode). 
        
        parents: Return all the nodes that point to the indicated vertex.
        
        undirected : Return all neighbors connected by undirected edges.
        
        children: Return all the nodes that are pointed to by the indicated vertex.
        
        unAndChild: Return both, undirected and children nodes.

        Parameters
        ----------
        Vertex:  Vertex in the PDAG.

        Raises
        ------
        ValueError
            The indicated vertex must be included in the PDAG, otherwise, an error is raised.

        Returns
        -------
        list
            list of neighbors of the indicated vertex.

        """
        
        if not self.isVertex(vertex):
            raise ValueError("The indicated vertex is not included in the graph")
          
        neighbor_indices = self.graph.neighbors(self.names_dict[vertex])
        
        adjacencies_names = [self.graph.vs[i]["name"] for i in neighbor_indices]
        if mode == "all":
            return adjacencies_names
        
        elif mode == "parents":

            parents = [node  for node in adjacencies_names  if self.hasDirection(node, vertex)]
                 
            return parents
        
        elif mode == "undirected":
            
            undirected = [node  for node in adjacencies_names  if self.getOrientation(node, vertex) == 0]
         
            return undirected
        
        elif mode == "children":
            children = [node  for node in adjacencies_names  if self.hasDirection(vertex, node)]
                 
            return children
        
        elif mode == "unAndChild":
            uAndC = [node  for node in adjacencies_names  if self.getOrientation(node, vertex) == 0 or  self.hasDirection(vertex, node)] 
            
            return uAndC
        
        else:
            raise ValueError(f"Error, mode {mode} is not known.")
            
        
        
        
    def nodes(self):
        """
        Returns list of nodes of the PDAG.

        """
        return list(self.names_dict.keys())
    
    
    def edges(self, mode = "all"):
        """
        Returns a list of triples (v1, v2, dir), each one corresponds to an edge. 
        Dir denotes the orientation of the edge (0, 1 or 2), v1 the source and v2 the target.
        
        The input variable mode denotes the type of edges wich is returned, all for all edges, 
        undirected for only undirected edges
        """
        edges = [(self.graph.vs["name"][edge[0]], self.graph.vs["name"][edge[1]], self.edge_dict[hashable_edge(edge[0], edge[1])]) for edge in self.graph.get_edgelist()]
        
        
        if mode == "all":
            return edges
        
        elif mode == "undirected":
            return [(edge[0], edge[1]) for edge in edges if edge[2] == 0]
        
        else:
            raise ValueError("Mode not recognized.")

    def elimAdjacency(self, edge):
        
        eid = self.__get_edge_id(edge, False)
        
        if eid == -1:
            raise ValueError("The indicated edge does not exist")
        
        
        #Eliminate edge from the dictionary
        self.edge_dict.pop(self.__to_h_edge(edge))
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
            
            if self.edge_dict[key] == 2:
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
                
            elif self.edge_dict[key] == 1 :
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
            
        if len(graphAux.es) <= 0:
            edge_arrow_size = 10
        else:
            edge_arrow_size= graphAux.es["arrow_size"]
            
        plot(
            graphAux,
            target=ax,
            layout=layout,
            edge_arrow_size= edge_arrow_size,
            vertex_frame_width=vertex_frame_width,  # Ensure thick borders render
            vertex_label= list(self.names_dict.keys())
            )
        ax.set_title(title, fontsize=16)
        plt.show()
        return layout.coords

    
    def initialSkeleton(self, new_variables = None):
        

        if not new_variables:
            return
        
        if any(self.isVertex(var) for var in new_variables):
            
            raise ValueError("All the vertices must be new.")
        
        
        
        n_oldvertices = self.n_vertices
        self.n_vertices += len(new_variables)
        
        #update names in the dictionary 
        self.__update_names_dict(new_variables, n_oldvertices, self.n_vertices)
        
        
        g_aux = (Graph.Full(len(new_variables))).as_directed(mode="arbitrary")
        
        g_aux.vs["name"] = new_variables
        g_aux.vs["color"] = "white"
        g_aux.vs["frame_color"] = "black"  # Border color
        
        #Each e dge between the new variables is undirected
        self.edge_dict.update({hashable_edge(edge[0]+n_oldvertices, edge[1]+n_oldvertices) : 0 for edge in  g_aux.get_edgelist()})
        
        self.graph = self.graph + g_aux
        
         #Connect each old variable with each new variable with an undirected edge
        for i in range(0, n_oldvertices):
            for j in range(0,  len(new_variables)):
                
                self.graph.add_edges([(i, j + n_oldvertices)])
                self.edge_dict.update({hashable_edge(i, j + n_oldvertices ) : 0})
                
    
    
    def __add_underlying_edges(self, edges, direction):
        
        if direction not in [0, 1, 2]:
            raise ValueError("The type of edge must be indicated with 0, 1 or 2")
            
        #check there are not repeated edges
        if(len(edges) != len(set(edges))):
            raise ValueError("The graph can not have two edges between the same vertices")
            
            
        
        for edge in edges:
            
            v1, v2 = edge[0], edge[1]
            
            if self.hasEdge(v1, v2) :
                raise ValueError("Only one edge is allowed between two variables.")
            
            source = self.names_dict[edge[0]]
            target =  self.names_dict[edge[1]]
            
            if source == target:
                raise ValueError("Edges must go from one variable to a different one")
            
            #Add edge to graph
            Graph.add_edge(self.graph, source, target)
            
            #Assign direction in the edge dictionary
            self.edge_dict.update({hashable_edge(source, target) : direction})
            
       
       
           
            
        
            
    def __update_names_dict(self, names, offset, newsize):
        
        # Check that the vertex names are unique and assign a number corresponding to the vertex id in the graph
        self.names_dict.update({ names[i] : i + offset  for i in range(0, len(names))})
        
        # If there are repeated names, raise error 
        if(len(self.names_dict) != newsize):
            raise ValueError("All vertices names must be unique.")
        
        
    def __get_edge_id(self, edge, directed = False, error = False):
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
    
       
            
        if not self.hasEdge(e0, e1):
            return -1
        if directed and not self.isDirected(e0, e1): #If directed is true and the edge exist but it is not directed then return -1 
            return -1
        
        res = self.graph.get_eid(v1 = self.names_dict[e0], v2 = self.names_dict[e1], directed = directed, error = error)
        
        if directed:

           return res 
       
        elif res == -1: #directed = False, the edge is in the dictionary but not in the graph
            raise ValueError("Internal error: Edge present in the edge dictionary but not in the graph. \nCheck how you add\\remove edges.")
        
        #directed = False and the edge is found both in the graph and in the dictionary
        return res
                
            
    def __to_h_edge(self, edge):
        """
        

        Parameters
        ----------
        edge : edge between two vertices (given by names).

        Action
        ---------
        Given an edge between two vertices given by it's names, return the corresponding hashable_edge.

        """
        return  hashable_edge(self.names_dict[edge[0]], self.names_dict[edge[1]])
    
    
    
  

    def __getitem__(self, variable): # allows accessing PDAGs by index.
        
        if  self.isVertex(variable):
            return self.neighbors(variable)
        
        raise KeyError(f"Variable {variable} not found in the graph.")

        

    def __str__(self):
        """Returns a string representation of the graph and the dictionary."""
        return f"Graph: {self.graph.summary()}\nAttributes: {self.attributes}"
    
    def __eq__(self, other):
    
        return self.getAdjacencyDifferences(other) == set()
    
    def getAdjacencyDifferences(self, other):
        if not isinstance(other, PDAG):
            return NotImplemented
        
        # Check if nodes match
        if set(self.names_dict.keys()) != set(other.names_dict.keys()):
            return False
        
        # Check if edges match, check for the edge and it's direction
        self_edges = {self.__get_edge_names(edge) for edge in self.edge_dict} 
        other_edges = {other.__get_edge_names(edge) for edge in other.edge_dict}
        
        return (self_edges - other_edges) | (other_edges - self_edges )
    
    def getGraphicalMetrics(self, other):
        """
        Return graphical metrics comparing self with other.
        TP -> number of true edges/arcs present
        FP -> number of false edges/arcs present
        TN -> number of true absent edges
        FN -> number of false absent edges
        TO -> number of true orientations present
        FO -> number of false orientations present
        TD -> number of true directions (For edges with one direction only)
        P  -> Precision, rate of correct edges discovered across all edges discovered
        R  -> Recall, rate of edges discovered across all true edges that could have been discovered
        F1 -> harmonic mean of Precision and recall
        SHD-> Number of insertions, deletions, and arc reversals need to transform {self} into {other}
        """
        names =   set(self.names_dict.keys())
        if names != set(other.names_dict.keys()):
            return False
        
        
        
        TP, FP, TN, FN, TO, FO, TD = 0, 0, 0, 0, 0, 0, 0
        
        #TD, FD consider only whether the edge has been directed correctly
        
        
        for name1, name2 in ( combinations(names, 2)):
            
        
                #Check if the edge is in the graph:
                 if (self.hasEdge(name1, name2)):
                     if (other.hasEdge(name1, name2)):
                         
                         TP = TP + 1 
                    
                    
                         #Both PDAGs have the same orientation for the same edge
                         if (self.getOrientation(name1, name2) == (self.getOrientation(name1, name2))):
                            
                             TO = TO + 1 #Orientation is right
                             TP = TP + 1 
                             
                             if self.getOrientation(name1, name2) not in [0, 2]:
                                 TD = TD + 1 #The direction is right (the edge has only one direction)
                             
                         else:
                          #  The pdags do not have the same edges
                             FO = FO +1
                             FP = FP + 1
                         
                     else :
                         FP = FP + 1
                        
                         
                    
                 else:
                    if other.hasEdge(name1, name2):
                        FN = FN + 1
                    else:
                        TN = TN + 1
                        
                        
        divisionByZero = (TP + FP  == 0) or (TP + FN == 0)
        if divisionByZero:
            P = -1 
            R = -1
            F1 = -1
            
        else:
            P = TP/(TP + FP)
            R = TP/(TP + FN)
            F1 = (2*P*R)/(P+R)
        
            
        SHD = FN + FP
        
        return [TP, FP, TN, FN, TO, FO, TD, P, R, F1, SHD]
        
    
    def __deepcopy__(self, memo):
        new_obj = PDAG()
        new_obj.names_dict = copy.deepcopy(self.names_dict)
        new_obj.edge_dict = copy.deepcopy(self.edge_dict)
        new_obj.n_vertices = self.n_vertices
        new_obj.graph = copy.deepcopy(self.graph)
        
        return new_obj
        
    
    def __get_edge_names(self, h_edge):
        """
        Given a hashable_edge return a tuple of 2 elements: 
          1) The pair of names that correspond to the edge, this pair is sorted if the edge is undirected.
          2) An an integer whith one of the possible values: 0,1,2. The integer indicates whether the edge is undirected, directed or bidirected

        """
        if self.edge_dict[h_edge] == 0 or self.edge_dict[h_edge] == 2:
            return (tuple(sorted((self.graph.vs[h_edge.source]["name"], self.graph.vs[h_edge.target]["name"]))), self.edge_dict[h_edge])
        
        elif self.edge_dict[h_edge] == 1:
            eid =  self.graph.get_eid(h_edge.source, h_edge.target, directed = False)
            v0 = self.graph.es[eid].source
            v1 = self.graph.es[eid].target
            return ((self.graph.vs[v0]["name"], self.graph.vs[v1]["name"]), 1)
        else:
            raise ValueError("Internal error: edge_dict can only take 3 values: 0, 1, 2.")
            
    def elimDirection(self, v1, v2):
        """
        Eliminate the direction of an edge, raises en error if the edge doesn't exist

        """
        if not self.hasEdge(v1, v2):
            raise ValueError("The indicated edge is not present in the graph.")
        
        self.edge_dict[self.__to_h_edge((v1, v2))] = 0
        
        
    def simplifyPDAG(self):
        """
        Simplifies the PDAG by resolving edge directions involved in v-structures. 
        Only edges that are part of v-structures retain their direction. 
        All other edges are left undirected.
        """
        for v1, v2, dir in self.edges():
            if dir > 0 and not self.isFromVStructure((v1, v2)):
                self.elimDirection(v1, v2)
                
            """
            TODO change bidirected edges to unidiredted if needed 
            not implemented because this method will be used to simplify BNs
            with only unidirected edges
            """
            #elif self.edge_dct[edge] == 1 etc.
    
    def undirected(self):
        "Eliminate all the directions of the graph, leaving it's skeleton."
        for v1, v2, dir in self.edges():
            self.elimDirection(v1, v2)
        
        
        
      
    

    
    
    
    def isFromVStructure(self, edge):
        """
        Returns true if the edge  is directed and its direction is part of an v-structure, returns false otherwise
        
        """
        if not self.isDirected(edge[0], edge[1]):
            return False
        
        
            
        orientation = self.getOrientation(edge[0], edge[1])
        
        if orientation == 2:
           return self.__check_v_structure(edge[0], edge[1]) or self.__check_v_structure(edge[1], edge[0])
       
        #Orientation == 1
        return self.__check_v_structure(orientation[0], orientation[1])
             
        
        
       
    def __check_v_structure(self, source, target):
           parents =  self.neighbors(target, mode = "parents") #only directed edges

           return any(
               (node != source) and
               (node not in self.neighbors(source, mode="all"))  # the node is not connected to the source node
               for node in parents
               )
        
        
        
    def __nodeIdToName(self, nodeId): 
        """
        

        Return the name of a node given its iGraph id

        """
        return self.graph.vs[nodeId]["name"]
        
    def __getOrientationVariables(self, edge):
        """
        Given a hashable_edge return the orientation of the edge in self.graph (even if the edge is not oriented in the pdag)

        """
        
        v1, v2 = edge.to_tuple()
        
        eid = self.graph.get_eid(v1 = v1, v2 = v2, directed = False, error = True)
        
        return self.__nodeIdToName(self.graph.es[eid].source),  self.__nodeIdToName(self.graph.es[eid].target)
    
    
     
    
    """
    def extramarital_affairs(self, vertex):
       
        if vertex not in self.names_dict:
            raise ValueError("The given vertex is not present in the graph.")
        
        vertex_num = self.names_dict[vertex]
        targetFathers =  self.graph.neighbors(vertex_num, mode = "in")
        
        
        return [self.graph.vs[node]["name"] for node in targetFathers 
                     if self.is_from_v_structure(hashable_edge(node, vertex_num)) ]
    """
    
    """    
    def eliminate_orientation(self, v1, v2):
        
        
        if v1 not in self.names_dict or v2 not in self.names_dict:
            raise ValueError("Both vertices must belong to the PDAG.")
        
        if not self.__is_in_edge_dict((v1, v2)):
            raise ValueError("Error: indicated edge does not belong to the graph.")
            
        source = self.names_dict[v1]
        target = self.names_dict[v2]
        h_edge = hashable_edge(source, target)
        
        if self.edge_dict[h_edge] == 0 : #if it is already undirected, return
            return 
        elif self.edge_dict[h_edge] == 1 and  self.has_direction(v1, v2): # v1 -> v2 is present
            self.edge_dict[h_edge] = 0
        
        elif self.edge_dict[h_edge] == 2:  #If edge is bidirected
            self.edge_dict[h_edge] = 1
            if  not self.has_direction(v1, v2): #If the underlying graph has the opposite direction
                self.__reverse_edge(v1, v2)
            
        else:  # edge is directed but in the opposite direction (v2->v1)
            return
        
    """


        
        
    def getOrientation(self, v1, v2):
        """
        

        Parameters
        ----------
        v1 : vertex name
        v2 : vertex name
        
        Raises
        ------
        Raises error if either v1 or v1 is not a vertex or if there is not any edge between v1 and v2.

        Returns
        -------
        If there is an edge but if it is undirected return 0, if it is bidirected return 2,
        If there is a directed edge return a tuple with both vertices ordered

        """
        
        if not self.isVertex(v1) or not self.isVertex(v2):
            raise ValueError("Both vertices must belong to the PDAG.")
        
        if not self.hasEdge(v1, v2):
            raise ValueError("Error: The indicated edge does not belong to the graph.")
    
     
        
        h_edge = self.__to_h_edge((v1, v2))
        
        if self.edge_dict[h_edge] == 0 or self.edge_dict[h_edge] == 2:
            return self.edge_dict[h_edge]
        else:
            return self.__getOrientationVariables(h_edge)
        
        
        
    def addDirection(self, v1, v2):
        """
        

        Parameters
        ----------
        v1 : Vertex name
        v2 : Vertex name
        
        Action
        ----------
        If there is an edge between v1 and v2 adds the direction v1 -> v2 (if it is not already in the PDAG).
        
        Note that if the edge v2 <- v1 was already present in the PDAG then PDAG will have a bidirected edge.
        
        Raises
        ------
        Raises error if either v1 or v1 is not a vertex or if there is not any edge between v1 and v2.


        """
       
        if not self.hasEdge(v1, v2):
            raise ValueError(f"The indicated edge, ({v1}, {v2}) does not belong to the graph.")
            
       
            
        
        source = self.names_dict[v1]
        target = self.names_dict[v2]
        
        
        
        if self.edge_dict[hashable_edge(source, target)] == 2:
            #If the edge is bidirected return
            return
        
        eid = self.graph.get_eid(v1 = source, v2 = target, directed = True, error = False)
        
        
        
        
        
        
        if(eid == -1):
            #look if the underlying graph has the opposite direction
            eid = self.graph.get_eid(v1 = target, v2 = source, directed = True, error = False)
            if(eid == -1): # If there is no edge in the underlying igraph raise error
                raise ValueError("Internal error: The indicated edge does not belong to the graph.")
            elif( self.edge_dict[hashable_edge(source, target)] == 1): #before: num2 -> num1 now num2 <->num1
                self.edge_dict[hashable_edge(source, target)] = 2
            else:  # PDAG was undirected and underlying igraph has the opposite direction
                self.graph.reverse_edges([eid])
                self.edge_dict[hashable_edge(source, target)] = 1
        
        else: #if the edge is undirected but it's direction is correct in the underlying igraph
            self.edge_dict[hashable_edge(source, target)] = 1 #If the edge was already directed it doesn't change anything
            
        
        
    """
    def __reverse_edge(self, v1, v2):
        
        Reverse the indicated edge in the underlying graph

        
        
        if not self.__is_in_edge_dict((v1, v2)):
            raise ValueError(f"The indicated edge, ({v1}, {v2}) does not belong to the graph.")
            
            
        source = self.names_dict[v1]
        target = self.names_dict[v2]
            
        eid = self.graph.get_eid(v1 = source, v2 = target, directed = True, error = False)
        
        
        
        if(eid == -1):
            raise ValueError(f"The directed edge {v1}->{v2} does not belong to the graph.")
        else:  
            self.plot(title = "before")
            self.graph.reverse_edges([eid])
            self.plot(title = "after")
    """
        
    def hasEdge(self, v1, v2):
        """
        Return true if there is and edge between v1 and v2, false otherwise.
        
        Raise an error if either v1 or v2 is not a vertex in t

        """
        if not self.isVertex(v1) or not self.isVertex(v2):
            raise ValueError("Both vertices must belong to the graph")
            
        edge = (v1, v2)
        return self.__to_h_edge(edge) in self.edge_dict
    
    def isDirected(self, v1, v2):
        
            
        if not self.isVertex(v1) or not self.isVertex(v2):
            raise ValueError("Vertices must belong to the graph.")
            
        return  self.hasEdge(v1, v2) and bool(self.edge_dict[self.__to_h_edge((v1, v2))])
    
    def hasDirection(self, v1, v2):
        """
        Parameters
        ----------
        v1 : vertex name
        v2 : vertex name
        
        Raises
        ------
        Raises error if either v1 or v2 is not a vertex.

        Returns
        -------
        True if v1->v2 in the graph or if it is bidirected, false otherwise

        """
        
        if self.hasEdge(v1, v2):
            
            
            orientation = self.getOrientation(v1, v2)
            return (orientation == (v1, v2)) or  (orientation == 2)
        
        return False
    
    
    
        
        
        
        
    def meekRule1(self):
        """
        For each X->Z-Y, orient edges to Z->Y
        
        Return True if any undirected edge has been directed, False otherwise.
        """
        
        uEdges = self.edges(mode = "undirected")
        
       
        
        newDirections = []
        
        for v1, v2 in uEdges:
            if self.neighbors(vertex = v1, mode = "parents") != []:
                
                newDirections.append((v1, v2)) 
            elif self.neighbors(vertex = v2, mode = "parents") != []:
                
                newDirections.append((v2, v1 ))
       
        for v1, v2 in newDirections:
            self.addDirection(v1, v2)
            
        return len(newDirections) > 0
    
    def meekRule2(self):
        """
        If X-Y and X is in ancestors(Y) then add the direction X->Y
        Return True if any undirected edge has been directed, False otherwise.

        """
        
        uEdges = self.edges(mode = "undirected")
        
        newDirections = []
        
        for v1, v2 in uEdges:
            if v1 in self.ancestors(v2) :
                newDirections.append((v1, v2)) 
                
            elif v2 in self.ancestors(v1):
                newDirections.append((v2, v1 ))
            
        for v1, v2 in newDirections:
            self.addDirection(v1, v2)
            
        return len(newDirections) > 0
    
    
    def meekRule3(self):
        """
        for each X-Z-Y with X->W, Y->W, and Z-W, orient edges to Z->W
        Return True if any undirected edge has been directed, False otherwise.

        """
        
        uEdges = self.edges(mode = "undirected")
        newDirections = []
        
        for Z, W in uEdges:
            
            
            if len(set(self.neighbors(Z, mode = "undirected")) & set(self.neighbors(W, mode = "parents"))) >= 2:
                newDirections.append((Z, W)) 
            elif len(set(self.neighbors(W, mode = "undirected")) & set(self.neighbors(Z, mode = "parents"))) >= 2:
                newDirections.append((W, Z)) 
            
        for v1, v2 in newDirections:
            self.addDirection(v1, v2)
            
        return len(newDirections) > 0
    
    
    def applyMeek(self):
        """
        Apply meek rules
        """
        
        progress = True
        
        while progress:
            progress = self.meekRule3()  #First, make sure there is not any cycle with rules 3 and 2
            progress = self.meekRule2() or progress
            progress = self.meekRule1() or progress
        
            
    def __ancestors(self, vertex, visited = set()):
        """
        List of ancestors of a given vertex. It allows repetitions.
        """
        
        
        #TODO: Consider using lazy sets and do  not allow repetitions 
        parents = self.neighbors(vertex, mode = "parents")
        visited = visited | {vertex}
        
        if parents == []: # Stop condition
            return [], visited
        
        
        
        
        ancestors = []
        
        
        
        
        for parent in parents:
            if parent not in visited:
                ancestors.append(parent)
                anc, vis = self.__ancestors(parent, visited)
                ancestors = ancestors + anc #Recursive call
                visited = visited | vis
           
        return ancestors, visited
    
    def ancestors(self, vertex):
        """
        List of ancestors of a given vertex without repetitions.
        """
        #TODO: Consider using lazy sets 
        
        if not self.isVertex(vertex):
            raise ValueError("Vertex must belong to the graph.")
        
        ancestors, visited = self.__ancestors(vertex, set())
        
        return ancestors
    
    def isVertex(self, name):
        """
        Check if the given name corresponds to a vertex

        """
        return name in self.names_dict


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
        
    
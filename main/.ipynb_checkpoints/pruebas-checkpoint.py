from igraph import Graph, plot

# Create a simple graph with 5 vertices and edges
g = Graph(edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], directed=True)

# Adding vertex and edge attributes (optional)
g.vs["label"] = range(g.vcount())  # Label vertices with their index
g.vs["color"] = "lightblue"        # Color vertices
g.es["color"] = "grey"             # Color edges

# Plot the graph
plot(g)

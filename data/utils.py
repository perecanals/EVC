import pickle

import networkx as nx

def load_pickle(path):
    """
    Loads pickle from path.

    Parameters
    ----------
    path : string or path-like object
        Path to pickle file.

    Returns
    -------
    pickle_object : object
        Object loaded from pickle.

    """
    with open(path, "rb") as f:
        pickle_object = pickle.load(f)

    return pickle_object

def node_transform(graph):
    """
    Applies node transform to networkx graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph in original form, where edges encode vessels.
    
    Returns
    -------
    transformed_graph : networkx.Graph
        Graph in node form, where nodes encode vessels.
        
    """
    # Declare empty necessary lists and dicts
    new_nodes = [] # list with new nodes
    edges_to_nodes = {} # dict linking former edges to new nodes (undirected)
    new_nodes_to_old_edges = {} # we wil use this dict to track edge attributes back
    new_node = 0 # auxiliary

    # We store a new node for each edge, and create a dict to pass the edge attributes from
    # the old graph to the nodes of the new graph
    for edge in graph.edges:
        new_nodes.append(new_node)
        edges_to_nodes[edge] = new_node
        edges_to_nodes[(edge[1], edge[0])] = new_node
        new_nodes_to_old_edges[new_node] = edge
        new_node += 1

    new_edges = [] # list with new edges
    
    # We define edges of the new graph as links between immediately neighbouring vessels
    for node in graph.nodes:
        if len(graph.edges(node)) > 1:
            # For each node connected to multiple edges, we create an auxiliary list with connected nodes
            edge_list_aux = [edges_to_nodes[edge] for edge in graph.edges(node)]
            # We iterate over all nodes except for the last one to connect them once
            for idx, src in enumerate(edge_list_aux):
                for dst in edge_list_aux[idx + 1:]:
                    new_edges.append([src, dst])

    # We create a new empty graph
    transformed_graph = nx.Graph()

    # We add nodes, node attributes (former edge attributes) and edges
    for node in new_nodes:
        transformed_graph.add_node(node)
        # Also tranfer the edge attributes to node attributes
        for attribute_key in graph[new_nodes_to_old_edges[node][0]][new_nodes_to_old_edges[node][1]].keys():
            transformed_graph.nodes[node][attribute_key] = graph[new_nodes_to_old_edges[node][0]][new_nodes_to_old_edges[node][1]][attribute_key]
     
    # Finally add edges       
    for edge in new_edges:
        transformed_graph.add_edge(edge[0], edge[1])

    return transformed_graph
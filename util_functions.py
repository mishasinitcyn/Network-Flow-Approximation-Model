import numpy as np
import random
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from collections import deque
import math

max_vertex_count, max_edge_count = 5,5
max_flow_value = 10

def generate_source_sink_network_graph(vertex_count, edge_count):
    if edge_count > (vertex_count - 2) * (vertex_count - 1):  # maximum edges for a directed graph with a single source and sink node
        raise ValueError("Edge count cannot be greater than the maximum possible edges for the given vertex count")

    matrix = np.zeros((vertex_count, vertex_count))

    # Mark the source and sink indices
    source = 0
    sink = vertex_count - 1
    
    edges_added = 0
    while edges_added < edge_count:
        i = random.randint(0, vertex_count - 1)
        j = random.randint(0, vertex_count - 1)

        # Ensure there's no self-loop, reverse edge, and edge already exists
        # Also ensure the source is not the destination and sink is not the origin
        if i != j and matrix[i][j] == 0 and matrix[j][i] == 0 and not (i == sink or j == source):
            matrix[i][j] = random.randint(1, max_flow_value)  # Assign random weight to the edge
            edges_added += 1

    return matrix, source, sink

def bfs(residual_graph, source, sink, parent):
        visited = [False] * len(residual_graph)
        queue = deque()

        queue.append(source)
        visited[source] = True

        while queue:
            u = queue.popleft()

            for ind, val in enumerate(residual_graph[u]):
                if not visited[ind] and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
                    if ind == sink:
                        return True
        return False

def ford_fulkerson_optimized_graph(graph, source, sink):
    residual_graph = graph.copy().tolist()
    parent = [-1] * len(graph)
    
    # Store the maximum flow
    max_flow = 0

    while bfs(residual_graph, source, sink, parent):
        path_flow = float("Inf")
        s = sink
        while s != source:
            path_flow = min(path_flow, residual_graph[parent[s]][s])
            s = parent[s]

        max_flow += path_flow

        v = sink
        while v != source:
            u = parent[v]
            residual_graph[u][v] -= path_flow
            residual_graph[v][u] += path_flow
            v = parent[v]

    # Create the optimized graph without reverse edges
    optimized_graph = np.zeros_like(graph)
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            if graph[i][j] > 0:  # If edge exists in the original graph
                flow = graph[i][j] - residual_graph[i][j]
                if flow > 0:  # If positive flow, add to optimized_graph
                    optimized_graph[i][j] = flow

    return max_flow, optimized_graph

def get_max_flow(graph, source, sink):
    residual_graph = graph.copy().tolist()
    parent = [-1] * len(graph)
    
    # Store the maximum flow
    max_flow = 0

    while bfs(residual_graph, source, sink, parent):
        path_flow = float("Inf")
        s = sink
        while s != source:
            path_flow = min(path_flow, residual_graph[parent[s]][s])
            s = parent[s]

        max_flow += path_flow

        v = sink
        while v != source:
            u = parent[v]
            residual_graph[u][v] -= path_flow
            residual_graph[v][u] += path_flow
            v = parent[v]
    return max_flow

def draw_graph_from_adj_matrix(adj_matrix):
    G = nx.DiGraph()

    for i, row in enumerate(adj_matrix):
        for j, weight in enumerate(row):
            if weight != 0:  # If there's an edge between vertices i and j
                G.add_edge(i, j, weight=weight)

    pos = nx.spring_layout(G)
    edge_labels = {(i, j): w["weight"] for i, j, w in G.edges(data=True)}

    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

def save_matrix_to_file(matrix, file_name):
    np.savetxt(file_name, matrix, delimiter=',', fmt='%d')

def generate_and_save_matrices(n):
    for i in range(n):
        graph, source, sink = generate_source_sink_network_graph(max_vertex_count, max_edge_count)
        flow, optimized_graph = ford_fulkerson_optimized_graph(graph, source, sink)
        
        if(flow):
            print(flow, optimized_graph)
            save_matrix_to_file(graph, f'adjacency_matrix_{i+1}.csv')
            save_matrix_to_file(optimized_graph, f'optimized_matrix_{i+1}.csv')
        

def save_data_to_pickle(data, file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def generate_and_save_matrices_in_pickle(n):
    adjacency_matrices = []
    optimized_matrices = []
    flows = []
    
    for i in range(n):
        graph, source, sink = generate_source_sink_network_graph(max_vertex_count, max_edge_count)
        flow, optimized_graph = ford_fulkerson_optimized_graph(graph, source, sink)
        
        if(flow):
            adjacency_matrices.append(graph)
            optimized_matrices.append(optimized_graph)
            flows.append(flow)
            save_data_to_pickle(adjacency_matrices, 'adjacency_matrices.pickle')
            save_data_to_pickle(optimized_matrices, 'optimized_matrices.pickle')
            save_data_to_pickle(flows, 'flows.pickle')
    
    
def load_data_from_pickle(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)
    
def round_down_small_values(matrix):
    return np.array(
        [
            [0 if abs(value) < 0.1 else value for value in row]
            for row in matrix
        ],
        dtype=np.float32
    )

def print_matrices(graph, optimized_graph, predicted_optimized_matrix, predicted_optimized_matrix_rounded):
    print('Original Adjacency Matrix:\n', graph, '\n')
    print('Optimized Matrix:\n', optimized_graph, '\n')
    print('Predicted Optimized Matrix:\n', predicted_optimized_matrix, '\n')
    print('Predicted Optimzied Matrix Rounded:\n', predicted_optimized_matrix_rounded, '\n')

def print_flow_comparison(predicted_flow, real_flow, flow_difference):
    print('Predicted Flow:', predicted_flow)
    print('Real Flow:', real_flow)
    print('Difference:', flow_difference)
    print('Accuracy:', 1-flow_difference)

def print_graph_comparison(graph, optimized_graph, predicted_optimized_matrix_rounded, predicted_optimized_matrix):
    print('Original Graph')
    draw_graph_from_adj_matrix(graph)
    print('Optimized Graph')
    draw_graph_from_adj_matrix(optimized_graph)
    print('Predicted Optimized Graph (Rounded)')
    draw_graph_from_adj_matrix(predicted_optimized_matrix_rounded)
    print('Predicted Optimized Graph (Raw)')
    draw_graph_from_adj_matrix(predicted_optimized_matrix)
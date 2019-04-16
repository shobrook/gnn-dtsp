import networkx as nx
import random as rand
from tsp_solver.greedy import solve_tsp

def create_random_graph(node_range=(5, 9), prob=0.25, weight_range=(0, 10)):
    n_nodes = rand.randint(*node_range)

    G = nx.complete_graph(n_nodes)
    for u, v, w in G.edges(data=True):
        u_deg, v_deg = G.degree(u), G.degree(v)
        if u_deg - 1 >= n_nodes / 2 and v_deg - 1 >= n_nodes / 2:
            if rand.random() < prob:
                G.remove_edge(u, v)
            else:
                w["weight"] = rand.randint(*weight_range)

    return G

def get_tsp_solution(graph):
    adj_matrix = nx.to_numpy_matrix(graph)
    return solve_tsp(adj_matrix) # Returns list of vertex indices

    # TODO: Return networkx graph with labeled edges

def create_example(graph):
    pass

def create_dataset(n_examples=20000):
    inputs, targets = [], []
    for _ in range(n_examples):
        input_graph = create_random_graph()
        target_graph = get_tsp_solution(input_graph)

        inputs.append(input_graph)
        targets.append(target_graph)

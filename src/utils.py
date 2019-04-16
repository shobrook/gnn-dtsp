#########
# GLOBALS
#########


import networkx as nx
import random as rand
import matplotlib.pyplot as plt
from tsp_solver.greedy import solve_tsp as solve


#########
# HELPERS
#########


def create_random_graph(node_range=(5, 9), prob=0.25, weight_range=(1, 10)):
    n_nodes = rand.randint(*node_range)

    G = nx.complete_graph(n_nodes)
    H = G.copy()
    for u, v, w in G.edges(data=True):
        u_deg, v_deg = H.degree(u), H.degree(v)
        if u_deg - 1 >= n_nodes / 2 and v_deg - 1 >= n_nodes / 2:
            if rand.random() < prob:
                H.remove_edge(u, v)
            else:
                H[u][v]["weight"] = rand.randint(*weight_range)

    return H


def solve_tsp(graph):
    adj_matrix = nx.to_numpy_matrix(graph)
    hamil_path = solve(adj_matrix)

    path_edges = [(hamil_path[i], hamil_path[i + 1])
                  for i in range(len(hamil_path) - 1)]
    path_edges.append((hamil_path[-1], hamil_path[0]))

    for u, v in graph.edges():
        graph[u][v]["solution"] = int(
            any(u == src and v == targ for src, targ in path_edges))

    return graph


def visualize_network(G, filename, dpi=1000):
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, "solution")

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw(G, pos)

    plt.savefig("../figures/" + filename, dpi=dpi)
    plt.close()


#########
# EXPORTS
#########


# TODO: Put things here

if __name__ == "__main__":
    G = create_random_graph()
    visualize_network(G, "original_graph")
    solved_G = solve_tsp(G)
    visualize_network(solved_G, "solved_graph")

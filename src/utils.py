import networkx as nx
import random as rand
from tsp_solver.greedy import solve_tsp

def create_random_graph(node_range=(5, 9), prob=0.25, weight_range=(1, 10)):
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



def to_one_hot(indices, max_value, axis=-1):
  one_hot = np.eye(max_value)[indices]
  if axis not in (-1, one_hot.ndim):
    one_hot = np.moveaxis(one_hot, -1, axis)
  return one_hot


def graph_to_input_target(graph):
  """Returns 2 graphs with input and target feature vectors for training.

  Args:
    graph: An `nx.Graph` instance.

  Returns:
    The input `nx.Graph` instance.
    The target `nx.Graph` instance.

  Raises:
    ValueError: unknown node type
  """

  def create_feature(attr, fields):
    return np.hstack([np.array(attr[field], dtype=float) for field in fields])

  #input_node_fields = ("solution",)
  input_edge_fields = ("weight",)
  #target_node_fields = ("solution",)
  target_edge_fields = ("solution",)

  input_graph = graph.copy()
  target_graph = graph.copy()

  solution_length = 0
  for node_index in graph.nodes():
    input_graph.add_node(node_index)
    target_graph.add_node(node_index)

  for receiver, sender, features in graph.edges(data=True):
    input_graph.add_edge(
        sender, receiver, features=create_feature(features, input_edge_fields))
    target_edge = to_one_hot(
        create_feature(features, target_edge_fields).astype(int), 2)[0]
    target_graph.add_edge(sender, receiver, features=target_edge)
    solution_length += features["weight"] * features["solution"]

  input_graph.graph["features"] = np.array([0.0])
  target_graph.graph["features"] = np.array([solution_length], dtype=float)

  return input_graph, target_graph

def generate_networkx_graphs(num_graphs, node_range=(5, 9), prob=0.25, weight_range=(1, 10)):
  """Generate graphs for training.

  Args:
    num_graphs: number of graphs to generate
    num_range: a 2-tuple with the [lower, upper) number of nodes per
      graph
    prob: the probability of removing an edge between any two nodes
    weight_range: a 2-tuple with the [lower, upper) weight to randomly assign
        to (non-removed) edges

  Returns:
    input_graphs: The list of input graphs.
    target_graphs: The list of output graphs.
    graphs: The list of generated graphs.
  """

  input_graphs = []
  target_graphs = []
  graphs = []

  for i in range(num_graphs):
    graph = create_random_graph(node_range, prob, weight_range)
    graph = get_tsp_solution(graph)
    input_graph, target_graph = graph_to_input_target(graph)
    input_graphs.append(input_graph)
    target_graphs.append(target_graph)
    graphs.append(graph)
  return input_graphs, target_graphs, graphs

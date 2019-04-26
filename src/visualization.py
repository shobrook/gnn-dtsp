#########
# GLOBALS
#########


import tensorflow as tf
import numpy as np
import pandas as pd
import time
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from graph_nets import utils_np
from graph_nets.demos import models

# Local modules
import utils


#########
# HELPERS
#########

# small changes to accommodate networkxs
# now target and output should be data_dicts
# returns arrays instead of floats
def compute_accuracy(target, output, use_nodes=True, use_edges=False):
    if not use_nodes and not use_edges:
        raise ValueError("Nodes or edges (or both) must be used")

    # tdds = utils_np.graphs_tuple_to_data_dicts(target)
    # odds = utils_np.graphs_tuple_to_data_dicts(output)

    cs, ss = [], []
    for td, od in zip(target, output):

        xe = np.argmax(td["edges"], axis=-1)
        ye = np.argmax(od["edges"], axis=-1)

        c = [xe == ye] if use_edges else []
        c = np.concatenate(c, axis=0)

        s = np.all(c)
        cs.append(c)
        ss.append(s)

    # correct = np.mean(np.concatenate(cs, axis=0))
    # solved = np.mean(np.stack(ss))

    correct = np.concatenate(cs, axis=0)
    solved = np.stack(ss)

    return correct, solved


def visualize_network(G, H, filename, dpi=1000):
    # G is input, H is target/output

    G = G.to_undirected()
    H = H.to_undirected()
    pos = nx.spring_layout(G)

    edge_weights = nx.get_edge_attributes(G, "features")
    # new_weights = [int(edge_weight) for edge_weight in edge_weights]

    edge_labels = list(nx.get_edge_attributes(H, "features").values())
    new_labels = [np.argmax(edge_label) for edge_label in edge_labels]

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
    nx.draw(G, pos, edge_color=labels_to_colors(new_labels, 'r', 'k'))

    plt.savefig("../figures/" + filename, dpi=dpi)
    plt.close()

# def visualize_output(G, H, filename, dpi=1000):
#     # G is input, H is output
#     G = G.to_undirected()
#     pos = nx.spring_layout(G)
#     edge_labels = list(nx.get_edge_attributes(G, "features").values())
#     new_labels = [np.argmax(edge_label) for edge_label in edge_labels]
#
#     # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     nx.draw(G, pos, edge_color=labels_to_colors(new_labels, 'r', 'k'))
#
#     plt.savefig("../figures/" + filename, dpi=dpi)
#     plt.close()


def labels_to_colors(labels, col1, col2):
    colors = [col1 * label + col2 * (1-label)
        for label in labels]
    return colors

###########
# VISUALIZE
###########


# read file
file = open("../data/pickles/test_results.pkl", "rb")
dict = pickle.load(file)

outputs = dict["outputs"]
targets = dict["targets"]
inputs = dict["inputs"]
n = len(outputs)

outputs_dd = [utils_np.networkx_to_data_dict(output) for output in outputs]
targets_dd = [utils_np.networkx_to_data_dict(target) for target in targets]
correct, solved = compute_accuracy(outputs_dd, targets_dd, use_edges=True)

for i in range(200):
    input = inputs[i]
    output = outputs[i]
    target = targets[i]

    if solved[i] == True:
        visualize_network(input, output, "correct/output{}.png".format(i))
        visualize_network(input, target, "correct/target{}.png".format(i))
    else:
        visualize_network(input, output, "incorrect/output{}.png".format(i))
        visualize_network(input, target, "incorrect/target{}.png".format(i))

# print(outputs[0].edges(data=True))
# print('----------------')
# print(targets)
# print('----------------')
# print(inputs[0].edges(data=True))

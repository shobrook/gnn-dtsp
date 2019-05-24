#!/usr/bin/env python3
from IPython import embed
import pickle
import networkx as nx
import numpy as np

def is_solution_edge(feature_vector):
    # if it's solution, one hot vector is (0, 1)
    if feature_vector[1] > feature_vector[0]: return True
    return False

# return a dictionary with vertices as keys
# each item contains its neighbor vertices connected
# by the labelled edges
def get_neighbor_dict(my_graph):
    ret_dict = {}
    edge_list = [i for i in my_graph.edges(data = True)]
    for i in edge_list:
        if is_solution_edge(i[2]["features"]):
            try:
                ret_dict[i[0]].append(i[1])
            except KeyError:
                ret_dict[i[0]] = [i[1]]
            try:
                ret_dict[i[1]].append(i[0])
            except KeyError:
                ret_dict[i[1]] = [i[0]]
    return ret_dict

def hamiltonian_path_check(my_graph):
    neighbor_dict = get_neighbor_dict(my_graph)
    my_vertices = [i for i in my_graph.nodes()]
    visited_dict = [False] * len(my_vertices)
    # since hamiltonian path is closed; choose a random vertex to start
    # actually nvm, maybe just use the first vertex...
    start_vertex = my_vertices[0]
    # test 1: every vertex is in the neighbor_dict
    for v in my_vertices:
        try:
            neighbor_dict[v]
        except KeyError:
            return False, neighbor_dict
    # test 2: every vertex connects to two and only two other vertices
    for k, i in neighbor_dict.items():
        if len(i) != 2: return False, neighbor_dict
    # test 3: start from a node and perform the walk, we will end up at the same node
    current_vertex = start_vertex
    next_vertex = neighbor_dict[start_vertex][0]
    while True:
        if visited_dict[current_vertex]: print("WTF???")
        visited_dict[current_vertex] = True
        if next_vertex == start_vertex: break
        if neighbor_dict[next_vertex][0] == current_vertex:
            current_vertex = next_vertex
            next_vertex = neighbor_dict[next_vertex][1]
        else:
            current_vertex = next_vertex
            next_vertex = neighbor_dict[next_vertex][0]
    for mark in visited_dict:
        if mark == False: return False, neighbor_dict
    return True, neighbor_dict

def get_ori_weight(input_graph, u, v):
    return input_graph.get_edge_data(u, v)["weight"]

def evaluate(input_graphs, output_graphs, target_graphs):
    assert len(output_graphs) == len(target_graphs)
    assert len(input_graphs) == len(output_graphs)
    assert isinstance(output_graphs, list)
    assert isinstance(target_graphs, list)

    n_num = len(output_graphs)

    # sort graph by correctly labelled edges
    # seperate output into bad/average/good sets

    is_hamiltonian_path_count = 0
    output_solution_len_list = []
    target_solution_len_list = []

    non_output_soln_len_list = []
    non_target_soln_len_list = []

    for i in range(len(output_graphs)):
        ret, my_dict = hamiltonian_path_check(output_graphs[i])
        if ret:
            is_hamiltonian_path_count += 1
            # get output solution length
            output_soln_len = 0
            for u in my_dict.keys():
                for v in my_dict[u]:
                    if u > v: continue
                    output_soln_len += get_ori_weight(input_graphs[i], u, v)
            # get target solution length
            tar_soln_len = 0
            for e in input_graphs[i].edges(data = True):
                if e[2]["solution"] == 1:
                    tar_soln_len += e[2]["weight"]
            output_solution_len_list.append(output_soln_len)
            target_solution_len_list.append(tar_soln_len)
        else:
            # it's not hamiltonian... but we want the error
            output_soln_len = 0
            for u in my_dict.keys():
                for v in my_dict[u]:
                    if u > v: continue
                    output_soln_len += get_ori_weight(input_graphs[i], u, v)
            # get target solution length
            tar_soln_len = 0
            for e in input_graphs[i].edges(data = True):
                if e[2]["solution"] == 1:
                    tar_soln_len += e[2]["weight"]
            non_output_soln_len_list.append(output_soln_len)
            non_target_soln_len_list.append(tar_soln_len)
    # calculate difference between output and target graphs
    print("{0} out of {1} solutions are Hamiltonian.".format(is_hamiltonian_path_count, n_num))
    output_solution_len_list = np.array(output_solution_len_list)
    target_solution_len_list = np.array(target_solution_len_list)
    diff_ = output_solution_len_list - target_solution_len_list
    print("Among Hamiltonian cycles found by the GNN, the average difference in solution graph with ground truth is {0}".format(np.mean(diff_)))
    output_vec = np.array(non_output_soln_len_list)
    target_vec = np.array(non_target_soln_len_list)
    diff_ = output_vec - target_vec
    print("Among those that were incorrectly labelled, the average error in solution length is {0}".format(np.mean(diff_)))
    total_weight = 0
    for g in input_graphs:
        for e in g.edges(data = True):
            total_weight += e[2]["weight"]
    average_weight = total_weight / n_num
    print("On average, the total weight of each graph we used is {0}. So the average relative error is {1}".format(average_weight, np.mean(diff_) / average_weight))

def main(file_path = "../data/pickles/test_results.pkl"):
    graphs_dict = pickle.load(open(file_path, 'rb'))
    evaluate(graphs_dict["inputs"], graphs_dict["outputs"], graphs_dict["targets"])

if __name__ == '__main__':
    main()

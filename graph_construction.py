__author__ = "shekkizh"

import numpy as np
from non_neg_qpsolver import non_negative_qpsolver
import scipy.sparse as sparse

def knn_graph(G, mask, knn_param, reg):
    '''
    Function to generate KNN graph given similarity matrix and mask
    :param G: Similarity matrix
    :param mask: each row corresponds to the neighbors to be considered for NNK optimization
    :param knn_param: maximum number of neighbors for each node
    :param reg: weights below this threshold are removed (set to 0)
    :return: Adjacency matrix of size num of nodes x num of nodes
    '''
    num_of_nodes = G.shape[0]
    neighbor_indices = np.zeros((num_of_nodes, knn_param))
    weight_values = np.zeros((num_of_nodes, knn_param))
    for node_i in range(num_of_nodes):
        non_zero_index = np.array(mask[node_i, :])
        non_zero_index = np.delete(non_zero_index, np.where(non_zero_index == node_i))
        weight_values[node_i, :] = G[non_zero_index, node_i]
        neighbor_indices[node_i, :] = non_zero_index
    row_indices = np.expand_dims(np.arange(0, num_of_nodes), 1)
    row_indices = np.tile(row_indices, [1, knn_param])
    adjacency = sparse.coo_matrix((weight_values.ravel(), (row_indices.ravel(), neighbor_indices.ravel())),
                                  shape=(num_of_nodes, num_of_nodes)).toarray()
    ## Symmetrize adjacency matrix
    adjacency = np.maximum(adjacency, adjacency.T)
    return adjacency

def nnk_graph(G, mask, knn_param, reg=1e-6):
    '''
    Function to generate NNK graph given similarity matrix and mask
    :param G: Similarity matrix
    :param mask: each row corresponds to the neighbors to be considered for NNK optimization
    :param knn_param: maximum number of neighbors for each node
    :param reg: weights below this threshold are removed (set to 0)
    :return: Adjacency matrix of size num of nodes x num of nodes
    '''
    num_of_nodes = G.shape[0]
    neighbor_indices = np.zeros((num_of_nodes, knn_param))
    weight_values = np.zeros((num_of_nodes, knn_param))
    error_values = np.zeros((num_of_nodes, knn_param))

    for node_i in range(num_of_nodes):
        non_zero_index = np.array(mask[node_i, :])
        non_zero_index = np.delete(non_zero_index, np.where(non_zero_index == node_i))
        G_i = G[np.ix_(non_zero_index, non_zero_index)]
        g_i = G[non_zero_index, node_i]
        x_opt, check = non_negative_qpsolver(G_i, g_i, g_i, reg)
        error_values[node_i, :] = G[node_i, node_i] - 2 * np.dot(x_opt, g_i) + np.dot(x_opt, np.dot(G_i, x_opt))
        weight_values[node_i, :] = x_opt
        neighbor_indices[node_i, :] = non_zero_index

    row_indices = np.expand_dims(np.arange(0, num_of_nodes), 1)
    row_indices = np.tile(row_indices, [1, knn_param])
    adjacency = sparse.coo_matrix((weight_values.ravel(), (row_indices.ravel(), neighbor_indices.ravel())),
                                  shape=(num_of_nodes, num_of_nodes)).toarray()
    error = sparse.coo_matrix((error_values.ravel(), (row_indices.ravel(), neighbor_indices.ravel())),
                              shape=(num_of_nodes, num_of_nodes))
    error_index = sparse.find(error > error.T)
    adjacency[error_index[0], error_index[1]] = 0
    adjacency = np.maximum(adjacency, adjacency.T)
    return adjacency

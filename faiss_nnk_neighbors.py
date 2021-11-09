__author__ = "shekkizh"

import faiss
import numpy as np
from non_neg_qpsolver import non_negative_qpsolver
import warnings


#%%
def nnk_neighbors(train_features, queries, top_k=50, use_gpu=False):
    """
    NNK nieghborhood definition with normalized cosine kernel (range \in [0,1]) 
    train_features: shape [n_train, d] Feature vectors of available dataset
    queries: shape [n_queries, d] Query feature vectors for which neighbors are to be selected
    top_k: Maximum number of neighbors to select
    use_gpu: Boolean flag to signal use of GPU for neighbor search
    """
    dim = train_features.shape[1]
    normalized_features = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(dim)
    if use_gpu:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(normalized_features)

    normalized_queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    n_queries = queries.shape[0]

    weight_values = np.zeros((n_queries, top_k))
    similarities, indices = index.search(normalized_queries, top_k)

    for ii, x_test in enumerate(normalized_queries):
        neighbor_indices = indices[ii, :]
        x_support = normalized_features[neighbor_indices]
        g_i = 0.5 + 0.5*similarities[ii, :]
        G_i = 0.5 + 0.5*np.dot(x_support, x_support.T)
        x_opt, check = non_negative_qpsolver(G_i, g_i, g_i, x_tol=1e-10)
        # x_opt = g_i
        non_zero_indices = np.nonzero(x_opt)
        x_opt = x_opt / np.sum(x_opt[non_zero_indices])
        weight_values[ii, :] = x_opt
        if ii % 10000 == 0:
            print(f"{ii}/{n_queries} processed...")


    return weight_values, indices

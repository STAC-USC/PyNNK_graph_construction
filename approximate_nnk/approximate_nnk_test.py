__author__ = "shekkizh"
"""Demo of Approximate NNK neighborhood solver with normalized cosine kernel similarity"""

import argparse
import os
import random
import time

import faiss
import faiss.contrib.torch_utils  # import needed to work with torch tensors that are in GPU
import numpy as np
import torch

from approximate_nnk_solver import approximate_nnk

# import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# import matplotlib
#
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

seed_value = 4629
os.environ["PYTHONHASHSEED"] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

parser = argparse.ArgumentParser(description='Approximate NNK demo')
parser.add_argument('--logs_dir', default='./logs/')
parser.add_argument("--data_dir", default="./datasets/")
parser.add_argument('--top_k', default=50, type=int, help="initial no. of neighbors")
parser.add_argument('--max_iter', default=10, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
parser.add_argument('--nouse_gpu', dest='use_gpu', action='store_false')
parser.set_defaults(use_gpu=True)


class FAISS_Search:
    def __init__(self, dim, use_gpu=False):
        self.matrix = None
        if use_gpu:
            res = faiss.StandardGpuResources()
            # # res.noTempMemory()
            self.index = faiss.GpuIndexFlat(res, dim, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(dim)

    @staticmethod
    def _faiss_preprocess(X):
        return X.contiguous()

    def add(self, matrix):
        self.matrix = self._faiss_preprocess(matrix)
        self.index.add(self.matrix)

    def search(self, queries, top_k):
        queries = self._faiss_preprocess(queries)
        similarities, indices = self.index.search(queries, top_k)
        return similarities, indices

    def get_support(self):
        return self.matrix

    def reset(self):
        self.index.reset()
        del self.matrix


@torch.no_grad()
def _process_data(X):
    X = torch.nn.functional.normalize(X, p=2, dim=1)
    return torch.cat((X, torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)), axis=1)


# %%
def nnk_directed_graph_demo(argv=None):
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    # %%
    cuda_available = torch.cuda.is_available() and args.use_gpu
    device = torch.device("cuda" if cuda_available else "cpu")

    # %% Using random data for testing
    d = 64  # dimension
    nb = 100000  # database size
    nq = 10000  # nb of queries
    xb_torch = _process_data(torch.rand(nb, d, dtype=torch.float32).to(device))
    xq_torch = _process_data(torch.rand(nq, d, dtype=torch.float32).to(device))

    start_time = time.time()
    faiss_search = FAISS_Search(dim=d+1, use_gpu=cuda_available)
    faiss_search.add(xb_torch)

    similarities, indices = faiss_search.search(xq_torch, top_k=args.top_k)
    support_matrix = faiss_search.get_support()[indices]
    support_similarites = torch.bmm(support_matrix, support_matrix.transpose(1, 2))
    weight_values, error = approximate_nnk(0.5 * support_similarites, 0.5 * similarities, 0.5 * similarities,
                                           x_tol=1e-6,
                                           num_iter=100, eta=0.05)
    nnk_time = time.time() - start_time
    print(f"Neighborhood sparsity: {torch.count_nonzero(weight_values>1e-6)}, Time taken: {nnk_time}")
    return weight_values, error
    # row_indices = np.expand_dims(np.arange(0, nq), 1)
    # row_indices = np.tile(row_indices, [1, args.top_k]).ravel()
    # W = torch.sparse_coo_tensor(np.stack(row_indices, (torch.ravel(indices).cpu().numpy()), axis=0),
    #                                  weight_values.ravel(), (nq, nb), dtype=torch.float32)

    # return W


if __name__ == "__main__":
    args = parser.parse_args()
    weight_values, error = nnk_directed_graph_demo(args)

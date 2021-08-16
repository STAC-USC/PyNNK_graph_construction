from __future__ import division

__author__ = "shekkizh"
"""Simple demo of NNK graph construction"""
from absl import flags, app
import os, time
import numpy as np
import graph_utils as utils
from graph_construction import knn_graph, nnk_graph

FLAGS = flags.FLAGS

# %% Experiment associated settings
flags.DEFINE_string('data_dir', 'datasets/', 'dataset directory to import data from')
flags.DEFINE_string('logs_dir', 'logs/', 'log directory to save results and outputs')
flags.DEFINE_string('dataset', 'toy_points', 'dataset to use for experiment')
# %% Algorithm specific parameters
flags.DEFINE_integer('knn_param', 5, 'number of neighbors to use for NNK')
# flags.DEFINE_integer('sigma_k', 10, 'choice of "k"th neighbor for sigma calculation')
flags.DEFINE_float('thresh', 1e-6, 'threshold corresponding to minimum value of edge weights')
flags.DEFINE_string('metric', 'rbf', 'Similarity metric to use for finding neighbors: cosine, rbf')
flags.DEFINE_float('p', 2, 'type of Lp distance to use (if used)')


# %%
def get_toy_data(load_from_file=False):
    n_samples = 100
    n_dim = 2
    n_clusters = 2
    dataset_generator = utils.Create_Data(n_samples, n_dim, n_clusters)
    data_file = os.path.join(FLAGS.data_dir, FLAGS.dataset,
                             ("data_%d_samples_%d_dim_%d_clusters.npz" % (n_samples, n_dim, n_clusters)))
    if load_from_file and os.path.exists(data_file):
        data = np.load(data_file)
        return data['X'], data['y']
    data_dir = os.path.join(FLAGS.data_dir, FLAGS.dataset)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    X, y = dataset_generator.get_dataset(FLAGS.dataset)
    np.savez(data_file, X=X, y=y)
    return X, y


# %%
def main(argv=None):
    model_output_folder = os.path.join(FLAGS.logs_dir, FLAGS.dataset)
    if not os.path.exists(model_output_folder):
        os.makedirs(model_output_folder)
    X, y = get_toy_data(load_from_file=True)
    if FLAGS.metric == 'cosine':
        X_normalized = X / np.linalg.norm(X, axis=1)[:, None]
        G = 0.5 + np.dot(X_normalized, X_normalized.T) / 2.0
        knn_mask = utils.create_directed_KNN_mask(D=G, knn_param=FLAGS.knn_param, D_type='similarity')
    elif FLAGS.metric == 'rbf':
        D = utils.create_distance_matrix(X=X, p=FLAGS.p)
        knn_mask = utils.create_directed_KNN_mask(D=D, knn_param=FLAGS.knn_param, D_type='distance')
        sigma = np.mean(D[:, knn_mask[:, -1]]) / 3
        G = np.exp(-(D ** 2) / (2 * sigma ** 2))
    else:
        raise Exception("Unknown metric: " + FLAGS.metric)
    start_time = time.time()
    W_knn = knn_graph(G, knn_mask, FLAGS.knn_param, FLAGS.thresh)
    knn_time = time.time() - start_time
    utils.plot_graph(W_knn, X,
                     filename=os.path.join(model_output_folder, "KNN_%d_%s" % (FLAGS.knn_param, FLAGS.metric)),
                     vertex_color=y)
    start_time = time.time()
    W_nnk = nnk_graph(G, knn_mask, FLAGS.knn_param, FLAGS.thresh)
    nnk_time = time.time() - start_time
    utils.plot_graph(W_nnk, X,
                     filename=os.path.join(model_output_folder, "NNK_%d_%s" % (FLAGS.knn_param, FLAGS.metric)),
                     vertex_color=y)
    print("KNN - %f s, %d edges,  NNK - %f s, %d edges" % (
        knn_time, W_knn.nnz / 2, nnk_time, W_nnk.nnz / 2))

if __name__ == "__main__":
    app.run(main)

from __future__ import division

__author__ = "shekkizh"

r"""
THIS CODE WAS TAKEN FROM PyGSP SOURCE CODE AND MODIFIED FOR OUR PURPOSES

The :mod:`pygsp.plotting` module implements functionality to plot PyGSP objects
with a `matplotlib <https://matplotlib.org>.

Most users won't use this module directly.
Graphs (from :mod:`pygsp.graphs`) are to be plotted with
:meth:`pygsp.graphs.Graph.plot` and

"""

import numpy as np
from scipy import sparse
import functools


_plt_figures = []


def _import_plt():
    try:
        import matplotlib as mpl
        from matplotlib import pyplot as plt
        from mpl_toolkits import mplot3d
    except Exception as e:
        raise ImportError('Cannot import matplotlib. Choose another backend '
                          'or try to install it with '
                          'pip (or conda) install matplotlib. '
                          'Original exception: {}'.format(e))
    return mpl, plt, mplot3d


def _plt_handle_figure(plot):
    r"""Handle the common work (creating an axis if not given, setting the
    title) of all matplotlib plot commands."""

    # Preserve documentation of plot.
    @functools.wraps(plot)
    def inner(obj, **kwargs):

        # Create a figure and an axis if none were passed.
        if kwargs['ax'] is None:
            _, plt, _ = _import_plt()
            fig = plt.figure()
            global _plt_figures
            _plt_figures.append(fig)

            if (hasattr(obj, 'coords') and obj.coords.ndim == 2 and
                    obj.coords.shape[1] == 3):
                kwargs['ax'] = fig.add_subplot(111, projection='3d')
            else:
                kwargs['ax'] = fig.add_subplot(111)

        title = kwargs.pop('title')

        plot(obj, **kwargs)

        kwargs['ax'].set_title(title)

        try:
            fig.show(warn=False)
        except NameError:
            # No figure created, an axis was passed.
            pass

        return kwargs['ax'].figure, kwargs['ax']

    return inner


def close_all():
    r"""Close all opened windows."""

    # Windows can be closed by releasing all references to them so they can be
    # garbage collected. May not be necessary to call close().
    global _plt_figures
    for fig in _plt_figures:
        _, plt, _ = _import_plt()
        plt.close(fig)
    _plt_figures = []


def show(*args, **kwargs):
    r"""Show created figures, alias to plt.show().

    By default, showing plots does not block the prompt.
    Calling this function will block execution.
    """
    _, plt, _ = _import_plt()
    plt.show(*args, **kwargs)


def close(*args, **kwargs):
    r"""Close last created figure, alias to plt.close()."""
    _, plt, _ = _import_plt()
    plt.close(*args, **kwargs)


def _plot_graph(G, vertex_color, vertex_size, highlight,
                edges, edge_color, edge_width,
                indices, colorbar, limits, ax, title):
    r"""Plot a graph with signals as color or vertex size.

    Parameters
    ----------
    vertex_color : array-like or color
        Signal to plot as vertex color (length is the number of vertices).
        If None, vertex color is set to `graph.plotting['vertex_color']`.
        Alternatively, a color can be set in any format accepted by matplotlib.
        Each vertex color can by specified by an RGB(A) array of dimension
        `n_vertices` x 3 (or 4).
    vertex_size : array-like or int
        Signal to plot as vertex size (length is the number of vertices).
        Vertex size ranges from 0.5 to 2 times `graph.plotting['vertex_size']`.
        If None, vertex size is set to `graph.plotting['vertex_size']`.
        Alternatively, a size can be passed as an integer.
    highlight : iterable
        List of indices of vertices to be highlighted.
        Useful for example to show where a filter was localized.
    edges : bool
        Whether to draw edges in addition to vertices.
        Default to True if less than 10,000 edges to draw.
        Note that drawing many edges can be slow.
    edge_color : array-like or color
        Signal to plot as edge color (length is the number of edges).
        Edge color is given by `graph.plotting['edge_color']` and transparency
        ranges from 0.2 to 0.9.
        If None, edge color is set to `graph.plotting['edge_color']`.
        Alternatively, a color can be set in any format accepted by matplotlib.
        Each edge color can by specified by an RGB(A) array of dimension
        `n_edges` x 3 (or 4).
    edge_width : array-like or int
        Signal to plot as edge width (length is the number of edges).
        Edge width ranges from 0.5 to 2 times `graph.plotting['edge_width']`.
        If None, edge width is set to `graph.plotting['edge_width']`.
        Alternatively, a width can be passed as an integer.
    indices : bool
        Whether to print the node indices (in the adjacency / Laplacian matrix
        and signal vectors) on top of each node.
        Useful to locate a node of interest.
    colorbar : bool
        Whether to plot a colorbar indicating the signal's amplitude.
    limits : [vmin, vmax]
        Map colors from vmin to vmax.
        Defaults to signal minimum and maximum value.
    ax : :class:`matplotlib.axes.Axes`
        Axes where to draw the graph. Optional, created if not passed.
    title : str
        Title of the figure.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
    ax : :class:`matplotlib.axes.Axes`

    Notes
    -----
    The orientation of directed edges is not shown. If edges exist in both
    directions, they will be drawn on top of each other.
    """
    if not hasattr(G, 'coords') or G.coords is None:
        raise AttributeError('Graph has no coordinate set. '
                             'Please run G.set_coordinates() first.')
    check_2d_3d = (G.coords.ndim != 2) or (G.coords.shape[1] not in [2, 3])
    if G.coords.ndim != 1 and check_2d_3d:
        raise AttributeError('Coordinates should be in 1D, 2D or 3D space.')
    if G.coords.shape[0] != G.N:
        raise AttributeError('Graph needs G.N = {} coordinates.'.format(G.N))

    def check_shape(signal, name, length, many=False):
        if (signal.ndim == 0) or (signal.shape[0] != length):
            txt = '{}: signal should have length {}.'
            txt = txt.format(name, length)
            raise ValueError(txt)
        if (not many) and (signal.ndim != 1):
            txt = '{}: can plot only one signal (not {}).'
            txt = txt.format(name, signal.shape[1])
            raise ValueError(txt)

    def normalize(x):
        """Scale values in [0.25, 1]. Return 0.5 if constant."""
        ptp = x.ptp()
        if ptp == 0:
            return np.full(x.shape, 0.5)
        return 0.75 * (x - x.min()) / ptp + 0.25

    def is_color(color):
        mpl, _, _ = _import_plt()
        if mpl.colors.is_color_like(color):
            return True  # single color
        try:
            return all(map(mpl.colors.is_color_like, color))  # color list
        except TypeError:
            return False  # e.g., color is an int

    if vertex_color is None:
        limits = [0, 0]
        colorbar = False
        vertex_color = (G.plotting['vertex_color'],)
    elif is_color(vertex_color):
        limits = [0, 0]
        colorbar = False
    else:
        vertex_color = np.asarray(vertex_color).squeeze()
        check_shape(vertex_color, 'Vertex color', G.n_vertices,
                    many=(G.coords.ndim == 1))

    if vertex_size is None:
        vertex_size = G.plotting['vertex_size']
    elif not np.isscalar(vertex_size):
        vertex_size = np.asarray(vertex_size).squeeze()
        check_shape(vertex_size, 'Vertex size', G.n_vertices)
        vertex_size = G.plotting['vertex_size'] * 4 * normalize(vertex_size) ** 2

    if edges is None:
        edges = G.Ne < 10e3

    if edge_color is None:
        edge_color = (G.plotting['edge_color'],)
    elif not is_color(edge_color):
        edge_color = np.asarray(edge_color).squeeze()
        check_shape(edge_color, 'Edge color', G.n_edges)
        edge_color = 0.9 * normalize(edge_color)
        edge_color = [
            np.tile(G.plotting['edge_color'][:3], [len(edge_color), 1]),
            edge_color[:, np.newaxis],
        ]
        edge_color = np.concatenate(edge_color, axis=1)

    if edge_width is None:
        edge_width = G.plotting['edge_width']
    elif not np.isscalar(edge_width):
        edge_width = np.array(edge_width).squeeze()
        check_shape(edge_width, 'Edge width', G.n_edges)
        edge_width = G.plotting['edge_width'] * 2 * normalize(edge_width)

    if limits is None:
        limits = [1.05 * vertex_color.min(), 1.05 * vertex_color.max()]

    if title is None:
        title = G.__repr__(limit=4)

    return _plt_plot_graph(G, vertex_color=vertex_color, vertex_size=vertex_size, highlight=highlight, edges=edges,
                           indices=indices, colorbar=colorbar, edge_color=edge_color, edge_width=edge_width,
                           limits=limits, ax=ax, title=title)


@_plt_handle_figure
def _plt_plot_graph(G, vertex_color, vertex_size, highlight,
                    edges, edge_color, edge_width,
                    indices, colorbar, limits, ax):
    mpl, plt, mplot3d = _import_plt()

    if edges and (G.coords.ndim != 1):  # No edges for 1D plots.

        sources, targets, _ = G.get_edge_list()
        edges = [
            G.coords[sources],
            G.coords[targets],
        ]
        edges = np.stack(edges, axis=1)

        if G.coords.shape[1] == 2:
            LineCollection = mpl.collections.LineCollection
        elif G.coords.shape[1] == 3:
            LineCollection = mplot3d.art3d.Line3DCollection
        ax.add_collection(LineCollection(
            edges,
            linewidths=edge_width,
            colors=edge_color,
            linestyles=G.plotting['edge_style'],
            zorder=1,
        ))

    try:
        iter(highlight)
    except TypeError:
        highlight = [highlight]
    coords_hl = G.coords[highlight]

    if G.coords.ndim == 1:
        ax.plot(G.coords, vertex_color, alpha=0.5)
        ax.set_ylim(limits)
        for coord_hl in coords_hl:
            ax.axvline(x=coord_hl, color='C1', linewidth=2)

    else:
        sc = ax.scatter(*G.coords.T,
                        c=vertex_color, s=vertex_size,
                        marker='o', linewidths=0, alpha=0.5, zorder=2,
                        vmin=limits[0], vmax=limits[1])
        if np.isscalar(vertex_size):
            size_hl = vertex_size
        else:
            size_hl = vertex_size[highlight]
        ax.scatter(*coords_hl.T,
                   s=2 * size_hl, zorder=3,
                   marker='o', c='None', edgecolors='C1', linewidths=2)

        if G.coords.shape[1] == 3:
            try:
                ax.view_init(elev=G.plotting['elevation'],
                             azim=G.plotting['azimuth'])
                ax.dist = G.plotting['distance']
            except KeyError:
                pass

    if G.coords.ndim != 1 and colorbar:
        plt.colorbar(sc, ax=ax)

    if indices:
        for node in range(G.N):
            ax.text(*tuple(G.coords[node]),  # accomodate 2D and 3D
                    s=node,
                    color='white',
                    horizontalalignment='center',
                    verticalalignment='center')


def _get_coords(G, edge_list=False):
    sources, targets, _ = G.get_edge_list()

    if edge_list:
        return np.stack((sources, targets), axis=1)

    coords = [np.stack((G.coords[sources, d], G.coords[targets, d]), axis=0)
              for d in range(G.coords.shape[1])]

    if G.coords.shape[1] == 2:
        return coords

    elif G.coords.shape[1] == 3:
        return [coord.reshape(-1, order='F') for coord in coords]


class Graph:
    r"""Base graph class.

    Parameters
    ----------
    W : sparse matrix or ndarray
        The weight matrix which encodes the graph.
    coords : ndarray
        Vertices coordinates (default is None).

    Attributes
    ----------
    N : int
        the number of nodes / vertices in the graph.
    Ne : int
        the number of edges / links in the graph, i.e. connections between
        nodes.
    W : sparse matrix
        the weight matrix which contains the weights of the connections.
        It is represented as an N-by-N matrix of floats.
        :math:`W_{i,j} = 0` means that there is no direct connection from
        i to j.
    coords : ndarray
        vertices coordinates in 2D or 3D space. Used for plotting only. Default
        is None.
    plotting : dict
        plotting parameters.

    """

    def __init__(self, W, coords):
        if len(W.shape) != 2 or W.shape[0] != W.shape[1]:
            raise ValueError('W has incorrect shape {}'.format(W.shape))

        # CSR sparse matrices are the most efficient for matrix multiplication.
        # They are the sole sparse matrix type to support eliminate_zeros().
        if sparse.isspmatrix_csr(W):
            self.W = W
        else:
            self.W = sparse.csr_matrix(W)

        # Don't keep edges of 0 weight. Otherwise Ne will not correspond to the
        # real number of edges. Problematic when e.g. plotting.
        self.W.eliminate_zeros()

        self.n_vertices = W.shape[0]

        diagonal = np.count_nonzero(self.W.diagonal())
        off_diagonal = self.W.nnz - diagonal
        self.n_edges = off_diagonal // 2 + diagonal

        self.coords = coords

        self.plotting = {'vertex_size': 100,
                         'vertex_color': (0.12, 0.47, 0.71, 0.5),
                         'edge_color': (0.5, 0.5, 0.5, 0.5),
                         'edge_width': 2,
                         'edge_style': '-'}

        # TODO: kept for backward compatibility.
        self.Ne = self.n_edges
        self.N = self.n_vertices

    def get_edge_list(self):
        r"""Return an edge list, an alternative representation of the graph.

        Each edge :math:`e_k = (v_i, v_j) \in \mathcal{E}` from :math:`v_i` to
        :math:`v_j` is associated with the weight :math:`W[i, j]`. For each
        edge :math:`e_k`, the method returns :math:`(i, j, W[i, j])` as
        `(sources[k], targets[k], weights[k])`, with :math:`i \in [0,
        |\mathcal{V}|-1], j \in [0, |\mathcal{V}|-1], k \in [0,
        |\mathcal{E}|-1]`.

        Returns
        -------
        sources : vector of int
            Source node indices.
        targets : vector of int
            Target node indices.
        weights : vector of float
            Edge weights.

        Notes
        -----
        The weighted adjacency matrix is the canonical form used in this
        package to represent a graph as it is the easiest to work with when
        considering spectral methods.

        Edge orientation (i.e., which node is the source or the target) is
        arbitrary for undirected graphs.
        The implementation uses the upper triangular part of the adjacency
        matrix, hence :math:`i \leq j \ \forall k`.

        """
        W = sparse.triu(self.W, format='coo')

        sources = W.row
        targets = W.col
        weights = W.data

        assert self.n_edges == sources.size == targets.size == weights.size
        return sources, targets, weights

    def plot(self, vertex_color=None, vertex_size=None, highlight=[],
             edges=None, edge_color=None, edge_width=None,
             indices=False, colorbar=True, limits=None, ax=None,
             title=None):
        r"""Docstring overloaded at import time."""
        return _plot_graph(self, vertex_color=vertex_color,
                           vertex_size=vertex_size, highlight=highlight,
                           edges=edges, indices=indices, colorbar=colorbar,
                           edge_color=edge_color, edge_width=edge_width,
                           limits=limits, ax=ax, title=title)


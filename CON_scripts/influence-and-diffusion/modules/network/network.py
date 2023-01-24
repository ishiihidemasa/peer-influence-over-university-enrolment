from abc import ABC, abstractmethod

import matplotlib as mpl
import networkx as nx
import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
from scipy import sparse


class network(ABC):
    """
    Abstract Base class for various networks.

    Attributes
    ----------
    _G : nx.Graph
    _N : int
        The number of nodes in `G`.
    _pos : dict
        Positions of nodes for plotting `G`.
    _nodelist : ArrayLike
        The list of nodes. Nodes are ordered along `nodelist` in the adjacency matrix.
    _seed : int
        Seed for the random number generator.
    """

    def __init__(self, seed: int):
        self._seed = seed
        self.initialize()

    def initialize(self) -> None:
        self.del_attr()
        self._G = self.generate()
        self._N = self._G.number_of_nodes()
        self._k = sum(self.get_degree_dist()) / self._N

    @abstractmethod
    def generate(self) -> nx.Graph:
        """
        generates a network.

        Returns
        -------
        G : nx.Graph
            A network.
        """
        pass

    def del_attr(self) -> None:
        """
        Deletes attributes when (re)generating `self._G`.
        """
        attr2del = ["_pos", "_N", "_k"]
        target = [
            getattr(self, attr) for attr in attr2del if hasattr(self, attr)
        ]
        for attr in target:
            del attr

    ##########
    # getters
    ##########
    @property
    def num_nodes(self) -> int:
        return self._N

    @property
    def mean_degree(self) -> float:
        return self._k

    #############################
    # Get information on network
    #############################
    @abstractmethod
    def get_nodelist(self) -> np.ndarray:
        """
        Returns
        -------
        nodelist : np.array
            Its shape is (self.N, 2).
            The first and second columns list node names and degrees, respectively.
        """
        pass

    def unsorted_nodelist(self) -> np.ndarray:
        """
        Returns a list of nodes that is not sorted.

        Returns
        -------
        nodelist : np.array
            Its shape is (self.N, 2).
            The first and second columns list node names and degrees, respectively.
        """
        nodelist = np.array(self._G.degree())
        return nodelist

    def sort_by_degree(self) -> np.ndarray:
        """
        Returns a list of nodes that is sorted according to their degrees.

        Returns
        -------
        nodelist : np.array
            Its shape is (self.N, 2).
            The first and second columns list node names and degrees, respectively.
        """
        # sort nodes according to their degrees
        degrees = np.array(self._G.degree())[:, 1]  # degree array
        idx = np.argsort(degrees)[::-1]  # sorted indices
        nodelist = np.array(self._G.degree())[idx]  # sorted array
        return nodelist

    def group_by_degree(self) -> dict:
        nodelist = self.unsorted_nodelist()  # shape: (2, N)
        deg, indices = np.unique(nodelist[:, 1], return_inverse=True)
        # create a dictionary {degree: [node names]}
        d_grouped = {}
        for i in range(deg.size):
            d_grouped[deg[i]] = nodelist[:, 0][indices == i].tolist()
        return d_grouped

    def get_degree_dist(self) -> np.ndarray:
        """
        Returns an sorted array whose each element is the degree of a node.
        """
        degrees = sorted((d for n, d in self._G.degree()), reverse=True)
        return degrees

    def get_dist_from_source(
        self, sourceid: int, cutoff: int = None
    ) -> np.ndarray:
        """
        Returns
        -------
        ndarray
            An array whose first and second columns represent node names
            and shortest path lengths from the source to all nodes, respectively.
            It is sorted by node degrees in ascending order.
            Its shape is (Number of nodes, 2).

        Note
        ----
            It is assumed that self._G is connected.
        """
        d_dist = nx.single_source_shortest_path_length(
            self._G, sourceid, cutoff
        )
        nodenames = self.get_nodelist()[:, 0]  # sorted array of node names
        a_dist = np.array([[n, d_dist[n]] for n in nodenames])
        return a_dist

    def group_by_dist(
        self, sourceids: int | np.ndarray, cutoff: int = None
    ) -> dict:
        if type(sourceids) == np.ndarray:
            if sourceids.size == 1:
                source = sourceids[0]
            else:
                raise NotImplementedError(
                    "ndarray for sourceids must have the size of 1"
                )
        else:
            source = sourceids

        a_dist = self.get_dist_from_source(source, cutoff)
        dist, indices = np.unique(a_dist[:, 1], return_inverse=True)
        # create a dictionary {distance: [node names]}
        d_grouped = {}
        for i in range(dist.shape[0]):
            d_grouped[dist[i]] = a_dist[indices == i, 0].tolist()
        return d_grouped

    def info_by_degree(self, min_num:int = 5) -> tuple:
        """
        Return the number of nodes and the mean of average neighbour degree
        by node degree.

        Parameters
        ----------
        min_num : int, optional
            If the number of nodes with degree `k` is less than `min_num`,
            data for `k` is not plotted.

        Returns
        -------
        data_k : list
        data_num : list
        data_knn : list
        """
        d_nodebydeg: dict = self.group_by_degree()
        d_knn: dict = nx.average_neighbor_degree(self._G)
        data_k: list = []  # degree
        data_knn: list = []  # mean of average neighbour degree
        data_num: list = []  # number of nodes
        for k in d_nodebydeg.keys():
            if len(d_nodebydeg[k]) < min_num:
                continue
            data_k.append(k)
            l_knn = [d_knn[i] for i in d_nodebydeg[k]]
            data_knn.append(np.mean(l_knn))
            data_num.append(len(d_nodebydeg[k]))
        
        return (data_k, data_num, data_knn)

    #############################
    # Calculate Laplacian Matrix
    #############################
    def get_sparse_adjacency(self) -> sparse.csr_array:
        # sort nodes according to their degrees
        nodenames = self.get_nodelist()[:, 0]  # sorted array of node ids
        # NOTE: No need to wrap with csr_array in Networkx 3.0
        return sparse.csr_array(
            nx.adjacency_matrix(self._G, nodelist=nodenames)
        )

    def get_laplacian(self) -> sparse.csr_array:
        A = self.get_sparse_adjacency()  # adjacency matrix: sorted
        # NOTE: No need to wrap with csr_array in Networkx 3.0
        D = sparse.csr_array(sparse.diags(A.sum(axis=0)))  # (in-)degree matrix
        L = D - A  # Laplacian matrix of the network
        return L

    #####################
    # Visualizaion tools
    #####################
    def get_pos(self) -> dict:
        """
        Returns
        -------
        dict
            A dictionary representing positions of nodes calculated with nx.spring_layout().
        """
        try:
            return self._pos
        except AttributeError:
            self.initialize_pos()
            return self._pos

    def initialize_pos(self) -> None:
        self._pos = nx.spring_layout(G=self._G, seed=self._seed)

    def display_graph(self, node_size: float = 20) -> mpl.figure.Figure:
        """
        Display the network wi"th Fruchterman-Reingold force-directed algorithm.
        """
        fig, ax = plt.subplots()
        ax.set_title(f"$N = {self._N}$")
        pos = self.get_pos()
        nodes = nx.draw_networkx_nodes(
            self._G,
            pos,
            ax=ax,
            node_size=node_size,
            node_color="tab:cyan",
            linewidths=0.5,
            edgecolors="k",
        )
        edges = nx.draw_networkx_edges(
            self._G, pos, ax=ax, width=0.5, edge_color="gray"
        )
        plt.show()
        return fig

    def plot_colored_graph(
        self,
        ax: mpl.axes.Axes,
        t: float,
        ut: np.ndarray,
        vmin: float,
        vmax: float,
        cmapname: str,
        node_size: float,
    ) -> tuple:
        """
        Parameters
        ----------
        ax : Axes
        t : float
        ut : ndarray
        vmin : float
        vmax : float
        cmapname : str
        """
        ax.set_title(rf"$t \approx {t:.3f}$")
        pos = self.get_pos()
        nodes = nx.draw_networkx_nodes(
            self._G,
            pos,
            ax=ax,
            node_size=node_size,
            linewidths=0.5,
            edgecolors="k",
            node_color=ut,
            cmap=colormaps[cmapname],
            vmin=vmin,
            vmax=vmax,
        )
        edges = nx.draw_networkx_edges(
            self._G, pos, ax=ax, width=0.5, edge_color="gray"
        )
        return (nodes, edges)

    def plot_deg_corr(
        self,
        min_num: int = 10,
        deglogscale: bool = False,
        figw: float = 3.5,
        figh: float = 2.5,
    ) -> plt.Figure:
        """
        NOT USED IN THE THESIS.

        Parameters
        ----------
        min_num : int, optional
            If the number of nodes with degree `k` is less than `min_num`,
            data for `k` is not plotted.
        """
        data_k, data_num, data_knn = self.info_by_degree(min_num)

        fig, ax = plt.subplots(
            figsize=(figw, figh), dpi=300, layout="constrained"
        )
        ax.plot(
            data_k,
            data_knn,
            c="k",
            marker="o",
            markersize=5,
            mew=0.5,
            fillstyle="none",
            ls="none",
        )
        ax.set_xlabel("Degree $k$")
        ax.set_ylabel(r"$\langle k_{\mathrm{nn}, i} \rangle (k)$")

        ax2 = ax.twinx()
        ax2.bar(x=data_k, height=data_num, width=0.6, color="darkgray")
        ax2.set_ylabel("Number of nodes")
        if deglogscale:
            ax2.set_yscale("log")
        ax.set_zorder(ax2.get_zorder() + 1)  # put ax in front of ax2
        ax.patch.set_visible(False)  # hide the 'canvas'

        fig.text(
            x=0.5,
            y=1,
            s=rf"$N = {self.num_nodes}$, $\langle k_i \rangle$ = {self.mean_degree:.2f}",
            ha="center",
            va="bottom",
            fontsize="large",
        )

        plt.show()
        return fig

    def plot_num_k_neighbour(
        self, sourceid: int = 0, figw: float = 3.5, figh: float = 2.5
    ) -> plt.Figure:
        """
        NOT USED IN THE THESIS.

        Parameters
        ----------
        sourceid : int
            The index of the focal node in `nodelist` (not the node name in `self._G`).
        figw : float
        figh : float
        """
        nodename, k = self.get_nodelist()[sourceid]
        (figw, figh) = (3, 2)

        d_nodebydist = self.group_by_dist(sourceids=nodename)

        fig, ax = plt.subplots(
            figsize=(figw, figh), dpi=300, layout="constrained"
        )
        ax.bar(
            x=list(d_nodebydist.keys()),
            height=[len(l) for l in d_nodebydist.values()],
            width=0.6,
            color="darkgray",
        )

        ax.text(
            x=0.02,
            y=0.98,
            s=f"$k = {k}$",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )
        ax.set_xlabel("Distance")
        ax.set_ylabel("Number of nodes")

        plt.show()

        return fig

import networkx as nx
import numpy as np

from .network import network


class erdos_renyi(network):
    """
    Subclass for an Erdos-Renyi graph.
    """

    def __init__(self, N: int, k: float, seed: int = 2022) -> None:
        """
        Parameters
        ----------
        N : int
            The number of nodes.
            Resulting network may have fewer nodes than `N`,
            since nodes with no edge are removed.
        k : float
            Target mean degree of the network.
        seed : int, optional
            Seed for random number generator.
        """
        self._N = N
        self._k = k
        super().__init__(seed)  # network is generated

    # override
    def generate(self) -> nx.Graph:
        """
        Generates an Erdos-Renyi graph with `N` nodes and mean degree of `k`.
        Nodes having no edges are removed.
        """
        p = self._k / self._N  # when N is large, k = pN approximately holds.
        G = nx.erdos_renyi_graph(self._N, p, seed=self._seed, directed=False)
        # G = nx.fast_gnp_random_graph(N, p, seed=seed, directed=False) # faster when p is small
        G.remove_nodes_from(
            list(nx.isolates(G))
        )  # remove nodes with zero degree
        return G

    # override
    def get_nodelist(self) -> np.ndarray:
        # Use the list sorted by degree
        return self.sort_by_degree()

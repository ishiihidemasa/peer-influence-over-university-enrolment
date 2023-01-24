import networkx as nx
import numpy as np

from .network import network


class barabasi_albert(network):
    """
    Subclass for a barabsi-albert model.
    """

    def __init__(self, N: int, m: int, seed: int = 2023) -> None:
        """
        Parameters
        ----------
        N : int
        m : int
        seed : int, optional
            Seed for random number generator.
        """
        self._N = N
        self._m = m
        super().__init__(seed)  # network is generated

    # override
    def generate(self) -> nx.Graph:
        """
        Generates an undirected random graph with a given degree distribution.
        """
        G = nx.barabasi_albert_graph(n=self._N, m=self._m, seed=self._seed)
        if not nx.is_connected(G):
            raise NotImplementedError("Resulting graph is not connected!")
        return G

    # override
    def get_nodelist(self) -> np.ndarray:
        # Use the list sorted by degree
        return self.sort_by_degree()

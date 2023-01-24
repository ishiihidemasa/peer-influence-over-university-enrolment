import networkx as nx
import numpy as np

from .network import network


class config(network):
    """
    Subclass for a configuration model.
    """

    def __init__(self, deg_sequence: np.ndarray, seed: int = 2023) -> None:
        """
        Parameters
        ----------
        deg_sequence : np.ndarray
            Parameter for `networkx.configuration_model()`.
        seed : int, optional
            Seed for random number generator.
        """
        self._deg_sequence: np.ndarray = deg_sequence
        super().__init__(seed)  # network is generated

    # override
    def generate(self) -> nx.Graph:
        """
        Generates an undirected random graph with a given degree distribution.
        """
        G = nx.configuration_model(self._deg_sequence, seed=self._seed)
        G = nx.Graph(G)  # remove parallel edges
        G.remove_edges_from(nx.selfloop_edges(G))  # remove self-loops
        # extract the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        return G

    # override
    def get_nodelist(self) -> np.ndarray:
        # Use the list sorted by degree
        return self.sort_by_degree()


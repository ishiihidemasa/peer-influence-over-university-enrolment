import networkx as nx
import numpy as np

from .network import network


class two_node(network):
    """
    Subclass for a graph used in two-node model's simulations.

    Note
    ----
        This network is not connected.
    """

    def __init__(self, num_edges: int, seed: int = 2022) -> None:
        self._num_edges = num_edges
        super().__init__(seed)  # network is generated

    # override
    def generate(self) -> nx.Graph:
        """
        Generate a network with `num_edges` pairs of nodes.
        Each pair has one edge, and no edge exists between pairs.
        """
        x = 2 * np.arange(self._num_edges)
        y = x + 1
        edgeiter = zip(x, y)

        G = nx.Graph()
        G.add_edges_from(edgeiter)

        return G

    # override
    def get_nodelist(self) -> np.ndarray:
        # Use the unsorted node list
        return self.unsorted_nodelist()

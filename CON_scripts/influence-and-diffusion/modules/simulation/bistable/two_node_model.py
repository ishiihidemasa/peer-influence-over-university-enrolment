from math import floor

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from modules.network import network
from modules.simulation.bistable import bistable


class two_node_model(bistable):
    """
    A subclass for conducting simulations on two-box models.

    Attributes
    ----------
    num_pairs : int
    """

    def __init__(
        self,
        nw: network,  # simulation
        t_max: float,  # simulation
        r: float,  # dw_diffusive
        D: float,  # dw_diffusive
        alpha: float = 0.05,  # dw_diffusive
        dt: float = 5 * 10 ** (-3),  # dw_diffusive
        max_sample: int = 10001,  # simulation
        seed: int = 634,  # simulation
    ) -> None:
        super().__init__(
            nw=nw,
            t_max=t_max,
            r=r,
            D=D,
            alpha=alpha,
            dt=dt,
            max_sample=max_sample,
            seed=seed,
        )
        self.num_pairs: int = int(self.nw.num_nodes / 2)

    #######################
    # Initialization tools
    #######################
    def init_activate_y(self) -> None:
        activated = 2 * np.arange(self.nw.num_nodes / 2, dtype="int") + 1
        # initial condition
        init = np.zeros(shape=self.nw.num_nodes)
        init[activated] = 1  # activate node 0
        self.activated, self.init = (activated, init)

    ####################
    # Calculation tools
    ####################
    def _within_square(
        self, xrange: tuple[float], yrange: tuple[float]
    ) -> np.ndarray:
        """
        Returns an ndarray whose ij element is True
        if system (pair) i is whithin (`xrange` x `yrange`) at time j.

        Parameters
        ----------
        xrange : tuple[float]
        yrange : tuple[float]

        Returns
        -------
        within_square : (Number of pairs, Number of time points) np.ndarray
            An array with the shape (Number of pairs, Number of time points).
        """
        x = self._res_y[0::2]
        y = self._res_y[1::2]
        x_within = (xrange[0] < x) & (x < xrange[1])
        y_within = (yrange[0] < y) & (y < yrange[1])
        within_square = x_within & y_within
        return within_square

    def escape_times(
        self, xrange: tuple[float], yrange: tuple[float]
    ) -> np.ndarray:
        """
        Retruns an ndarray whose i-th element shows
        when system (pair) i first entered the region (`xrange` x `yrange`).

        Parameters
        ----------
        xrange : tuple[float]
        yrange : tuple[float]

        Returns
        -------
        escape_times : (Number of pairs) np.ndarray
            An array with the shape (Number of pairs).
        """
        within_square = self._within_square(xrange, yrange)
        escape_times = np.array(
            [
                min(self._res_t[within_square[i]], default=2 * self.t_max)
                for i in range(self.num_pairs)
            ]
        )
        return escape_times

    def potential(
        self, x: float | np.ndarray, y: float | np.ndarray
    ) -> float | np.ndarray:
        schloegl = (
            lambda u: u**4 / 4
            - (1 + self._r) / 3 * u**3
            + self._r * u * u / 2
        )
        v = schloegl(x) + schloegl(y) + self.D / 2 * (x - y) * (x - y)
        return v

    ######################
    # Visualization tools
    ######################
    def plot_potential(
        self,
        ax: mpl.axes.Axes,
        cmap: str = "bone",
        levels: int = 51,
        fixlevels: bool = True,
        face: bool = False,
        line: bool = True,
    ):
        arr = np.linspace(self.ymin, self.ymax, 101)
        X, Y = np.meshgrid(arr, arr, indexing="ij")

        if fixlevels:
            v_ss = self.potential(self._r, self._r)
            v_aa = self.potential(1, 1)
            points = (np.linspace(0, 1, levels)) ** (
                1 / 2
            )  # sample points within [0, 1]
            points /= points[int(levels * 0.75)]  # rescale
            levels = (
                points * (v_ss - v_aa) + v_aa
            )  # lines are drawn at V == levels

        if face:
            contf = ax.contourf(
                X,
                Y,
                self.potential(X, Y),
                levels=levels,
                cmap=cmap,
                zorder=1,
            )

        if line:
            cont = ax.contour(
                X,
                Y,
                self.potential(X, Y),
                levels=levels,
                cmap=cmap,
                vmax=levels[-1] + 0.15 * (levels[-1] - levels[0]),
                linewidths=1,
                zorder=2,
            )

        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        return

    def plot_scatter(
        self,
        plot_pot: bool = True,
        pairs_to_plot: list[int] | int = 3,
        figw: float = 5,
        figh: float = 4,
        alpha: float = 0.5,
        **potkwargs,
    ) -> plt.Figure:
        """
        Parameters
        ----------
        plot_pot : bool, optional
            If `True`, show contour of potential.
        pairs_to_plot : list[int] | int, optional
            If this is an integer, `pairs_to_plot` systems (pairs) are selected and their time evolutions are shown.
            This may also be a list of indeces of systems (pairs) to plot.
        figw : float, optional
        figh : float, optional
        alpha : float, optional
            Transparancy of markers for sample paths.
        potkwargs : optional
            Passed to `plot_potential()`.
        """
        fig, ax = plt.subplots(
            figsize=(figw, figh), dpi=300, layout="constrained"
        )
        ax.set_title(f"$D = {self._D:.3f}$")

        if type(pairs_to_plot) == int:
            # if pairs_to_plot is an integer
            step: int = max(floor(self.num_pairs / pairs_to_plot), 1)
            pairs_to_plot = np.arange(int(self.nw.num_nodes / 2))[
                ::step
            ].tolist()

        if plot_pot:
            self.plot_potential(ax, **potkwargs)

        for pair in pairs_to_plot:
            s = ax.scatter(
                x=self._res_y[2 * pair],
                y=self._res_y[2 * pair + 1],
                c=self._res_t,
                cmap="plasma",
                s=0.8,
                linewidth=0,
                alpha=alpha,
                zorder=9,
            )
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim((self.ymin, self.ymax))
        ax.set_ylim((self.ymin, self.ymax))
        fig.colorbar(s, label="time")
        return fig

from abc import ABC, abstractmethod
from math import ceil
from typing import Callable

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from modules.network import network
from scipy.integrate import solve_ivp

# mpl.rcParams.update(mpl.rcParamsDefault)
# plt.rcParams["mathtext.fontset"] = "stix"


class simulation(ABC):
    """
    Abstract Base class for simulations for dynamical systems on network.

    Attributes
    ----------
    nw : network
    t_max : float
    max_sample : int
    seed : int
    nodelist : ndarray
    init : np.ndarray
    _res_t : ndarray
    _res_y : ndarray
    ymin : float
    ymax : float
    """

    def __init__(
        self,
        nw: network,
        t_max: float,
        ymin: float,
        ymax: float,
        max_sample: int = 10001,
        seed: int = 634,
    ) -> None:
        self.dict_plotmethod = {
            "deg-and-x": self.plot_deg_and_x,
            "colored-network": self.plot_colored_network,
        }
        # list of attributes to delete upon parameter changes
        self.attr2del = ["_res_t", "_res_y"]

        self.nw = nw
        self.t_max = t_max
        self.ymin = ymin
        self.ymax = ymax
        self.max_sample = max_sample
        self.seed = seed

        self.nodelist = self.nw.get_nodelist()
        # first column: node names
        # second column: degrees (ascending)

    ##################
    # setter / getter
    ##################
    def del_attr(self) -> None:
        """
        Deletes calculation results when changing parameters.
        """
        target = [
            getattr(self, attr)
            for attr in self.attr2del
            if hasattr(self, attr)
        ]
        for attr in target:
            del attr

    ##################################
    # main method to run a simulation
    ##################################
    def simulate(self, **kwargs: str) -> None:
        self._res_t, self._res_y = self._evolve(**kwargs)

    ########################
    # Numerical integration
    ########################
    @abstractmethod
    def _evolve(self, **kwargs: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        res_t : np.ndarray
            Calculation points (times).
        res_y : np.ndarray
            Calculated time series (state variables).
        """
        pass

    @staticmethod
    def _solve_ode(
        fun, t_max: float, init: np.ndarray, args: tuple
    ) -> tuple[np.ndarray, np.ndarray]:
        res = solve_ivp(
            fun=fun,
            t_span=(0, t_max),
            y0=init,
            args=args,
            atol=1e-8,
            rtol=1e-8,
        )
        return (res.t, res.y)

    @staticmethod
    def _euler_maruyama(
        drift: Callable,
        drift_args: tuple,
        diffusion: Callable,
        diff_args: tuple,
        t_max: float,
        init: np.ndarray,
        dt: float,
        max_sample: int,
        seed: int = None,
        rng: np.random.Generator = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        drift : Callable
            Drift coefficient in the SDE returning `float` or `np.ndarray`.
            Current time `t`, state variables `y_old` and other args are passed as `drift(t, y_old, *drift_args)`.
        drift_args : tuple
        diffusion : Callable[[float, np.ndarray], float | np.ndarray]
            Diffusion coefficient in the SDE returning `float` or `np.ndarray`.
            Current time `t`, state variables `y_old` and other args are passed as `diffusion(t, y_old, *diff_args)`.
        diff_args : tuple
        init : np.ndarray
            Initial condition.
        dt : float
            Time step.
        t_max : float
            End point of calculation.
        max_sample : int, optional
            The maximum number of samples in the returned arrays.
        seed : int, optional
        rng : np.random.Generator, optional
            When `rng` is given, the passed `rng` is used.
            `seed` is not used when `rng` is passed.

        Returns
        -------
        res_t : np.ndarray
            Calculation points (times).
        res_y : np.ndarray
            Calculated time series (state variables), mimicking the output from `scipy.integrate.solve_ivp()`.
        """
        # preparations
        interval: int = ceil(t_max / dt / (max_sample - 1))
        if rng is None:
            rng = np.random.default_rng(seed=seed)
        n = init.size

        t: float = 0
        count: int = 0
        y_old = init
        l_t = [t]
        l_y = [y_old]

        while t < t_max:
            count += 1
            xi = rng.normal(loc=0, scale=np.sqrt(dt), size=n)
            y_new = (
                y_old
                + drift(t, y_old, *drift_args) * dt
                + diffusion(t, y_old, *diff_args) * xi
            )
            t += dt
            y_old = y_new

            if count % interval == 0:
                l_t.append(t)
                l_y.append(y_new)

        # convert lists to ndarray
        a_t = np.array(l_t)
        a_y = np.array(
            l_y
        ).T  # transpose to match res.y returned by solve_ivp()
        return (a_t, a_y)

    ######################
    # Visualization tools
    ######################
    def simple_plot(self, l_focal: list[int]) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 4))
        for f in l_focal:
            ax.plot(
                self._res_t,
                self._res_y[f, :],
                label=f"#{self.nodelist[f, 0]} ({f})",
            )
        ax.set_xlabel("$t$", fontsize="large")
        ax.set_ylabel("$u$", fontsize="large")
        ax.legend()
        plt.show()
        return fig

    def get_tid_list(self, num_t: int, max_t_prop: float) -> list:
        if not 0 <= max_t_prop <= 1:
            raise ValueError("max_t_prop should be in range [0, 1].")
        max_tid = int((self._res_t.shape[0] - 1) * max_t_prop)

        # indices of res.t to plot
        l_tid = [int(tau) for tau in np.linspace(0, max_tid, num_t)]
        return l_tid

    def snapshot_fig(
        self,
        num_t: int,
        max_t_prop: float,
        plotmethod: str,
        **kwargs,
    ) -> plt.Figure:
        """
        Creates multiple axes of snapshots at several times.

        Parameters
        ----------
        num_t : int
        max_t_prop : float
        plotmethod : str
            One of the keys of `self.dict_plotmethod`.
        kwargs
            Parameters for plotting function.
        """
        plotfunc = self.dict_plotmethod[plotmethod]
        tid2plot = self.get_tid_list(num_t, max_t_prop)

        fig, axs = plt.subplots(
            ncols=int(num_t / 2),
            nrows=2,
            figsize=(3.5 * num_t / 2, 5),
            tight_layout=True,
        )
        for n, t_id in enumerate(tid2plot):
            _ = plotfunc(axs.flatten()[n], t_id, **kwargs)
        plt.show()
        return fig

    def plot_deg_and_x(
        self,
        ax: mpl.axes.Axes,
        t_id: int,
        ascending: bool = False,
    ) -> tuple[mpl.lines.Line2D]:
        """
        Parameters
        ----------
        ax : Axes
        t_id : int
        ascending : bool, optional
            If True, nodes are ordered by their degrees in ascending order in the plot.
        """
        t = self._res_t[t_id]

        xt = self._res_y[:, t_id]
        nodelist = self.nodelist.copy()

        if ascending:
            # increasing order in degrees
            nodelist = nodelist[::-1, :]
            xt = xt[::-1]

        # Display the time t
        ax.set_title(rf"$t \approx {t:.3f}$")

        ax.set_xlabel("Node $i$")

        # plot activity level x(t)
        ax.set_ylabel("$x_i(t)$", fontsize="large")
        ax.set_ylim((self.ymin, self.ymax))
        (ln_x,) = ax.plot(
            xt, ls="none", marker=".", markersize=1, c="blue", label="$x_i$"
        )

        # plot degrees with different scale
        ax_deg = ax.twinx()
        ax_deg.set_ylabel("degree $k_i$", fontsize="large")
        (ln_deg,) = ax_deg.plot(
            nodelist[:, 1],
            ls="none",
            marker=".",
            markersize=1,
            c="r",
            label="$k_i$",
        )
        # put ax in front of ax_deg
        ax.set_zorder(ax_deg.get_zorder() + 1)
        ax.patch.set_visible(False)  # hide the 'canvas'

        return (ln_x, ln_deg)

    def plot_colored_network(
        self,
        ax: mpl.axes.Axes,
        t_id: int,
        cmapname: str = "plasma",
        node_size: float = 15,
    ) -> tuple:
        """
        Parameters
        ----------
        ax : Axes
        t_id : int
        """
        t = self._res_t[t_id]
        ut = self._res_y[:, t_id]

        # sort u(t) according to node names
        # sort ut by node names
        idx = np.argsort(self.nodelist[:, 0])
        ut_byname = ut[idx]

        nodes, edges = self.nw.plot_colored_graph(
            ax,
            t,
            ut_byname,
            vmin=self.ymin,
            vmax=self.ymax,
            cmapname=cmapname,
            node_size=node_size,
        )
        fig = ax.figure
        fig.colorbar(nodes, ax=ax)
        return (nodes, edges)

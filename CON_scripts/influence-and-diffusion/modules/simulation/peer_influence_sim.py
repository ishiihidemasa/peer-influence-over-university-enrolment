from functools import reduce

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from modules.network import network
from scipy.sparse import csr_array

from .simulation import simulation


class peer_influence_sim(simulation):
    """
    Subclass for direct numerical simulations of a model for influence of peers over enrolment behaviour.

    Attributes
    ----------
    _D : float
    _descending_deg : bool
    _costs : np.ndarray
    _det_nodes : np.ndarray
        Indeces of determined students.
        Its elements do not represent node names of `nw._G`.
    """

    def __init__(
        self,
        nw: network,  # simulation
        t_max: float,  # simulation
        D: float,  # peer_influence_sim
        det_nodes: np.ndarray,  # peer_influence_sim
        costs: np.ndarray,  # peer_influence_sim
        descending_deg: bool,  # peer_influence_sim
        max_sample: int = 10001,  # simulation
        seed: int = 634,  # simulation
    ) -> None:
        """
        Parameters
        ----------
        nw: network
        t_max: float
        D: float
        det_nodes: np.ndarray
            The list of determined nodes in the original (mean field) simulation.
        costs: np.ndarray
            The list of costs in the original (mean field) simulation.
        max_sample: int, optional
        seed: int, optional
        """
        self._D = D
        self._descending_deg = descending_deg
        super().__init__(
            nw=nw,
            t_max=t_max,
            ymin=-0.05,
            ymax=1.05,
            max_sample=max_sample,
            seed=seed,
        )

        # drop some nodes to match the number of students
        rng = np.random.default_rng(seed=2023)
        N_original = costs.size
        remain = np.sort(
            rng.choice(
                np.arange(N_original, dtype="i8"),
                size=self.nw.num_nodes,
                replace=False,
            )
        )
        # nodes are always sorted in descending order by degree
        # thus, cost distribution need be flipped when degree is ascending
        self._costs = (
            costs[remain] if self._descending_deg else costs[remain][::-1]
        )
        self._det_nodes = reduce(
            np.intersect1d,
            (det_nodes, remain, np.arange(self.nw.num_nodes, dtype="i8")),
        )

    #############
    # Properties
    #############
    @property
    def D(self) -> float:
        return self._D

    @D.setter
    def D(self, newD: float) -> None:
        self.del_attr()  # delete existing results to avoid confusion
        self._D = newD

    @property
    def Nd_tot(self) -> int:
        return self._det_nodes.size

    ##########################
    # main part of simulation
    ##########################
    # override
    def _evolve(self) -> tuple[np.ndarray, np.ndarray]:
        init: np.ndarray = np.zeros(shape=self.nw.num_nodes, dtype="f8")
        init[self._det_nodes] = 1

        # laplacian matrix of the network
        L = self.nw.get_laplacian()

        return self._solve_ode(
            fun=self.peer_influence_model,
            t_max=self.t_max,
            init=init,
            args=(self._costs, self._D, L, self._det_nodes),
        )

    ##################
    # model equations
    ##################
    @staticmethod
    def peer_influence_model(
        t: float,
        v: np.ndarray,
        c: np.ndarray,
        D: float,
        L: csr_array,
        det_nodes: np.ndarray,
    ) -> np.ndarray:
        vdot = -c * v + D * -L.dot(v)
        vdot[det_nodes] = 0
        return vdot

    ######################
    # visualization tools
    ######################
    def plot_mean(self) -> plt.Figure:
        plotkwargs: dict = {
            "c": "k",
            "alpha": 0.5,
            "marker": "o",
            "markersize": 3,
            "fillstyle": "none",
            "mew": 0.5,
            "ls": "none",
        }
        fig, ax = plt.subplots(
            figsize=(3, 2),
            layout="constrained",
            dpi=300,
        )

        ax.plot(self._res_t, self._res_y.mean(axis=0), **plotkwargs)
        ax.set_xlabel("time $t$")
        ax.set_ylabel(r"$\langle x_i \rangle$")

        plt.show()
        return fig

    def plot_deg_cost_x(
        self,
        t_id: int,
        msindot: float = 8,
        costlogscale: bool = False,
        theory: bool = True,
        leg2_loc: str = "upper center",
        leg2_bbox_to_anchor: tuple = (0.5, 1),
        ylim2: tuple = (0, 1.4),
        **leg2kwargs,
    ) -> plt.Figure:
        # obtain data
        t = self._res_t[t_id]
        x = self._res_y[:, t_id]
        N = self.nw.num_nodes

        # adjust the order of nodes to match `peer_influence.py`
        # originally, degeres are in descending order
        # flipping is needed when cost is descending
        if self._descending_deg:
            step = 1
            det_nodes = self._det_nodes
        else:
            step = -1
            det_nodes = np.array(
                [N - 1 - i for i in self._det_nodes], dtype="i8"
            )

        # prepare texts
        suptitle: str = rf"$t \approx {t:.2f}$"

        footnote: str = (
            rf"$N = {N}$, $N_D = {self.Nd_tot}$, "
            + rf"$D = {self._D}$, $\langle c_i \rangle = {self._costs.mean():.3f}$,"
            + "\n"
            + rf"$\langle k_i \rangle = {self.nw.mean_degree:.3f}$"
        )

        # generate figure and axes
        fig, ax = plt.subplots(
            2,
            1,
            sharex=True,
            height_ratios=(2, 3),
            figsize=(5, 4.2),
            layout="constrained",
            dpi=300,
        )

        # font size
        plt.rcParams["axes.labelsize"] = "large"

        # marker settings
        markersize: float = (
            msindot * 72.0 / fig.dpi
        )  # [dot] * [point / inch] / [dot / inch]
        markerkwargs: dict = {
            "alpha": 0.5,
            "marker": "o",
            "markersize": markersize,
            "fillstyle": "none",
            "mew": 0.3 * markersize,
        }
        legscale: float = 5 / markersize  # marker in legends

        # Panel 1
        # cost
        (line_c,) = ax[0].plot(
            self._costs[::step],
            c="tab:orange",
            ls="none",
            **markerkwargs,
            label="Cost",
            zorder=3,
        )
        ax[0].set_ylabel("Cost $c_i$")
        if costlogscale:
            ax[0].set_yscale("log")

        # degrees
        degrees = self.nw.get_nodelist()[:, 1]
        ax_d = ax[0].twinx()
        (line_k,) = ax_d.plot(
            degrees[::step],
            c="tab:blue",
            ls="none",
            **markerkwargs,
            label="Degree",
            zorder=3,
        )
        ax_d.set_ylabel("Degree $k_i$")

        ax[0].set_zorder(ax_d.get_zorder() + 1)  # put ax in front of ax2
        ax[0].patch.set_visible(False)  # hide the 'canvas'

        # legend
        handles = [line_c, line_k]

        ax[0].legend(
            handles=handles,
            ncols=2,
            loc="upper center",
            markerscale=legscale,
            handlelength=1.2,
            handletextpad=0.3,
            framealpha=0.5,
            borderpad=0.3,
            borderaxespad=0.15,
        )

        # Panel 2
        # mean
        m = x.mean()
        ax[1].plot(
            (0, N - 1),
            (m, m),
            c="k",
            ls=(0, (5, 5)),  # loosely dashed
            lw=0.8,
            label=rf"Mean $({m:.3f})$",
            zorder=1,
        )
        # fixed points
        # - determined students
        ax[1].plot(
            det_nodes,
            x[::step][det_nodes],
            c="tab:blue",
            ls="none",
            **markerkwargs,
            label="Determined",
            zorder=3,
        )
        # - undecided students
        # create an array indicating if each student is undecided
        mask = np.ones(N, dtype=bool)
        mask[self._det_nodes] = False  # in original order

        ax[1].plot(
            np.arange(N)[mask[::step]],
            x[::step][mask[::step]],
            c="tab:orange",
            ls="none",
            **markerkwargs,
            label="Undecided",
            zorder=3,
        )
        # - theoretical prediction
        if theory:
            xstar, xall = self.calc_xstar(mask)
            ax[1].plot(
                np.arange(N)[mask[::step]],
                xstar[::step],
                c="tab:red",
                marker=".",
                markersize=markerkwargs["markersize"],
                mec="none",
                ls="none",
                zorder=1,
                label="$x_i^*$ (Theory)",
            )
            footnote += (
                rf", $\langle x_i^* \rangle = {xall.mean():.3f}$ (Theory)"
            )

        # axes settings
        ax[1].set_xlabel("Node $i$")
        ax[1].set_ylabel("$x_i$")
        ax[1].set_ylim(ylim2)
        ax[1].legend(
            handlelength=2,
            # handletextpad=0.3,
            markerscale=legscale,
            loc=leg2_loc,
            bbox_to_anchor=leg2_bbox_to_anchor,
            ncol=2,
            **leg2kwargs,
            framealpha=0.5,
            borderpad=0.3,
            borderaxespad=0.15,
        )

        ax[0].set_xmargin(0.02)

        fig.suptitle(suptitle, fontsize="x-large")
        fig.add_artist(
            mpl.lines.Line2D([0.05, 0.95], [0, 0], linewidth=0.5, color="k")
        )
        fig.text(
            x=0.5, y=-0.02, s=footnote, ha="center", va="top", fontsize="large"
        )

        plt.show()
        return fig

    def calc_xstar(self, mask: np.ndarray) -> tuple[np.ndarray]:
        ci: np.ndarray = self._costs[mask]
        degrees = self.nw.get_nodelist()[:, 1]
        ki: np.ndarray = degrees[mask]
        k_array: np.ndarray = np.arange(
            degrees.min(), degrees.max() + 1, dtype="i4"
        )

        # CALCULATE Nd = (N_{S, k})
        values, counts = np.unique(
            degrees[self._det_nodes], return_counts=True
        )
        # augment zero for degree values with no node
        Nd = np.zeros(shape=(degrees.max() - degrees.min() + 1), dtype="i4")
        Nd[np.isin(k_array, values)] = counts

        kappa_i: np.ndarray = self._D * ki
        mf = np.inner(k_array, Nd) / (
            degrees.sum() - (kappa_i * ki / (ci + kappa_i)).sum()
        )
        # print(mf)
        # fixed points for undecided students
        xstar: np.ndarray = kappa_i / (ci + kappa_i) * mf

        # augment 1 for determined students
        xall: np.ndarray = np.ones(self.nw.num_nodes)
        xall[mask] = xstar

        return (xstar, xall)

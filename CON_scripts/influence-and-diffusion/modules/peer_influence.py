# methods for calculating and visualizing mean fields
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


class peer_influence:
    """
    Attributes
    ----------
    N : int
    Nd_tot : int

    D : float
    _mf : float
        The value of the mean field.
    _xstar : np.ndarray
        Fixed points for undecided students.
    _xall : np.ndarray
        Fixed points for all students.

    det_method : str
    det_nodes : np.ndarray
    mask : np.ndarray
        `i`th element is `True` if student `i` is not determined.

    homogeneouscost : bool
    cost_dist : str
    costs : float | np.ndarray

    globalcoupling : bool
    deg_dist : str
    _descending_deg : bool
    k_array : np.ndarray
    degrees : np.ndarray
    Nd : np.ndarray
    """

    def __init__(
        self,
        N: int,
        Nd_tot: int,
        D: float,
        det_method: str,
        cmean: float,
        homogeneouscost: bool = False,
        globalcoupling: bool = False,
        **kwargs,
    ) -> None:
        """
        Paramteres
        ----------
        N : int
        Nd_tot : int
            The total number of determined students.
        D : float
            The diffusion constant.
        det_method : str
            Possible values are `"low"`, `"large"` and `"rand"`.
        cmean : float
            The mean of costs.
        homogeneouscost : bool, optional
            `True` when the homogeneity of costs is assumed.
            If `True`, the costs of all nodes are set to `cmean`.
        globalcoupling : bool, optional
            `True` when global coupling is assumed.

        cost_dist : str, optional
            It must be passed when `homogeneouscost` is `False`.
            Possible values are "uniform", "pareto" and "normal".

        deg_dist : str, optional
            It must be passed when `globalcoupling` is `False`.
            Possible values are "uniform" and "power".
        descending_deg : bool, optional
            It must be passed when `globalcoupling` is `False`.
            If `True`, degrees are set in descending order against node indeces.
        kmin : int, optional
            It must be passed when `globalcoupling` is `False`.
            Passed to `set_degrees()`.
        kmax : int, optional
            It must be passed when `globalcoupling` is `False`.
            Passed to `set_degrees()`.
        """
        # the number of students
        self.N = N
        self.Nd_tot = Nd_tot
        # diffusion constant
        self._D = D
        # determined students
        self.det_method = det_method
        self.pick_determined()
        # costs
        self.homogeneouscost = homogeneouscost
        if homogeneouscost:
            self.cost_dist = "homogeneous"
            self._costs = cmean
        else:
            self.cost_dist = kwargs["cost_dist"]
            self.set_costs(cmean)
        # network structure
        self.globalcoupling = globalcoupling
        if globalcoupling:
            self.deg_dist = "global"
        else:
            self.deg_dist = kwargs["deg_dist"]
            self._descending_deg = kwargs["descending_deg"]
            self.set_degrees(
                kwargs["kmin"],
                kwargs["kmax"],
            )

    def pick_determined(self, seed: int = 2023) -> None:
        rng = np.random.default_rng(seed=seed)

        if self.det_method == "low":
            # nodes with SMALLEST degrees
            det: np.ndarray = np.arange(self.Nd_tot)
        elif self.det_method == "large":
            # nodes with LARGEST degrees
            det: np.ndarray = np.arange(self.N - self.Nd_tot, self.N)
        elif self.det_method == "rand":
            # RANDOM nodes
            det: np.ndarray = rng.permutation(self.N)[: self.Nd_tot]

        self._det_nodes = det

        # create an array indicating if each student is undecided
        mask = np.ones(self.N, dtype=bool)
        mask[det] = False
        self.mask = mask

    def set_degrees(
        self,
        kmin: int,
        kmax: int,
        seed: int = 634,
    ) -> None:
        """
        Generate degree distribution.
        Sampling is repeated until a distribution such that
        the sum of degrees is even is realized.

        Parameters
        ----------
        kmin : int
            The lower limit of degree.
        kmax : int
            The upper limit of degree.
            The degrees are within `[kmin, kmax)`.
        seed : int, optional
        """
        rng = np.random.default_rng(seed=seed)

        # SET DEGREES
        # distribution-type specific part
        if self.deg_dist == "uniform":
            # uniform distribution
            density: np.ndarray = np.ones(shape=(kmax - kmin))
        elif self.deg_dist == "power":
            # power-law
            gamma: float = 1.5
            density: np.ndarray = np.power(
                np.arange(kmin, kmax, dtype="f8"), -gamma
            )
        else:
            raise ValueError("Unknown value passed for deg_dist")

        # general part
        density /= density.sum()  # normalize
        cdf = density.cumsum()  # Cumulative Distribution
        self.k_array: np.ndarray = np.arange(kmin, kmax, dtype="i8")

        while True:
            # in ascending order
            randnum: np.ndarray = np.sort(rng.uniform(size=self.N))
            # preparation
            degrees: np.ndarray = np.zeros(shape=(self.N), dtype="i8")

            for i, k in enumerate(self.k_array):
                if k == kmin:
                    degrees[randnum <= cdf[i]] = k
                elif k == kmax:
                    degrees[cdf[i - 1] < randnum] = k
                else:
                    degrees[(cdf[i - 1] < randnum) & (randnum <= cdf[i])] = k
            if degrees.sum() % 2 == 0:
                break

        if self._descending_deg:
            # in descending order
            degrees = degrees[::-1]
        self._degrees: np.ndarray = degrees

        # CALCULATE Nd = (N_{S, k})
        values, counts = np.unique(
            degrees[self._det_nodes], return_counts=True
        )
        # augment zero for degree values with no node
        Nd = np.zeros(shape=(kmax - kmin), dtype="i4")
        Nd[np.isin(self.k_array, values)] = counts
        self.Nd: np.ndarray = Nd

    def set_costs(self, cmean: float, seed: int = 1014) -> None:
        """
        Parameters
        ----------
        cmean : float
            The mean value of costs.
        seed : int, optional
        """
        rng = np.random.default_rng(seed=seed)

        # costs are in ascending order
        if self.cost_dist == "uniform":
            # uniform cost distribution
            cmin: float = 0.01 * cmean
            cmax: float = 1.99 * cmean
            c: np.ndarray = np.sort(rng.uniform(cmin, cmax, size=self.N))
        elif self.cost_dist == "pareto":
            # Pareto distribution
            shape: float = 3
            # mean = shape * scale / (shape - 1)
            scale: float = (1 - 1 / shape) * cmean
            c: np.ndarray = np.sort(
                (rng.pareto(a=shape, size=self.N) + 1) * scale
            )
        elif self.cost_dist == "normal":
            cmin: float = 0.01 * cmean
            mu: float = cmean
            sigma: float = 1 / 2 * cmean
            c: np.ndarray = np.sort(
                rng.normal(loc=mu, scale=sigma, size=self.N)
            )  # in ascending order
            c[c < cmin] = cmin  # replace values smaller than cmin with cmin

        self._costs = c

    def calc_mf_and_fp(self) -> None:
        ci: float | np.ndarray = (
            self._costs if self.homogeneouscost else self._costs[self.mask]
        )
        if self.globalcoupling:
            # Case 1
            kappa: float = self.N * self._D
            mf: float = self.Nd_tot / (self.N - (kappa / (ci + kappa)).sum())
            # fixed points for undecided students
            xstar: np.ndarray = kappa / (ci + kappa) * mf
        else:
            # Cases 2 and 3
            ki: np.ndarray = self._degrees[self.mask]
            kappa_i: np.ndarray = self._D * ki
            mf = np.inner(self.k_array, self.Nd) / (
                self._degrees.sum() - (kappa_i * ki / (ci + kappa_i)).sum()
            )
            # fixed points for undecided students
            xstar: np.ndarray = kappa_i / (ci + kappa_i) * mf
        # augment 1 for determined students
        xall: np.ndarray = np.ones(self.N)
        xall[self.mask] = xstar

        self._mf = mf
        self._xstar = xstar
        self._xall = xall

    def visualize_res(
        self,
        leg_loc: str = "upper center",
        bbox_to_anchor: tuple = (0.5, 1),
        msindot: tuple = 8,
    ) -> tuple[plt.Figure, str]:
        # setups
        d_dist_label = {
            "global": "Global coupling",
            "homogeneous": "Homogeneous",
            "uniform": "Uniform",
            "pareto": "Pareto",
            "normal": "Normal",
            "power": "Power-law",
        }

        # prepare texts
        mean_c: float = (
            self._costs if self.homogeneouscost else self._costs.mean()
        )

        suptitle: str = (
            "Cost: "
            + d_dist_label[self.cost_dist]
            + "; Degree: "
            + d_dist_label[self.deg_dist]
        )

        footnote: str = (
            rf"$N = {self.N}$, $N_D = {self.Nd_tot}$, "
            + rf"$D = {self._D}$, $\langle c_i \rangle = {mean_c:.3f}$,"
            + "\n"
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
            [self._costs] * self.N if self.homogeneouscost else self._costs,
            c="tab:orange",
            ls="none",
            **markerkwargs,
            label="Cost",
            zorder=3,
        )
        ax[0].set_ylabel("Cost $c_i$")
        if self.cost_dist == "pareto":
            ax[0].set_yscale("log")

        # degrees
        if not self.globalcoupling:
            # if global coupling is not assumed:
            mean_k = self._degrees.mean()
            footnote += rf"$\langle k_i \rangle = {mean_k:.3f}$, "

            ax_d = ax[0].twinx()
            (line_k,) = ax_d.plot(
                self._degrees,
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
        if self.globalcoupling:
            handles = [line_c]
        else:
            handles = [
                line_c,
                line_k,
            ]

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
        m = self._xall.mean()
        footnote += rf"(Mean field)$= {self._mf:.3f}$"
        ax[1].plot(
            (0, self.N - 1),
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
            self._det_nodes,
            np.ones(self._det_nodes.size),
            c="tab:blue",
            ls="none",
            **markerkwargs,
            label="Determined",
            zorder=3,
        )
        # - undecided students
        ax[1].plot(
            np.arange(self.N)[self.mask],
            self._xstar,
            c="tab:orange",
            ls="none",
            **markerkwargs,
            label="Undecided",
            zorder=3,
        )
        ax[1].set_xlabel("Node $i$")
        ax[1].set_ylabel("Fixed point $x_i^*$")
        ax[1].set_ylim((0, 1.35))
        ax[1].legend(
            ncols=2,
            handlelength=2,
            markerscale=legscale,
            loc=leg_loc,
            bbox_to_anchor=bbox_to_anchor,
            framealpha=0.5,
            borderpad=0.3,
            borderaxespad=0.15,
        )

        ax[0].set_xmargin(0.02)

        fig.suptitle(suptitle, fontsize="x-large")
        fig.add_artist(Line2D([0.05, 0.95], [0, 0], linewidth=0.5, color="k"))
        fig.text(
            x=0.5, y=-0.02, s=footnote, ha="center", va="top", fontsize="large"
        )

        plt.show()

        # return a text indicating distribution settings
        str_dist: str = self.cost_dist + "_"
        try:
            if self._descending_deg:
                str_dist += "de-"
        except AttributeError:
            pass
        str_dist += self.deg_dist

        return (fig, str_dist)

    ##########
    # getters
    ##########
    @property
    def D(self) -> float:
        return self._D

    @property
    def det_nodes(self) -> np.ndarray:
        return self._det_nodes

    @property
    def costs(self) -> np.ndarray:
        return self._costs

    @property
    def degrees(self) -> np.ndarray:
        return self._degrees

    @property
    def mf(self) -> float:
        try:
            return self._mf
        except AttributeError:
            self.calc_mf_and_fp()
            return self._mf

    @property
    def xstar(self) -> np.ndarray:
        try:
            return self._xstar
        except AttributeError:
            self.calc_mf_and_fp()
            return self._xstar

    @property
    def descending_deg(self) -> bool:
        return self._descending_deg

from datetime import datetime, timedelta, timezone

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from modules.network import network
from modules.simulation import simulation
from scipy.sparse import csr_array


class bistable(simulation):
    """
    Subclass for simulating coupled bistable reaction-diffusion systems.
    Schloegl model is formulated as:
    f(x) = -x (x - r) (x - 1).

    Attributes
    ----------
    _r : float
        The location of the maximum of the potential.
    _D : float | np.ndarray
        A value or an array of values of the diffusion coefficient.
    _alpha : float
        Noise strength in SDE.
    _threshold : float
        A node is said to be at the active state when `x >= threshold` holds.
        When the setter receives `None`, it is set to `(1 + self._r) / 2'.
    _prop : float
        Collective escape time is the time when `_prop` of all nodes reached the active state.
    dt : float
        Step size for Euler-Maruyama method.
    activated : np.ndarray
        Indices of activated nodes.
    init : np.ndarray
    init_method : str
    a_det_prop : np.ndarray
    """

    def __init__(
        self,
        nw: network,  # simulaion
        t_max: float,  # simulation
        r: float,  # bistable
        D: float | np.ndarray,  # bistable
        alpha: float,  # bistable
        threshold: float = None,  # bistable
        prop: float = 0.9,  # bistable
        dt: float = 5 * 10 ** (-3),  # bistable
        max_sample: int = 10001,  # simulation
        seed: int = 634,
    ) -> None:
        self._r: float = r
        self._D: float | np.ndarray = D
        self._alpha = alpha
        self.dt = dt

        super().__init__(
            nw=nw,
            t_max=t_max,
            ymin=-0.15,
            ymax=1.15,
            max_sample=max_sample,
            seed=seed,
        )

        # attributes to delete upon parameter changes
        self.attr2del + ["init_method", "a_det_prop"]
        # call setter
        self.threshold = threshold
        self.prop = prop
        # register a plotting function
        self.dict_plotmethod["mean-xt"] = self.plot_mean_xt

    ##################################
    # methods for running simulations
    ##################################
    # override
    def _evolve(self) -> tuple[np.ndarray, np.ndarray]:
        """
        The model SDE is numerically integrated with Euler-Maruyama scheme.
        """
        # check if self._D is a number (not an array)
        try:
            _ = iter(self._D)
            raise TypeError("D should be float.")
        except TypeError:
            pass

        # laplacian matrix of the network
        L = self.nw.get_laplacian()

        return self._euler_maruyama(
            drift=self.bistable_reaction_diffusion,
            drift_args=(self._r, self._D, L),
            diffusion=lambda t, v: self._alpha,
            diff_args=(),
            t_max=self.t_max,
            init=self.init,
            dt=self.dt,
            max_sample=self.max_sample,
            seed=self.seed,
        )


    def get_tau_f(self, res_t: np.ndarray, ts_prop: np.ndarray) -> float:
        """res_t[0] == 0 is assumed."""
        t_id = np.arange(res_t.size, dtype="i8")[ts_prop > 0][0]
        return res_t[t_id]

    def get_tau_s(self, res_t: np.ndarray, ts_prop: np.ndarray) -> float:
        """res_t[0] == 0 is assumed."""
        # get the last time when no node is active
        t_id: int = np.arange(res_t.size, dtype="i8")[ts_prop == 0][-1]
        # res_t[t_id + 1] results in error when t_id == res_t.size - 1
        # start time is next time step (or later)
        return res_t[t_id] + self.dt

    def get_tau_c(self, res_t: np.ndarray, ts_prop: np.ndarray) -> float:
        """res_t[0] == 0 is assumed."""
        # get index of first time when most nodes were activated
        t_id: int = np.arange(res_t.size, dtype="i8")[ts_prop >= self.prop][0]
        return res_t[t_id]

    def measure_escape_time(
        self,
        M: int,
        max_epoch: int,
        prefix: str = "",
    ) -> tuple[str]:
        """
        Conduct numerical integration `M` times for each value of `self._D`,
        and record the first escape time, the cascade-start time and the collective escape time.
        Values of measured escape time are exported as an csv file.

        Parameters
        ----------
        M : int
            The number of trials for each value of D.
        max_epoch : int
            Maximum number of epochs within each trial.
            Each epoch consists of one run of `self._euler_maruyama()`
            with `t_max=self.t_max`.
        prefix : str, optional

        Returns
        -------
        tuple[str]
            Each element represents a file name of a generated csv file.
        """
        # raise an error if self._D is not iterable
        try:
            _ = iter(self._D)
        except TypeError:
            raise TypeError("D should be numpy.ndarray.")

        # laplacian matrix of the network
        L = self.nw.get_laplacian()

        # always use this instance of generator
        # rng = np.random.default_rng(seed=self.seed)

        l_first = []  # first escape time
        l_start = []  # cascade-start time
        l_collective = []  # collective escape time

        for D_val in self._D:
            print(f"D = {D_val}:")

            # regenerate an instance of generator for each D
            rng = np.random.default_rng(seed=self.seed)

            l_first_d = []
            l_start_d = []
            l_collective_d = []

            for i in range(M):
                init: np.ndarray = np.copy(self.init)
                elapsed: float = 0  # collective escape time
                initial: bool = True  # True before the first escape
                tau_s: float = 0  # start time of cascades
                active: bool = False

                for epoch in range(max_epoch):
                    if D_val == 0:
                        res_t, res_y = self._euler_maruyama(
                            drift=self.schloegl,
                            drift_args=(self._r,),
                            diffusion=lambda t, v: self._alpha,
                            diff_args=(),
                            t_max=self.t_max,
                            init=init,
                            dt=self.dt,
                            max_sample=self.max_sample,
                            rng=rng,
                        )
                    else:
                        res_t, res_y = self._euler_maruyama(
                            drift=self.bistable_reaction_diffusion,
                            drift_args=(self._r, D_val, L),
                            diffusion=lambda t, v: self._alpha,
                            diff_args=(),
                            t_max=self.t_max,
                            init=init,
                            dt=self.dt,
                            max_sample=self.max_sample,
                            rng=rng,
                        )

                    # get time series of proportion
                    ts_prop = (res_y >= self.threshold).mean(axis=0)

                    # record first escape and start time
                    if (ts_prop == 0).any():
                        # no-active-node state was realized
                        if (ts_prop > 0).any():
                            # escape also occurred
                            if initial:
                                # this is the first escape!
                                tau_f: float = elapsed + self.get_tau_f(
                                    res_t, ts_prop
                                )
                                # record the time
                                l_first_d.append(tau_f)
                                # change flag
                                initial = False
                            # get the last time when no node is active
                            tau_s = elapsed + self.get_tau_s(res_t, ts_prop)

                    # if most nodes reached active state, stop loop.
                    if (ts_prop >= self.prop).any():
                        active = True
                        break

                    # update initial condition
                    init = res_y[:, -1]
                    elapsed += self.t_max

                # when one calculation is over
                if active:
                    # find when most nodes reached active state
                    elapsed += self.get_tau_c(res_t, ts_prop)
                else:
                    print(f"Active state not reached: D = {D_val:.3f}")

                # store the result from this sample path
                l_start_d.append(tau_s)
                l_collective_d.append(elapsed)

            # when M calculations are over
            # record results for this value of D
            l_first.append(l_first_d)
            l_start.append(l_start_d)
            l_collective.append(l_collective_d)

        a_first = np.array(l_first, dtype="f8")
        a_start = np.array(l_start, dtype="f8")
        a_coll = np.array(l_collective, dtype="f8")

        # save results in a csv file
        # record D values in the header
        header: str = (
            prefix
            + self.init_method
            + "; D values:,"
            + ",".join([str(d) for d in self._D])
        )
        # date and time in JST
        now = datetime.now(timezone(timedelta(hours=9)))
        basefname: str = (
            now.strftime("%y%m%d-%H%M%S_") + prefix + self.init_method + "_"
        )
        fname_f: str = basefname + "first.csv"
        fname_s: str = basefname + "start.csv"
        fname_c: str = basefname + "collective.csv"
        np.savetxt(fname_f, a_first, delimiter=",", header=header)
        np.savetxt(fname_s, a_start, delimiter=",", header=header)
        np.savetxt(
            fname_c,
            a_coll,
            delimiter=",",
            header=header,
        )

        return (fname_f, fname_s, fname_c)

    def solve_ode(
        self,
        t_epoch: float,
        max_epoch: int,
        show: bool = False,
        TINY: float = 1e-6,
    ) -> None:
        """ NOT USED IN THE THESIS. """

        # laplacian matrix of the network
        L = self.nw.get_laplacian()

        l_prop = []
        l_stat = []

        for D_val in self._D:
            print(f"D = {D_val}:")
            l_stat.append(False)
            init: np.ndarray = np.copy(self.init)
            count: int = 0

            for epoch in range(max_epoch):
                if D_val == 0:
                    res_t, res_y = self._solve_ode(
                        fun=self.schloegl,
                        t_max=t_epoch,
                        init=init,
                        args=(self._r,),
                    )
                else:
                    res_t, res_y = self._solve_ode(
                        fun=self.bistable_reaction_diffusion,
                        t_max=t_epoch,
                        init=init,
                        args=(
                            self._r,
                            D_val,
                            L,
                        ),
                    )
                if epoch > 0:
                    if ((res_y[:, -1] - init) ** 2).mean() <= TINY:
                        # change during last epoch was small
                        l_stat[-1] = True
                        break

                count += 1
                init = res_y[:, -1]

            if show:
                print(f"After {t_epoch * count} time:")
                self._res_t, self._res_y = (res_t, res_y)
                _ = self.snapshot_fig(
                    num_t=6, max_t_prop=1, plotmethod="deg-and-x"
                )
            l_prop.append((res_y[:, -1] >= self.threshold).mean())

        a_prop = np.array(l_prop, dtype="f8")
        self.a_det_prop = a_prop
        print(
            "Stationary state not reached when: D = ",
            self._D[~np.array(l_stat)],
        )
        return

    ##################
    # model equations
    ##################
    @staticmethod
    def bistable_reaction_diffusion(
        t: float,
        v: np.ndarray,
        r: float,
        D: float,
        L: csr_array,
    ) -> np.ndarray:
        schloegl = -v * (v - r) * (v - 1)
        coupling = D * -L.dot(v)
        return schloegl + coupling

    @staticmethod
    def schloegl(t: float, v: np.ndarray, r: float) -> np.ndarray:
        schloegl = -v * (v - r) * (v - 1)
        return schloegl

    ##############################
    # Generate initial conditions
    ##############################
    # Methods other than init_no_activation() are not used in the thesis.
    def init_no_activation(self) -> None:
        self.del_attr()
        self.init_method = "none"
        activated = np.array([])  # no node is activated
        init = np.zeros(shape=self.nw.num_nodes)
        self.activated, self.init = (activated, init)

    def init_one_mindeg(self) -> None:
        self.del_attr()
        self.init_method = "onemin"
        # array of node ids that are activated
        activated = np.array(
            [self.nodelist[-1, 0]]
        )  # create 1D array (not 0D)

        # initial condition
        init = np.zeros(shape=self.nw.num_nodes)
        init[-1] = 1
        self.activated, self.init = (activated, init)

    def init_all_mindeg(self) -> None:
        self.del_attr()
        self.init_method = "allmin"
        degrees = self.nodelist[:, 1]
        mindeg = degrees.min()
        # array of node ids that are activated
        activated = self.nodelist[degrees == mindeg, 0]
        # initial condition
        init = np.where(
            degrees == mindeg,
            np.ones(shape=self.nw.num_nodes),
            np.zeros(shape=self.nw.num_nodes),
        )
        self.activated, self.init = (activated, init)

    def init_one_maxdeg(self) -> None:
        self.del_attr()
        self.init_method = "onemax"
        # array of node ids that are activated
        activated = np.array([self.nodelist[0, 0]])  # create 1D array (not 0D)

        # initial condition
        init = np.zeros(shape=self.nw.num_nodes)
        init[0] = 1
        self.activated, self.init = (activated, init)

    def init_all_maxdeg(self) -> None:
        self.del_attr()
        self.init_method = "allmax"
        degrees = self.nodelist[:, 1]
        maxdeg = degrees.max()
        # array of node ids that are activated
        activated = self.nodelist[degrees == maxdeg, 0]
        # initial condition
        init = np.where(
            degrees == maxdeg,
            np.ones(shape=self.nw.num_nodes),
            np.zeros(shape=self.nw.num_nodes),
        )
        self.activated, self.init = (activated, init)

    def init_maxdeg_neighbour(
        self, k: int = 1, max_num_activated: int = None
    ) -> None:
        """
        A node with maximum degree and its k-th neighbours are activated.

        Parameters
        ----------
        k : int, optional
        max_num_activated : int or None, optional
            When specified, some of neighbouring nodes are extracted so that
            the number of activated nodes is not greater than max_num_activated.
            If None, all the k-th neighbours of the node with maximum degree are activated.
        """
        self.del_attr()
        self.init_method = "maxneigh"
        hub = np.array([self.nodelist[0, 0]])  # a node with maximum degree
        d_neighbours = self.nw.get_grouped_by_dist(hub)
        # array of node names that are activated
        if max_num_activated is None:
            activated = np.array(
                sum(
                    [hub.tolist()],
                    [
                        d_neighbours[d]
                        for d in d_neighbours.keys()
                        if 0 < d <= k
                    ],
                )
            )
        else:
            activated = hub.tolist()
            for d in range(k):
                d += 1
                n = max_num_activated - len(activated)
                if len(d_neighbours[d]) <= n:
                    # all the i-th neighbours where i <= d can be activated.
                    activated += d_neighbours[d]
                else:
                    activated += d_neighbours[d][:n]
            activated = np.array(activated)
        # initial condition
        init = np.where(
            np.isin(self.nodelist[:, 0], activated),
            np.ones(shape=self.nw.num_nodes),
            np.zeros(shape=self.nw.num_nodes),
        )
        self.activated, self.init = (activated, init)

    #####################
    # visualizaion tools
    #####################
    def plot_mean_xt(self, ax: mpl.axes.Axes, t_id: int) -> mpl.lines.Line2D:
        """
        Parameters
        ----------
        ax : Axes
        t_id : int
        """
        d_nodebydist = self.nw.group_by_dist(self.activated)
        t = self._res_t[t_id]
        xt = self._res_y[:, t_id]

        l_dist = list(d_nodebydist.keys())

        # sort ut by node names
        idx = np.argsort(self.nodelist[:, 0])
        xt_byname = xt[idx]
        # calculate average u_i(t) for each group
        xt_ave = [xt_byname[d_nodebydist[d]].mean() for d in l_dist]

        # create plot
        ax.set_title(f"$t = {t:.3f}$")
        ax.set_xlabel("distance from source")
        ax.set_ylabel("average $x_i(t)$")
        ax.set_ylim((self.ymin, self.ymax))
        line = ax.plot(l_dist, xt_ave, ls=":", c="k", marker="o")
        return line

    def plot_escape_time(
        self,
        fname_c: str,
        fname_f: str = None,
        fname_s: str = None,
        quiescent: bool = False,
        beginning: bool = False,
        transient: bool = False,
        theory: float = None,
        plot_det: bool = False,
        figwidth: float = None,
        figheight: float = None,
    ):
        """
        Parameters
        ----------
        fname_c : str
            File name of the csv file containing the collective escape times.
        fname_f : str, optional
            File name of the csv file containing the first escape times.
        fname_s : str, optional
            File name of the csv file containing the start times.
        quiescent : bool, optional
            If `True`, the durations of the quiescend periods are plotted.
        beginning : bool, optional
            If `True`, the durations of the beginning periods are plotted.
        transient : bool, optional
            If `True`, the durations of transient processes, calculated as
            (duration) = (collective escape time) - (start time),
            are plotted.
            Ignored when `beginning` is `True`.
        theory : float, optional
            Theoretical mean escape time for the case of N = 1 (no coupling).
        plot_det : bool, optional
            Show the stationary fraction of active nodes in deterministic case.
        figwidth : float, optional
        figheight : float, optional

        Note
        ----
        `self._D` is assumed to coincide with a list of `D` values for the csv files.
        """
        a_coll = np.loadtxt(fname_c, delimiter=",")
        if quiescent or beginning:
            if fname_f is None:
                raise ValueError(
                    "fname_f must be given when quiescent or beginning is True."
                )
            a_first = np.loadtxt(fname_f, delimiter=",")
        if beginning or transient:
            if fname_s is None:
                raise ValueError(
                    "fname_s must be given when beginning or transient is True."
                )
            a_start = np.loadtxt(fname_s, delimiter=",")

        figw = 4 if plot_det else 3.5  # default
        figw = figw if figwidth is None else figwidth  # override
        figh = 2.5 if figheight is None else figheight

        if quiescent:
            data = a_first
        elif beginning:
            data = a_start - a_first
        elif transient:
            data = a_coll - a_start
        else:
            data = a_coll

        mean = data.mean(axis=1)
        std = data.std(axis=1)

        fig, ax = plt.subplots(
            figsize=(figw, figh), layout="constrained", dpi=300
        )

        if theory is not None:
            ax.axhline(y=theory, c="k", ls="dashed", lw=0.5)

        ax.errorbar(
            x=self._D,
            y=mean,
            yerr=std,
            marker="x",
            ls="none",
            c="tab:blue",
            elinewidth=0.5,
            capsize=2,
        )
        ax.set_xlabel("$D$")

        if quiescent:
            ax.set_ylabel("Mean quiescent duration")
        elif beginning:
            ax.set_ylabel("Mean beginning duration")
        elif transient:
            ax.set_ylabel("Mean transient duration")
        else:
            ax.set_ylabel("Mean collective escape time")

        ax.grid(axis="y", c="lightgray", zorder=1, lw=0.5)

        if plot_det:
            ax2 = ax.twinx()
            ax2.plot(
                self._D,
                100 * self.a_det_prop,
                ls="none",
                c="tab:orange",
                marker="+",
            )
            ax2.axhline(y=0, ls="solid", lw=1, c="k", zorder=5)
            ax2.set_ylabel(r"% of active nodes")

            # put ax in front of ax2
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)  # hide the 'canvas'

        plt.show()
        return fig

    def plot_cv(
        self,
        fname_c: str,
        fname_f: str,
        fname_s: str,
        figw: float = 6.5,
        figh: float = 2.5,
        ytop: float = None,
    ):
        """
        Parameters
        ----------
        fname_c : str
            File name of the csv file containing the collective escape times.
        fname_f : str, optional
            File name of the csv file containing the first escape times.
        fname_s : str, optional
            File name of the csv file containing the start times.
        figw : float, optional
        figh : float, optional
        ytop : float, optional
        """
        a_coll = np.loadtxt(fname_c, delimiter=",")
        a_first = np.loadtxt(fname_f, delimiter=",")
        a_start = np.loadtxt(fname_s, delimiter=",")
        a_begin = a_start - a_first
        a_trans = a_coll - a_start

        cv = lambda arr: arr.std(axis=1) / arr.mean(axis=1)
        cv_coll = cv(a_coll)
        cv_first = cv(a_first)
        cv_begin = cv(a_begin)
        cv_trans = cv(a_trans)

        fig, ax = plt.subplots(
            figsize=(figw, figh), layout="constrained", dpi=300
        )

        x = np.arange(self._D.size)  # the label locations
        width = 0.2  # the width of the bars
        alp = 0.8

        rects1 = ax.bar(
            x - width * 3 / 2,
            cv_coll,
            width,
            label="collective",
            color="tab:gray",
            zorder=5,
        )
        rects2 = ax.bar(
            x - width / 2,
            cv_first,
            width,
            label="quiescent",
            alpha=alp,
            zorder=5,
        )
        rects3 = ax.bar(
            x + width / 2,
            cv_begin,
            width,
            label="beginning",
            alpha=alp,
            zorder=5,
        )
        rects4 = ax.bar(
            x + width * 3 / 2,
            cv_trans,
            width,
            label="transient",
            alpha=alp,
            zorder=5,
        )

        ax.set_ylim(top=ytop)
        ax.grid(axis="y", c="lightgray", zorder=1, lw=0.5)
        ax.set_xmargin(0.02)
        ax.set_xlabel("$D$")
        ax.set_xticks(x, [f"{d:.2f}" for d in self._D])
        ax.set_ylabel("Coefficient of variation")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=4)

        plt.show()
        return fig

    def stackplot_times(
        self,
        fname_c: str,
        fname_f: str,
        fname_s: str,
        theory: float = None,
        figw: float = 3.5,
        figh: float = 2.5,
    ):
        """
        Parameters
        ----------
        fname_c : str
            File name of the csv file containing the collective escape times.
        fname_f : str
            File name of the csv file containing the first escape times.
        fname_s : str
            File name of the csv file containing the start times.
        theory : float, optional
            Theoretical mean escape time for the case of N = 1 (no coupling).
        figwidth : float, optional
        figheight : float, optional

        Note
        ----
        `self._D` is assumed to coincide with a list of `D` values for the csv files.
        """
        a_coll = np.loadtxt(fname_c, delimiter=",")
        a_first = np.loadtxt(fname_f, delimiter=",")
        a_start = np.loadtxt(fname_s, delimiter=",")
        a_begin = a_start - a_first  # fluctuating process
        a_trans = a_coll - a_start

        ave = lambda arr: arr.mean(axis=1)
        ave_first = ave(a_first)
        ave_begin = ave(a_begin)
        ave_trans = ave(a_trans)

        fig, ax = plt.subplots(
            figsize=(figw, figh), layout="constrained", dpi=300
        )
        if theory is not None:
            ax.axhline(y=theory, c="k", ls="dashed", lw=0.5)

        width = 0.6 * 0.01
        rects1 = ax.bar(self._D, ave_first, width, label="quiescent", zorder=5)
        rects2 = ax.bar(
            self._D,
            ave_begin,
            width,
            bottom=ave_first,
            label="beginning",
            zorder=5,
        )
        rects3 = ax.bar(
            self._D,
            ave_trans,
            width,
            bottom=ave(a_start),
            label="transient",
            zorder=5,
        )

        ax.grid(axis="y", c="lightgray", zorder=1, lw=0.5)
        ax.set_xlabel("$D$")
        ax.set_ylabel("time", labelpad=1.5)
        ax.legend(
            loc="upper left", bbox_to_anchor=(0.15, 1), borderaxespad=0.2
        )

        plt.show()
        return fig

    def plot_sample_path(
        self,
        top_k: int = 5,
        y: float = None,
        showtau: bool = True,
        showd:bool=True,
        figw: float = 6.5,
        figh: float = 2.5,
        step: int = 10,
        topalpha: float = 1,
        restalpha: float = 0.3,
        marker: str = ".",
        msindot: float = 5,
        **markerkwargs,
    ):
        """
        Parameters
        ----------
        top_k : int, optional
        y : float, optional
            If a value is given, a horizontal dashed line is shown at y = `y`.
        showtau : bool, optional
            If `True`, `tau_f`, `tau_s` and `tau_c` are shown.
        showd : bool, optional
            If `True`, the value of D is shown in the figure.
        figw : float, optional
        figh : float, optional
        step : int, optional
            `res_y[::step]` will be plotted.
        topalpha : float, optional
        restalpha : float, optional
        marker : str, optional
        msindot : float, optional
        **markerkwargs
            Passed to `plot()`.
        """
        # function for changing unit (dot -> point)
        ms = lambda x: x * 72 / fig.dpi

        fig, ax = plt.subplots(
            figsize=(figw, figh), layout="constrained", dpi=300
        )

        if y is not None:
            ax.axhline(y=y, ls="dashed", lw=0.5, c="k", zorder=1)

        if showtau:
            taums: float = 30
            taumew: float = 6
            ts_prop: np.ndarray = (self._res_y >= self.threshold).mean(axis=0)
            tau_f: float = self.get_tau_f(self._res_t, ts_prop)
            tau_s: float = self.get_tau_s(self._res_t, ts_prop)
            tau_c: float = self.get_tau_c(self._res_t, ts_prop)

            taus: list = [tau_f, tau_s, tau_c]
            colors: list = ["tab:blue", "tab:orange", "tab:green"]
            labels: list = [
                r"$\tau_{\mathrm{f}}$",
                r"$\tau_{\mathrm{s}}$",
                r"$\tau_{\mathrm{c}}$",
            ]
            markers: list = ["o", "v", "s"]
            for tau, c, label, m in zip(taus, colors, labels, markers):
                ax.axvline(x=tau, c=c, lw=0.8, ls="dashed", zorder=4)
                ax.plot(
                    tau,
                    self.threshold,
                    c=c,
                    label=label,
                    marker=m,
                    ls="none",
                    fillstyle="none",
                    ms=ms(taums),
                    mew=ms(taumew),
                    zorder=5,
                )
            ax.legend(handlelength=1.2)

        if top_k > 0:
            ax.plot(
                self._res_t[::step],
                self._res_y[:top_k, ::step].T,
                ls="none",
                marker=marker,
                markersize=ms(msindot + 3),
                c="k",
                mec="none",
                alpha=topalpha,
                zorder=3,
                **markerkwargs,
            )

        ax.plot(
            self._res_t[::step],
            self._res_y[top_k:, ::step].T,
            ls="none",
            marker=marker,
            markersize=ms(msindot),
            c="tab:gray",
            mec="none",
            alpha=restalpha,
            zorder=2,
            **markerkwargs,
        )

        if showd:
            ax.text(
                x=0.01,
                y=0.98,
                s=f"$D = {self.D:.2f}$",
                ha="left",
                va="top",
                transform=ax.transAxes,
            )

        ax.set_xmargin(0)
        ax.set_xlabel("time $t$", fontsize="large")
        ax.set_ylabel("$x_i$", fontsize="large")
        ax.set_ylim((self.ymin, self.ymax))
        plt.show()
        return fig

    ##################
    # getter / setter
    ##################
    @property
    def D(self) -> float:
        return self._D

    @D.setter
    def D(self, newD: float) -> None:
        self.del_attr()  # delete existing results to avoid confusion
        self._D = newD

    @property
    def r(self) -> float:
        return self._r

    @r.setter
    def r(self, newr: float) -> None:
        self.del_attr()  # delete existing results to avoid confusion
        self._r = newr

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, newa: float) -> None:
        self.del_attr()  # delete existing results to avoid confusion
        self._alpha = newa

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, newth: float | None) -> None:
        self.del_attr()  # delete existing results to avoid confusion
        self._threshold = (self.r + 1) / 2 if newth is None else newth

    @property
    def prop(self) -> float:
        return self._prop

    @prop.setter
    def prop(self, newprop: float) -> None:
        self.del_attr()  # delete existing results to avoid confusion
        self._prop = newprop

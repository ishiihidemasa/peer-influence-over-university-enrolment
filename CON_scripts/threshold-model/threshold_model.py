# methods for analysis of threshold models
# Chapter 2. Threshold model for university enrolment

from itertools import product

import numpy as np
from matplotlib import pyplot as plt
from numba import njit


@njit
def f(
    p: float | np.ndarray, t: float, g: float, b: float, r: float
) -> np.ndarray:
    """
    Parameters
    ----------
    p : float | np.ndarray
    t : float
        theta
    g : float
        gamma
    b : float
        beta
    r : float
        rho_i
    """
    eb2 = np.exp(b / 2)
    ebp = np.exp(b * p)
    egt = np.exp(g * t)
    # fmt: off
    y = (
        1 / (egt + 1)
        + (1 / (1 + np.exp(g * (t - 1))) - 1 / (1 + egt)) / 2
        * (1 + r - 2 * r * eb2 * (ebp - 1) / ((eb2 - 1) * (eb2 + ebp)))
    )
    # fmt: on
    f = 1 - t - np.log(y / (1 - y)) / g
    return f


@njit
def df(
    p: float | np.ndarray, t: float, g: float, b: float, r: float
) -> np.ndarray:
    """
    Parameters
    ----------
    p : float | np.ndarray
    t : float
        theta
    g : float
        gamma
    b : float
        beta
    r : float
        rho_i
    """
    eb = np.exp(b)
    eb2 = np.exp(b / 2)
    ebp = np.exp(b * p)
    eg = np.exp(g)
    egt = np.exp(g * t)
    # fmt: off
    df = (
        4 / g * b * r * (eb - 1) * (eg - 1) * (egt + 1) * (egt + eg) * ebp * eb2 / ((
            -2 * eb2 * egt + 2 * eb * egt + (1 - r) * eb * eg - (1 + r) * eb2 * eg
            - eb2 * (1 - r) + eb * (1 + r) - 2 * egt * ebp + 2 * egt * ebp * eb2
            - (1 - r) * eg * ebp + (1 + r) * eg * ebp * eb2 + (1 - r) * ebp * eb2 - (1 + r) * ebp
        ) * (egt * (
            (r -1) * eb2 * eg + (1 + r) * eb * eg + (1 - r) * eb - (1 + r) * eb2
            + (1 - r) * eg * ebp * eb2 - (1 + r) * eg * ebp - (1 - r) * ebp 
            + (1 + r) * ebp * eb2
        ) + 2 * eg * (eb2 - 1) * (eb2 + ebp)
        ))
    )
    # fmt: on
    return df


@njit
def _find_fp(
    t: float, g: float, b: float, r: float, num_p: int
) -> tuple[np.ndarray]:
    """
    Parameters
    ----------
    t : float
        theta
    g : float
        gamma
    b : float
        beta
    r : float
        rho_i
    num_p : int
        The number of scetions in `[0, 1]`.
    """
    if r == 1:
        # When rho_I = 1, exclude p = 0 and 1 to avoid instability
        margin = 1 / num_p
    else:
        margin = 0
    a_p = np.linspace(margin, 1 - margin, num_p)  # array of p values
    # a_p = np.linspace(0, 1, num_p)  # array of p values
    a_f = f(a_p, t, g, b, r)  # array of f(p) values

    # estimate fixed points
    a_diff = a_f - a_p  # element-wise: difference between f(p) and p
    # if f(p) crosses 45 degree line in the RHS interval
    a_ifcross = (
        a_diff[:-1] * a_diff[1:]
    ) < 0  # False if sign of diff_i and diff_{i + 1} coincide

    # p: left points of intervals containing fixed points
    a_pi = a_p[:-1][a_ifcross]
    # p: right points of intervals containing fixed points
    a_pii = a_p[1:][a_ifcross]
    # f(p): at left points of intervals containing fixed points
    a_fpi = a_f[:-1][a_ifcross]
    # f(p): at right points of intervals containing fixed points
    a_fpii = a_f[1:][a_ifcross]
    # approximate value of p at fixed points
    a_fp = (a_pii * a_fpi - a_pi * a_fpii) / (a_pii - a_pi - a_fpii + a_fpi)

    # estimate stabilities of fixed points
    # df() is used to calculate derivatives
    a_dffp = df(a_fp, t, g, b, r)  # f'(p) values at fixed points

    # True if approximate derivative absolute value is less than 1
    a_stable = np.abs(a_dffp) < 1

    return (a_p, a_f, a_fp, a_dffp, a_stable)


def find_fp(
    t: float, g: float, b: float, r: float, num_p: int
) -> tuple[np.ndarray]:
    """
    Parameters
    ----------
    t : float
        theta
    g : float
        gamma
    b : float
        beta
    r : float
        rho_i
    num_p : int
        The number of scetions in `[0, 1]`.
    """
    # for p values in (0, 1):
    (a_p, a_f, a_fp, a_dffp, a_stable) = _find_fp(t, g, b, r, num_p)

    # p = 0 and 1 are fixed points iff rho_I = 1
    if r == 1:
        a_fp = np.append(np.insert(a_fp, 0, 0), 1)

        df_vals = df(np.array([0, 1], dtype="f8"), t, g, b, r)

        # stability of p = 0:
        if np.abs(df_vals[0]) < 1:
            a_stable = np.insert(a_stable, 0, True)
        else:
            a_stable = np.insert(a_stable, 0, False)

        # stability of p = 1:
        if np.abs(df_vals[1]) < 1:
            a_stable = np.append(a_stable, True)
        else:
            a_stable = np.append(a_stable, False)

    return (a_p, a_f, a_fp, a_dffp, a_stable)


def plot_f(
    t: float,
    g: float,
    b: float,
    r: float,
    a_p: np.ndarray,
    a_f: np.ndarray,
    a_fp: np.ndarray,
    a_stable: np.ndarray,
    save: bool = False,
    date: str = None,
    ext: str = ".png",
):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlabel("$p_{n}$")
    ax.set_ylabel("$p_{n + 1}$")
    ax.set_aspect("equal")
    ax.set_title(
        rf"$\theta = {t:.2f}$, $\gamma = {g}$, $\beta = {b}$, $\rho = {r}$"
    )

    ax.plot(a_p, a_f, label="$f(p)$")
    ax.plot(a_p, a_p, ls="dashed", c="k")
    ax.plot(
        a_fp[a_stable],
        a_fp[a_stable],
        ls="none",
        marker="o",
        mfc="red",
        mec="red",
    )
    ax.plot(
        a_fp[~a_stable],
        a_fp[~a_stable],
        ls="none",
        marker="o",
        mfc="white",
        mec="red",
    )
    ax.legend()
    if save:
        fig.savefig(
            date
            + f"_t{int(100 * t)}-g{int(g)}-b{int(b)}-r{int(100 * r)}"
            + ext,
            bbox_inches="tight",
            dpi=300,
        )
    plt.show()


def get_if_fixed(
    t: float | np.ndarray,
    g: float | np.ndarray,
    b: float | np.ndarray,
    r: float | np.ndarray,
) -> np.ndarray:
    if_fixed = np.array(
        [type(elem) != np.ndarray for elem in [t, g, b, r]], dtype=bool
    )
    return if_fixed


def get_phase(
    num_p: int,
    t: float | np.ndarray,
    g: float | np.ndarray,
    b: float | np.ndarray,
    r: float | np.ndarray,
) -> tuple[np.ndarray, list[list[tuple[float]]]]:
    """
    Phases are defined according to the number and stabilities of fixed points.
    phase 0: S
    phase 1: SU
    phase 2: US
    phase 3: SUS
    phase 4: USU
    phase 5: f(p)=p
    phase 6: others

    Parameters
    ----------
    num_p : int
    t: float | np.ndarray
    g: float | np.ndarray
    b: float | np.ndarray
    r: float | np.ndarray

    Returns
    -------
    if_fixed : np.ndarray
        Each element (`bool`) represents `t`, `g`, `b` and `r`.
    l_phase : list[list[tuple[float]]]
    """
    if_fixed = get_if_fixed(t, g, b, r)
    (arr_p1, arr_p2) = np.array([t, g, b, r], dtype=object)[~if_fixed]
    fixed = np.array(["t", "g", "b", "r"], dtype=str)[if_fixed]

    l_phase = [[] for _ in range(7)]

    if set(fixed) == {"t", "g"}:
        # rho_i-beta plane
        # theta and gamma are fixed
        for _b, _r in product(arr_p1, arr_p2):
            if _if_identity(t, g, _b, _r):
                l_phase[5].append((_r, _b))
                continue
            a_p, a_f, a_fp, a_dffp, a_stable = find_fp(
                t=t, g=g, b=_b, r=_r, num_p=num_p
            )
            _classify(
                l_phase=l_phase, p1=_r, p2=_b, a_fp=a_fp, a_stable=a_stable
            )

    elif set(fixed) == {"t", "b"}:
        # rho_i-gamma plane
        # theta and beta are fixed
        for _g, _r in product(arr_p1, arr_p2):
            if _if_identity(t, _g, b, _r):
                l_phase[5].append((_r, _g))
                continue
            a_p, a_f, a_fp, a_dffp, a_stable = find_fp(
                t=t, g=_g, b=b, r=_r, num_p=num_p
            )
            _classify(
                l_phase=l_phase, p1=_r, p2=_g, a_fp=a_fp, a_stable=a_stable
            )

    elif set(fixed) == {"t", "r"}:
        # gamma-beta plane
        # theta and rhoi are fixed
        for _g, _b in product(arr_p1, arr_p2):
            if _if_identity(t, _g, _b, r):
                l_phase[5].append((_g, _b))
                continue
            a_p, a_f, a_fp, a_dffp, a_stable = find_fp(
                t=t, g=_g, b=_b, r=r, num_p=num_p
            )
            _classify(
                l_phase=l_phase, p1=_g, p2=_b, a_fp=a_fp, a_stable=a_stable
            )

    elif set(fixed) == {"g", "b"}:
        # rho_i-theta plane
        # gamma and beta are fixed
        for _t, _r in product(arr_p1, arr_p2):
            if _if_identity(_t, g, b, _r):
                l_phase[5].append((_r, _t))
                continue
            a_p, a_f, a_fp, a_dffp, a_stable = find_fp(
                t=_t, g=g, b=b, r=_r, num_p=num_p
            )
            _classify(
                l_phase=l_phase, p1=_r, p2=_t, a_fp=a_fp, a_stable=a_stable
            )

    elif set(fixed) == {"g", "r"}:
        # beta-theta plane
        # gamma and rhoi are fixed
        for _t, _b in product(arr_p1, arr_p2):
            if _if_identity(_t, g, _b, r):
                l_phase[5].append((_b, _t))
                continue
            a_p, a_f, a_fp, a_dffp, a_stable = find_fp(
                t=_t, g=g, b=_b, r=r, num_p=num_p
            )
            _classify(
                l_phase=l_phase, p1=_b, p2=_t, a_fp=a_fp, a_stable=a_stable
            )

    elif set(fixed) == {"b", "r"}:
        # gamma-theta plane
        # beta and rhoi are fixed
        for _t, _g in product(arr_p1, arr_p2):
            if _if_identity(_t, _g, b, r):
                l_phase[5].append((_g, _t))
                continue
            a_p, a_f, a_fp, a_dffp, a_stable = find_fp(
                t=_t, g=_g, b=b, r=r, num_p=num_p
            )
            _classify(
                l_phase=l_phase, p1=_g, p2=_t, a_fp=a_fp, a_stable=a_stable
            )

    return (if_fixed, l_phase)


def _if_identity(t: float, g: float, b: float, r: float) -> bool:
    return t == 0.5 and b == g and r == 1


def _classify(
    l_phase: list, p1: float, p2: float, a_fp: np.ndarray, a_stable: np.ndarray
) -> None:
    """
    Phases are defined according to the number and stabilities of fixed points.
    phase 0: S
    phase 1: SU
    phase 2: US
    phase 3: SUS
    phase 4: USU
    phase 5: f(p)=p
    phase 6: others

    """
    elem = (p1, p2)
    if a_fp.size == 1:
        if a_stable[0]:
            # 0: S
            l_phase[0].append(elem)
            return
    elif a_fp.size == 2:
        if a_stable[0] and (not a_stable[1]):
            # 1: SU
            l_phase[1].append(elem)
            return
        elif (not a_stable[0]) and a_stable[1]:
            # 2: US
            l_phase[2].append(elem)
            return
    elif a_fp.size == 3:
        if a_stable[0] and (not a_stable[1]) and a_stable[2]:
            # 3: SUS
            l_phase[3].append(elem)
            return
        elif (not a_stable[0]) and a_stable[1] and (not a_stable[2]):
            # 4: USU
            l_phase[4].append(elem)
            return

    # last element: unexpected cases
    print(f"(param1, param2) = ({p1}, {p2}):")
    print("stable fixed points:", a_fp[a_stable])
    print("unstable fixed points:", a_fp[~a_stable])
    l_phase[6].append(elem)
    return


def set_size(w, h, ax):
    """
    Reference: https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    w, h: width, height in inches
    """
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = w / (r - l)
    figh = h / (t - b)
    ax.figure.set_size_inches(figw, figh)


def plot_pd(
    t: float | np.ndarray,
    g: float | np.ndarray,
    b: float | np.ndarray,
    r: float | np.ndarray,
    if_fixed: np.ndarray,
    l_phase: list,
    num_p: int,
    l_marker: list = None,
    ms: float = 1,
    save: bool = False,
    date: str = None,
    ext: str = ".png",
):
    """
    Phases are defined according to the number and stabilities of fixed points.
    phase 0: S
    phase 1: SU
    phase 2: US
    phase 3: SUS
    phase 4: USU
    phase 5: f(p)=p
    phase 6: others

    """
    axtitle_main = ""
    d_label = {
        "t": r"$\theta$",
        "g": r"$\gamma$",
        "b": r"$\beta$",
        "r": r"$\rho$",
    }
    l_label = ["S", "SU", "US", "SUS", "USU", "$f(p) = p$", "others"]
    l_color = [
        "tab:green",
        "tab:pink",
        "tab:cyan",
        "tab:brown",
        "tab:orange",
        "k",
        "antiquewhite",
    ]

    fixed = np.array(["t", "g", "b", "r"], dtype=str)[if_fixed]
    param = np.array(["t", "g", "b", "r"], dtype=str)[~if_fixed]

    if set(fixed) == {"t", "g"}:
        # rho_i-beta plane
        # theta and gamma are fixed
        (x_param, y_param) = (d_label["r"], d_label["b"])
        axtitle = (
            axtitle_main
            + d_label["t"]
            + f"$= {t}$, "
            + d_label["g"]
            + f"$= {g}$"
        )

    elif set(fixed) == {"t", "b"}:
        # rho_i-gamma plane
        # theta and beta are fixed
        (x_param, y_param) = (d_label["r"], d_label["g"])
        axtitle = (
            axtitle_main
            + d_label["t"]
            + f"$= {t}$, "
            + d_label["b"]
            + f"$= {b}$"
        )

    elif set(fixed) == {"t", "r"}:
        # gamma-beta plane
        # theta and rhoi are fixed
        (x_param, y_param) = (d_label["g"], d_label["b"])
        axtitle = (
            axtitle_main
            + d_label["t"]
            + f"$= {t}$, "
            + d_label["r"]
            + f"$= {r}$"
        )

    elif set(fixed) == {"g", "b"}:
        # rho_i-theta plane
        # gamma and beta are fixed
        (x_param, y_param) = (d_label["r"], d_label["t"])
        axtitle = (
            axtitle_main
            + d_label["g"]
            + f"$= {g}$, "
            + d_label["b"]
            + f"$= {b}$"
        )

    elif set(fixed) == {"g", "r"}:
        # beta-theta plane
        # gamma and rhoi are fixed
        (x_param, y_param) = (d_label["b"], d_label["t"])
        axtitle = (
            axtitle_main
            + d_label["g"]
            + f"$= {g}$, "
            + d_label["r"]
            + f"$= {r}$"
        )

    elif set(fixed) == {"b", "r"}:
        # gamma-theta plane
        # beta and rhoi are fixed
        (x_param, y_param) = (d_label["g"], d_label["t"])
        axtitle = (
            axtitle_main
            + d_label["b"]
            + f"$= {b}$, "
            + d_label["r"]
            + f"$= {r}$"
        )

    fig, ax = plt.subplots(dpi=200)
    ax.set_title(axtitle)
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    # ax.set_aspect("equal")
    ax.margins(0.01)

    for i, c in enumerate(l_phase):
        if len(c) < 1:
            continue
        c = np.array(c).T
        ax.plot(
            c[0],
            c[1],
            ls="none",
            marker="s",
            ms=ms,
            label=l_label[i],
            c=l_color[i],
        )

    if l_marker:  # None == False
        for point in l_marker:
            ax.plot(
                *point, ls="none", marker="x", markersize=13, c="k", zorder=9
            )

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        markerscale=5 / ms,
        handletextpad=0,
    )

    set_size(w=2.5, h=2.5, ax=ax)

    if save:
        date = date + "_" if date != "" else date
        fig.savefig(
            date + f"numpd_{num_p}_" + "-".join(param) + ext,
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()


def plot_bd(
    const1: float,
    const2: float,
    fix: np.ndarray,
    b: np.ndarray,
    param_role: list[str],
    num_p: int,
    axtitlepref: str = "",
    line: bool = False,
    lw: float = 1,
    ms: float = 1,
    umarker: str = "x",  # for indicating unstable fixed points
    umew: float = 0.5,  # for indicating unstable fixed points
    uscale: float = 4,  # for indicating unstable fixed points
    legend_loc: str = "best",
    bbox_to_anchor: tuple = None,
    save: bool = False,
    date: str = None,
    ext: str = ".png",
    **ukwargs,
) -> None:
    """
    Parameters
    ----------
    const1 : float
    const2 : float
    fix : np.ndarray
    b_param : np.ndarray
        Values of the bifurcation parameter.
    param_role : list[str]
        A list indicating the roles of parameters (`t`, `g`, `b`, and `r`) in this order.
        `const1`, `const2`, `fix` and `b_param` must be denoted by "c1", "c2", "f" and "b", respectively.
        Example: `["b", "c1", "c2", "f"]`
    num_p : int
    axtitlepref : str, optional
    line : bool, optional
        When True, solid and dashed lines instead of markers will be plotted.
        This feature is not implemented.
    lw : float, optional
    ms : float, optional
    umarker : str, optional
    umew : float, optional
        Marker edge width.
    uscale : float, optional
    legend_loc : str, optional
    bbox_to_anchor : tuple, optional
    save : bool, optional
    date : str, optional
    ext : str, optional
    **ukwargs
        Keyword arguments that are passed to plotting function
        for unstable fixed points.
    """
    # collect fixed points
    l_sfp = []
    l_ufp = []

    for v_fix in fix:
        l_sfp_i = []
        l_ufp_i = []
        for v_bif in b:
            d_vals = {"c1": const1, "c2": const2, "f": v_fix, "b": v_bif}
            args = [d_vals[s] for s in param_role]
            a_p, a_f, a_fp, a_dffp, a_stable = find_fp(*args, num_p=num_p)
            stable = a_fp[a_stable]
            unstable = a_fp[~a_stable]
            elem_s = [(v_bif, fp) for fp in stable]
            elem_u = [(v_bif, fp) for fp in unstable]
            l_sfp_i += elem_s
            l_ufp_i += elem_u
        l_sfp.append(np.array(l_sfp_i, dtype="f8"))
        l_ufp.append(np.array(l_ufp_i, dtype="f8"))

    # draw a figure
    d_role2name = dict(zip(param_role, ("t", "g", "b", "r")))
    d_label = {
        "t": r"$\theta$",
        "g": r"$\gamma$",
        "b": r"$\beta$",
        "r": r"$\rho$",
    }

    # generate axes title
    axtitle = (
        axtitlepref
        + d_label[d_role2name["c1"]]
        + f"$= {const1}$, "
        + d_label[d_role2name["c2"]]
        + f"$= {const2}$"
    )

    fig, ax = plt.subplots(dpi=200, layout="constrained")
    ax.set_title(axtitle)
    ax.set_xlabel(d_label[d_role2name["b"]])
    ax.set_ylabel("Fixed point $p^*$")
    ax.margins(0.03)

    if line:  # plot with lines
        # TODO: when bistability exists, data for two invariant curves must be separated
        raise NotImplementedError

    else:  # plot with markers
        for i, v_fix in enumerate(fix):
            c = plt.get_cmap("tab10")(i)
            labelstr = d_label[d_role2name["f"]] + f"$= {v_fix}$"
            if l_sfp[i].size > 0:
                ax.plot(
                    l_sfp[i][:, 0],
                    l_sfp[i][:, 1],
                    c=c,
                    ls="none",
                    marker=".",
                    ms=ms,
                    label=labelstr,
                )
            if l_ufp[i].size > 0:
                ax.plot(
                    l_ufp[i][:, 0],
                    l_ufp[i][:, 1],
                    c=c,
                    ls="none",
                    marker=umarker,
                    ms=uscale * ms,
                    mew=umew,
                    **ukwargs,
                )

        if bbox_to_anchor is None:
            ax.legend(
                loc=legend_loc,
                markerscale=5 / ms,
                handlelength=0.8,
                handletextpad=0.2,
            )
        else:
            ax.legend(
                loc=legend_loc,
                bbox_to_anchor=bbox_to_anchor,
                markerscale=5 / ms,
                handlelength=0.8,
                handletextpad=0.2,
            )

    set_size(w=2.5, h=2, ax=ax)

    if save:
        date = date + "_" if date != "" else date
        fig.savefig(
            date + f"bd_{num_p}_" + "-".join(param_role) + ext,
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()


def plot_timeseries(
    p_init: np.ndarray,
    t: float,
    g: float,
    b: float,
    r: float,
    num_p: int,
    tmax=100,
    save: bool = False,
    date: str = None,
    ext: str = ".png",
) -> np.ndarray:
    # generate time series data
    l_hist = [p_init]
    for time in range(tmax):
        p_next = f(l_hist[-1], t, g, b, r)
        l_hist.append(p_next)
    a_hist = np.array(l_hist).T  # shape = (len(p_init), tmax)

    # find fixed points
    a_p, a_f, a_fp, a_dffp, a_stable = find_fp(t, g, b, r, num_p=num_p)

    # plot time series and fixed points
    fig, ax = plt.subplots(constrained_layout=True, dpi=200)
    ax.set_title(rf"$(\theta, \gamma, \beta, \rho_I) = ({t}, {g}, {b}, {r})$")

    ax.set_xlim((-0.03 * tmax, 1.03 * tmax))
    ax.set_xlabel("$n$")
    ax.set_ylim((-0.05, 1.05))
    ax.set_ylabel("$p_n$")

    for i in range(a_fp.size):
        ax.axhline(
            y=a_fp[i],
            c="k",
            ls="solid" if a_stable[i] else "dashed",
            lw=1,
            zorder=1,
        )

    for i in range(p_init.size):
        ax.plot(
            a_hist[i],
            ls="solid",
            lw=1,
            marker="x",
            markersize=4,
            label=f"$p_0 = {p_init[i]}$",
        )

    ax.legend()

    set_size(w=2.5, h=2, ax=ax)

    if save:
        date = date + "_" if date != "" else date
        fig.savefig(
            date
            + f"hist_t{int(100 * t)}-g{int(g)}-b{int(b)}-r{int(100 * r)}"
            + ext,
            bbox_inches="tight",
            dpi=300,
        )
    plt.show()

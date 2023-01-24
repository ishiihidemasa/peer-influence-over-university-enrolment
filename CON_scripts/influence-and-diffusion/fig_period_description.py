#%%[markdown]
# # Figure 4.4
# Chapter 4. Diffusion of enrolment behaviour

#%%
import numpy as np
from matplotlib import pyplot as plt
from modules.network import barabasi_albert
from modules.simulation.bistable import bistable

#%%
# n = 20  # number of nodes
r = 0.1
alpha = 0.05

ext: str = ".pdf"  # extension
save: bool = True  # if True, export figures

#%%
nw = barabasi_albert(N=20, m=2, seed=195)
print("Number of nodes: ", nw.num_nodes)
print("Mean degree: ", nw.mean_degree)

bst = bistable(
    nw=nw, t_max=50, r=r, D=0.07, alpha=alpha, threshold=0.8, seed=50
)
bst.init_no_activation()
bst.simulate()

#%%
step = 10

# function for changing unit (dot -> point)
ms = lambda x: x * 72 / fig.dpi

fig, ax = plt.subplots(figsize=(8, 2.5), layout="constrained", dpi=300)

ax.axhline(y=bst.threshold, ls="dashed", lw=0.8, c="k", zorder=2)

# show tau
taums: float = 40
taumew: float = 6
ts_prop: np.ndarray = (bst._res_y >= bst.threshold).mean(axis=0)
tau_f: float = bst.get_tau_f(bst._res_t, ts_prop)
tau_s: float = bst.get_tau_s(bst._res_t, ts_prop)
tau_c: float = bst.get_tau_c(bst._res_t, ts_prop)

taus: list = [tau_f, tau_s, tau_c]
colors: list = ["tab:blue", "tab:orange", "tab:green"]
labels: list = [
    r"$\tau_{\mathrm{f}}$",
    r"$\tau_{\mathrm{s}}$",
    r"$\tau_{\mathrm{c}}$",
]
markers: list = ["o", "s", "v"]
for tau, c, label, m in zip(taus, colors, labels, markers):
    ax.axvline(x=tau, c=c, lw=0.8, ls="dashed", zorder=4)
    ax.plot(
        tau,
        bst.threshold,
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

ax.plot(
    bst._res_t[::step],
    bst._res_y[:, ::step].T,
    ls="solid",
    c="k",
    alpha=0.5,
    lw=ms(4),
    zorder=3,
)

(xmin, xmax) = (24, 37.5)
(ybot, ytop) = (-0.3, 1.05)
ax.fill_between(
    [0, tau_f], y1=ytop, y2=ybot, color=colors[0], zorder=1, alpha=0.1
)
ax.fill_between(
    [tau_f, tau_s], y1=ytop, y2=ybot, color=colors[1], zorder=1, alpha=0.1
)
ax.fill_between(
    [tau_f, tau_c], y1=ytop, y2=ybot, color=colors[2], zorder=1, alpha=0.1
)
ax.text(
    x=(xmin + tau_f) / 2,
    y=ybot + 0.03,
    s="quiescent",
    c=colors[0],
    ha="center",
    va="bottom",
    fontsize=14,
    weight="bold",
)
ax.text(
    x=(tau_f + tau_s) / 2,
    y=ybot + 0.03,
    s="beginning",
    c=colors[1],
    ha="center",
    va="bottom",
    fontsize=14,
    weight="bold",
)
ax.text(
    x=(tau_s + tau_c) / 2,
    y=ybot + 0.03,
    s="transient",
    c=colors[2],
    ha="center",
    va="bottom",
    fontsize=14,
    weight="bold",
)

ax.set_xmargin(0)
ax.set_xlabel("time $t$", fontsize="large")
ax.set_ylabel("$x_i$", fontsize="large")
ax.set_xlim((xmin, xmax))
ax.set_ylim((ybot, ytop))
plt.show()

if save:
    fig.savefig("period_description" + ext, dpi=300, bbox_inches="tight")
# %%

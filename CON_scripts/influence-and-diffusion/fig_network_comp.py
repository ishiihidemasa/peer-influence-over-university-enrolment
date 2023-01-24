#%%[markdown]
# # Figures 3.4, 4.6(a, b), B.1(a, b), B.2(a, b)
# - Chapter 3. Influence of peers over enrolment behaviour
# - Chapter 4. Diffusion of enrolment behaviour

#%%
import numpy as np
from matplotlib import pyplot as plt
from modules import peer_influence
from modules.network import barabasi_albert, config, erdos_renyi


#%%
def degree_dist(
    labels: tuple[str],
    networks: tuple,
    colors: tuple[str] = ("tab:red", "tab:blue"),
    markers: tuple[str] = ("s", "x"),
    figw: float = 3,
    figh: float = 2,
    logx: bool = False,
    logy: bool = False,
):
    """plot degree distribution."""
    fig, ax = plt.subplots(figsize=(figw, figh), dpi=300, layout="constrained")

    for nw, c, m, label in zip(networks, colors, markers, labels):
        # label += f":$N = {nw.num_nodes}$"
        data_k, data_num, data_knn = nw.info_by_degree(min_num=1)
        ax.plot(
            data_k,
            data_num,
            c=c,
            marker=m,
            label=label,
            fillstyle="none",
            ls="none",
            ms=3,
            alpha=0.8,
        )

    ax.set_xlabel("Degree $k$")
    ax.set_ylabel("Number of nodes")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.legend(handlelength=0.8, handletextpad=0.2, borderaxespad=0.15)

    plt.show()
    return fig


#%%
def knn(
    labels: tuple[str],
    networks: tuple,
    colors: tuple[str] = ("tab:red", "tab:blue"),
    markers: tuple[str] = ("s", "x"),
    min_num: int = 5,
    figw: float = 3,
    figh: float = 2,
    logx: bool = False,
    logy: bool = False,
):
    fig, ax = plt.subplots(figsize=(figw, figh), dpi=300, layout="constrained")

    for nw, c, m, label in zip(networks, colors, markers, labels):
        # label += rf":$\langle k_i \rangle = {nw.mean_degree:.2f}$"
        data_k, data_num, data_knn = nw.info_by_degree(min_num=min_num)
        ax.plot(
            data_k,
            data_knn,
            c=c,
            marker=m,
            label=label,
            fillstyle="none",
            ls="none",
            ms=3,
            alpha=0.8,
        )

    ax.set_xlabel("Degree $k$")
    ax.set_ylabel(r"$\langle k_{\mathrm{nn}, i} \rangle (k)$")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.legend(handlelength=0.8, handletextpad=0.2, borderaxespad=0.15)

    plt.show()
    return fig


#%%
# parameter for visualization
save: bool = False
ext: str = ".pdf"

#%%[markdown]
# ## Figure
# Config & BA, N = 1000: see `fig_peer-influence.py`

#%%
# generate config
pim = peer_influence(
    N=1000,
    Nd_tot=100,
    D=1,
    det_method="rand",
    cmean=1,
    cost_dist="pareto",
    deg_dist="power",
    descending_deg=True,
    kmin=3,
    kmax=50,
)
nw1 = config(pim.degrees)

# generate BA
nw2 = barabasi_albert(N=1000, m=6)

# create figure
labels = ("Config", "BA")
networks = (nw1, nw2)
colors = ("tab:red", "tab:blue")
markers = ("s", "o")

fig1 = degree_dist(labels, networks, colors, markers, logx=True, logy=True)
fig2 = knn(labels, networks, colors, markers)
if save:
    fig1.savefig("N1000_degree-dist" + ext, dpi=300, bbox_inches="tight")
    fig2.savefig("N1000_knn" + ext, dpi=300, bbox_inches="tight")

#%%[markdown]
# ## Figure
# Config & BA, N = 100: see `fig_peer-influence.py`

#%%
# generate config
pim = peer_influence(
    N=100,
    Nd_tot=10,
    D=1,
    det_method="rand",
    cmean=1,
    cost_dist="pareto",
    deg_dist="power",
    descending_deg=True,
    kmin=3,
    kmax=50,
)
nw1 = config(pim.degrees)

# generate BA
nw2 = barabasi_albert(N=100, m=4)

# create figure
labels = ("Config", "BA")
networks = (nw1, nw2)

fig1 = degree_dist(labels, networks, logx=True, logy=True)
fig2 = knn(labels, networks)
if save:
    fig1.savefig("N100_degree-dist" + ext, dpi=300, bbox_inches="tight")
    fig2.savefig("N100_knn" + ext, dpi=300, bbox_inches="tight")


#%%[markdown]
# ## Figure 4.6 (a, b)
# ER & BA, N = 500 (1): see `fig_diffusion.py`

#%%
# generate networks
er1 = erdos_renyi(N=500, k=6, seed=1138656)
ba1 = barabasi_albert(N=500, m=3, seed=2023)

# create figure
labels = ("ER", "BA")
networks = (er1, ba1)

fig1 = degree_dist(labels, networks, logx=True, logy=True)
fig2 = knn(labels, networks)
if save:
    fig1.savefig("nw1_degree-dist" + ext, dpi=300, bbox_inches="tight")
    fig2.savefig("nw1_knn" + ext, dpi=300, bbox_inches="tight")

#%%[markdown]
# ## Figure
# ER & BA, N = 500 (2): see `fig_diffusion.py`

#%%
# generate networks
er2 = erdos_renyi(N=500, k=6, seed=12345678)
ba2 = barabasi_albert(N=500, m=3, seed=2778561)

# create figure
labels = ("ER", "BA")
networks = (er2, ba2)

fig1 = degree_dist(labels, networks, logx=True, logy=True)
fig2 = knn(labels, networks)
if save:
    fig1.savefig("nw2_degree-dist" + ext, dpi=300, bbox_inches="tight")
    fig2.savefig("nw2_knn" + ext, dpi=300, bbox_inches="tight")

#%%[markdown]
# ## Figure
# ER & BA, N = 500 (3): see `fig_diffusion.py`

#%%
# generate networks
er3 = erdos_renyi(N=500, k=6, seed=57577)
ba3 = barabasi_albert(N=500, m=3, seed=230124)

# create figure
labels = ("ER", "BA")
networks = (er3, ba3)

fig1 = degree_dist(labels, networks, logx=True, logy=True)
fig2 = knn(labels, networks)
if save:
    fig1.savefig("nw3_degree-dist" + ext, dpi=300, bbox_inches="tight")
    fig2.savefig("nw3_knn" + ext, dpi=300, bbox_inches="tight")

# %%

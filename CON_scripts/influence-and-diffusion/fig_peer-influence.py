#%%[markdown]
# # Figures 3.5, 3.6, 3.7, 3.8
# Chapter 3. Influence of peers on enrolment behaviour

#%%
from modules import peer_influence
from modules.network import barabasi_albert, config
from modules.simulation import peer_influence_sim

#%%
# common parameters
cmean = 1

# if `save` is `True`, figures are saved in the current directory
save = True
ext = ".pdf"

#%%[markdown]
# ## Calculations with DBMF theory

#%%
# common parameters
N = 1000
Nd_tot = 100

#%%[markdown]
# ### Case 1

#%%
D = 0.01

#%%
pim = peer_influence(
    N=N,
    Nd_tot=Nd_tot,
    D=D,
    det_method="rand",
    cmean=cmean,
    globalcoupling=True,
    cost_dist="uniform",
)
pim.calc_mf_and_fp()
fig, s = pim.visualize_res()
if save:
    fig.savefig("C1_" + s + ext, bbox_inches="tight")

#%%
pim = peer_influence(
    N=N,
    Nd_tot=Nd_tot,
    D=D,
    det_method="rand",
    cmean=cmean,
    globalcoupling=True,
    cost_dist="pareto",
)
pim.calc_mf_and_fp()
fig, s = pim.visualize_res()
if save:
    fig.savefig("C1_" + s + ext, bbox_inches="tight")

#%%[markdown]
# ### Case 2

#%%
D = 1

#%%
pim = peer_influence(
    N=N,
    Nd_tot=Nd_tot,
    D=D,
    det_method="rand",
    cmean=cmean,
    homogeneouscost=True,
    deg_dist="uniform",
    descending_deg=False,
    kmin=3,
    kmax=20,
)
pim.calc_mf_and_fp()
fig, s = pim.visualize_res()
if save:
    fig.savefig("C2_" + s + ext, bbox_inches="tight")

#%%
pim = peer_influence(
    N=N,
    Nd_tot=Nd_tot,
    D=D,
    det_method="rand",
    cmean=cmean,
    homogeneouscost=True,
    deg_dist="power",
    descending_deg=False,
    kmin=3,
    kmax=50,
)
pim.calc_mf_and_fp()
fig, s = pim.visualize_res()
if save:
    fig.savefig("C2_" + s + ext, bbox_inches="tight")

#%%[markdown]
# ### Case 3

#%%
D = 1

#%%
pim = peer_influence(
    N=N,
    Nd_tot=Nd_tot,
    D=D,
    det_method="rand",
    cmean=cmean,
    cost_dist="uniform",
    deg_dist="power",
    descending_deg=False,
    kmin=3,
    kmax=50,
)
pim.calc_mf_and_fp()
fig, s = pim.visualize_res()
if save:
    fig.savefig("C3_" + s + ext, bbox_inches="tight")

#%%
pim = peer_influence(
    N=N,
    Nd_tot=Nd_tot,
    D=D,
    det_method="rand",
    cmean=cmean,
    cost_dist="pareto",
    deg_dist="power",
    descending_deg=False,
    kmin=3,
    kmax=50,
)
pim.calc_mf_and_fp()
fig, s = pim.visualize_res()
if save:
    fig.savefig("C3_" + s + ext, bbox_inches="tight")

#%%
pim = peer_influence(
    N=N,
    Nd_tot=Nd_tot,
    D=D,
    det_method="rand",
    cmean=cmean,
    cost_dist="uniform",
    deg_dist="power",
    descending_deg=True,
    kmin=3,
    kmax=50,
)
pim.calc_mf_and_fp()
fig, s = pim.visualize_res()
if save:
    fig.savefig("C3_" + s + ext, bbox_inches="tight")

#%%
pim = peer_influence(
    N=N,
    Nd_tot=Nd_tot,
    D=D,
    det_method="rand",
    cmean=cmean,
    cost_dist="pareto",
    deg_dist="power",
    descending_deg=True,
    kmin=3,
    kmax=50,
)
pim.calc_mf_and_fp()
fig, s = pim.visualize_res()
if save:
    fig.savefig("C3_" + s + ext, bbox_inches="tight")

# %%[markdown]
# ## Numerical results

#%%
def calc_and_plot(
    networktype: str,
    N: int,
    Nd_tot: int,
    D: float,
    cmean: float,
    cost_dist: str,
    deg_dist: str,
    descending_deg: bool,
    kmin: int,
    kmax: int,
    t_max: float,
    save: bool,
    ext: str,
    node_size: float = 15,
    max_t_prop: float = 0.3,
    deglogscale: bool = True,
    msindot: float = 8,
    costlogscale: bool = True,
    ba_m: int = None,
) -> None:
    """
    Obtain numerical solutions and generate four figures.
    """
    # prefix for file names
    prefix = networktype + f"_{N}_" + cost_dist + "_"
    prefix += "de-" if descending_deg else ""
    prefix += deg_dist

    # mean field theory
    pim = peer_influence(
        N=N,
        Nd_tot=Nd_tot,
        D=D,
        det_method="rand",
        cmean=cmean,
        cost_dist=cost_dist,
        deg_dist=deg_dist,
        descending_deg=descending_deg,
        kmin=kmin,
        kmax=kmax,
    )

    # generate graph
    if networktype == "config":
        # - config model
        nw = config(pim.degrees)
        pis = peer_influence_sim(
            nw=nw,
            t_max=t_max,
            D=pim.D,
            det_nodes=pim.det_nodes,
            costs=pim.costs,
            descending_deg=pim.descending_deg,
        )
    elif networktype == "BA":
        # - Barabasi-Albert model
        nw = barabasi_albert(N=N, m=ba_m)
        pis = peer_influence_sim(
            nw=nw,
            t_max=t_max,
            D=pim.D,
            det_nodes=pim.det_nodes,
            costs=pim.costs,
            descending_deg=pim.descending_deg,
        )

    # check degree correlation
    fig = nw.plot_deg_corr(min_num=5, deglogscale=deglogscale, figh=2)
    if save:
        fig.savefig(prefix + "_degcorr" + ext, dpi=300, bbox_inches="tight")

    pis.simulate()

    fig = pis.plot_mean()
    if save:
        fig.savefig(prefix + "_mean" + ext, dpi=300, bbox_inches="tight")

    # plot the state at the last time step
    fig = pis.plot_deg_cost_x(
        t_id=-1, costlogscale=costlogscale, msindot=msindot
    )

    if save:
        fig.savefig(prefix + "_deg-cost-x" + ext, dpi=300, bbox_inches="tight")

    # visualize on network
    fig = pis.snapshot_fig(
        num_t=6,
        max_t_prop=max_t_prop,
        plotmethod="colored-network",
        cmapname="RdYlGn",
        node_size=node_size,
    )
    if save:
        fig.savefig(prefix + "_network" + ext, dpi=300, bbox_inches="tight")


#%%
cost_dist = "pareto"
deg_dist = "power"
descending_deg = True
kmin = 3
kmax = 50

#%%[markdown]
# ### Configuration model

#%%
calc_and_plot(
    networktype="config",
    N=1000,
    Nd_tot=100,
    D=1,
    cmean=1,
    cost_dist=cost_dist,
    deg_dist=deg_dist,
    descending_deg=descending_deg,
    kmin=kmin,
    kmax=kmax,
    t_max=10,
    save=save,
    ext=ext,
)

#%%
calc_and_plot(
    networktype="config",
    N=100,
    Nd_tot=10,
    D=1,
    cmean=1,
    cost_dist=cost_dist,
    deg_dist=deg_dist,
    descending_deg=descending_deg,
    kmin=kmin,
    kmax=kmax,
    t_max=10,
    save=save,
    ext=ext,
    msindot=12,
    deglogscale=False,
    max_t_prop=0.4,
    node_size=25,
)

#%%[markdown]
# ### Barabsi-Albert model

#%%
calc_and_plot(
    networktype="BA",
    N=1000,
    Nd_tot=100,
    D=1,
    cmean=1,
    cost_dist=cost_dist,
    deg_dist=deg_dist,
    descending_deg=descending_deg,
    kmin=kmin,
    kmax=kmax,
    t_max=10,
    save=save,
    ext=ext,
    ba_m=6,
)

#%%
calc_and_plot(
    networktype="BA",
    N=100,
    Nd_tot=10,
    D=1,
    cmean=1,
    cost_dist=cost_dist,
    deg_dist=deg_dist,
    descending_deg=descending_deg,
    kmin=kmin,
    kmax=kmax,
    t_max=10,
    save=save,
    ext=ext,
    msindot=12,
    deglogscale=False,
    max_t_prop=0.4,
    node_size=25,
    ba_m=4,
)
# %%

#%%[markdown]
# # Figures 4.5, 4.6(c, d), 4.8, B.1(c, d), B.2(c, d), B.5
# Chapter 4. Diffusion of enrolment behaviour

#%%
import numpy as np
from modules.network import barabasi_albert, erdos_renyi
from modules.simulation.bistable import bistable

#%%
# MODEL PARAMETERS
N = 500  # number of nodes
r = 0.1
alpha = 0.05

a_D = np.linspace(0, 0.11, 12)  # for calculating escape times
l_D = a_D[[1, 5, 10]].tolist()  # for visualizing sample paths

# PARAMETERS FOR ANALYSIS
# whether to run simulations or not
# - when the data already exist, set to False
calculate: bool = False  
# threshold for the active state
# - if None, threshold is (x_S + x_A) / 2
threshold = 0.8  
# the number of trials for each D value
M = 100
# numerical integration lasts for (max_epoch * bistable.t_max) at longest
max_epoch = 10

# PARAMETERS FOR VISUALIZATION
theory = 75.17  # mean escape time (N = 1)

ext: str = ".pdf"  # extension
save: bool = False  # if True, export figures

#%%
def get_filenames(_bst, _calc: bool, _prefix: str, _fn_base: str):
    if _calc:
        # calculate data
        _fn_f, _fn_s, _fn_c = _bst.measure_escape_time(
            M=M, max_epoch=max_epoch, prefix=_prefix
        )
    else:
        # if data are already at hand
        _fn_f: str = _fn_base + "first.csv"
        _fn_s: str = _fn_base + "start.csv"
        _fn_c: str = _fn_base + "collective.csv"
    return (_fn_f, _fn_s, _fn_c)

#%%
def fig_on_times(
    _bst: bistable,
    _fn_c: str,
    _fn_f: str,
    _fn_s: str,
    _save: bool,
    _prefix: str,
):
    # fmt: off
    fig = bst.stackplot_times(_fn_c, _fn_f, _fn_s, theory=theory)
    #fig_c = _bst.plot_escape_time(_fn_c, theory=theory, plot_det=False)
    #fig_q = _bst.plot_escape_time(_fn_c, fname_f=_fn_f, quiescent=True, plot_det=False)
    fig_b = _bst.plot_escape_time(_fn_c, fname_f=_fn_f, fname_s=_fn_s, beginning=True, plot_det=False)
    fig_t = _bst.plot_escape_time(_fn_c, fname_s=_fn_s, transient=True, plot_det=False)
    fig_cv = _bst.plot_cv(_fn_c, _fn_f, _fn_s)
    #fig_cv2 = _bst.plot_cv(_fn_c, _fn_f, _fn_s, ytop=0.6)  # for closer look at CV

    if _save:
        fig.savefig(_prefix + "times" + ext, dpi=300, bbox_inches="tight")
        #fig_c.savefig(_prefix + "coll-escape" + ext, dpi=300, bbox_inches="tight")
        #fig_q.savefig(_prefix + "quiescent" + ext, dpi=300, bbox_inches="tight")
        #fig_b.savefig(_prefix + "beginning" + ext, dpi=300, bbox_inches="tight")
        #fig_t.savefig(_prefix + "transient" + ext, dpi=300, bbox_inches="tight")
        fig_cv.savefig(_prefix + "cv" + ext, dpi=300, bbox_inches="tight")
    # fmt: on


#%%
def sample_path(_nw, _save: bool, _prefix: str, _ext: str):
    for d_val in l_D:
        bst = bistable(
            nw=_nw, t_max=100, r=r, D=d_val, alpha=alpha, threshold=threshold
        )
        bst.init_no_activation()
        bst.simulate()
        fig = bst.plot_sample_path(top_k=1, y=bst.threshold)
        if _save:
            fig.savefig(
                _prefix + f"D{int(100 * d_val)}_sample-path" + _ext,
                dpi=300,
                bbox_inches="tight",
            )


#%%[markdown]
# ## Barabasi-Albert model

#%%
def setup_ba(_m: int, _seed: int):
    _nw = barabasi_albert(N=N, m=_m, seed=_seed)
    print("Number of nodes: ", _nw.num_nodes)
    print("Mean degree: ", _nw.mean_degree)

    _bst = bistable(
        nw=_nw, t_max=50, r=r, D=a_D, alpha=alpha, threshold=threshold
    )
    _bst.init_no_activation()

    return (_nw, _bst)


#%%[markdown]
# ### network 1

#%%
prefix = "ba1_"
nw, bst = setup_ba(3, 2023)

# %%
fn_f, fn_s, fn_c = get_filenames(
    bst, calculate, prefix, "data_th-0_8/230122-233608_ba1_none_"
)

#%%
fig_on_times(bst, fn_c, fn_f, fn_s, save, prefix)

#%%
sample_path(nw, save, prefix, ".png")

#%%[markdown]
# ### network 2

#%%
prefix = "ba2_"
nw, bst = setup_ba(3, 2778561)

#%%
fn_f, fn_s, fn_c = get_filenames(
    bst, calculate, prefix, "data_th-0_8/230123-002042_ba2_none_"
)

#%%
fig_on_times(bst, fn_c, fn_f, fn_s, save, prefix)

#%%
# sample_path(nw, False, prefix, ".png")

#%%[markdown]
# ### network 3

#%%
prefix = "ba3_"
nw, bst = setup_ba(3, 230124)

#%%
fn_f, fn_s, fn_c = get_filenames(
    bst, calculate, prefix, "data_th-0_8/230123-005624_ba3_none_"
)

#%%
fig_on_times(bst, fn_c, fn_f, fn_s, save, prefix)

#%%
# sample_path(nw, False, prefix, ".png")

#%%[markdown]
# ## Erdos-Renyi model

#%%
def setup_er(_k: int, _seed: int):
    _nw = erdos_renyi(N=N, k=_k, seed=_seed)
    print("Number of nodes:", _nw.num_nodes)
    print("Mean degree:", _nw.mean_degree)

    _bst = bistable(
        nw=_nw, t_max=50, r=r, D=a_D, alpha=alpha, threshold=threshold
    )
    _bst.init_no_activation()

    return (_nw, _bst)


#%%[markdown]
# ### network 4

#%%
prefix = "er1_"
nw, bst = setup_er(6, 1138656)

#%%
fn_f, fn_s, fn_c = get_filenames(
    bst, calculate, prefix, "data_th-0_8/230123-014500_er1_none_"
)

#%%
fig_on_times(bst, fn_c, fn_f, fn_s, save, prefix)

#%%
sample_path(nw, save, prefix, ".png")

#%%[markdown]
# ### network 5

#%%
prefix = "er2_"
nw, bst = setup_er(6, 12345678)

#%%
fn_f, fn_s, fn_c = get_filenames(
    bst, calculate, prefix, "data_th-0_8/230123-022947_er2_none_"
)

#%%
fig_on_times(bst, fn_c, fn_f, fn_s, save, prefix)

#%%
# sample_path(nw, False, prefix, ".png")

#%%[markdown]
# ### network 6

#%%
prefix = "er3_"
nw, bst = setup_er(6, 57577)

#%%
fn_f, fn_s, fn_c = get_filenames(
    bst, calculate, prefix, "data_th-0_8/230123-031501_er3_none_"
)

#%%
fig_on_times(bst, fn_c, fn_f, fn_s, save, prefix)

#%%
# sample_path(nw, False, prefix, ".png")

# %%

#%%[markdown]
# # Figure 4.3
# Chapter 4. Diffusion of enrolment behaviour

#%%
import numpy as np
from modules.network import two_node
from modules.simulation.bistable import two_node_model

#%%
# MODEL PARAMETERS
n = 250  # number of pairs
r = 0.1
alpha = 0.05

l_D = [0.001, 0.01, 0.06, 0.1]

# PARAMETERS FOR VISUALIZATION
ext: str = ".pdf"
save: bool = True

#%%
# generate network
nw = two_node(num_edges=250, seed=2022)

#%%
for d_val in l_D:
    tn = two_node_model(nw=nw, t_max=150, r=r, D=d_val, alpha=alpha)
    tn.init_no_activation()
    tn.simulate()
    fig = tn.plot_scatter(pairs_to_plot=[11], figw=5, figh=3)
    if save:
        fig.savefig(
            f"two-node_{int(d_val * 1000)}_scatter" + ext,
            dpi=300,
            bbox_inches="tight",
        )

# %%

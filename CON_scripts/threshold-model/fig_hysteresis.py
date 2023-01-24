#%%[markdown]
# # Figure 2.10
# Chapter 2. Threshold model for university enrolment

#%%
import numpy as np
from threshold_model import f
from matplotlib import pyplot as plt

#%%
def update(
    tau: float, p: float, theta: float, g: float, b: float, r: float
) -> tuple[float]:
    """
    Simulates the temporal evolution of the threshold model
    while gradually change theta.

    Parameters
    ----------
    tau : float
        time.
    p : float
    theta : float
    g : float
        gamma.
    b : float
        beta.
    r : float.
        rho.

    Returns
    -------
    p : float
    theta : float
    """
    dtheta = -0.003 if tau < 100 else 0.003
    newp = f(p, theta, g, b, r)
    newtheta = theta + dtheta
    return (newp, newtheta)


#%%
save: bool = False
ext: str = ".pdf"

#%%
# model parameters
r: float = 0.95  # rho
g: float = 15  # gamma
b: float = 25  # beta

# initial condition
p_init: float = 0.3
theta_init: float = 0.65

# calculate time series
tau_max: int = 200  #  maximum time
res_y: np.ndarray = np.zeros(
    shape=(2, tau_max + 1), dtype="f8"
)  # record trajectory
res_y[:, 0] = (p_init, theta_init)

for tau in range(tau_max):
    res_y[:, tau + 1] = update(tau, res_y[0, tau], res_y[1, tau], g, b, r)

#%%
# create figure
fig, ax = plt.subplots(figsize=(4, 2.5), dpi=300, layout="constrained")
ax.set_title(rf"$\gamma = {g}, \beta = {b}, \rho = {r}$")

s = ax.scatter(
    res_y[1, :],
    res_y[0, :],
    c=np.arange(tau_max + 1),
    cmap="turbo",
    s=10,
    marker="x",
    linewidth=0.5,
    alpha=0.8,
    zorder=9,
)

ax.set_xlim(0.3, 0.7)
ax.set_ylim(0, 1)
ax.set_xmargin(0)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("$p$")
fig.colorbar(s, label="time")

plt.show()

if save:
    fig.savefig("hysteresis" + ext, dpi=300, bbox_inches="tight")

# %%

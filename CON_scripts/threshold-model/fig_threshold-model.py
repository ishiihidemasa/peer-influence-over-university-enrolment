#%%[markdown]
# # Figures 2.5, 2.6, 2.7, 2.8, 2.9
# Chapter 2. Threshold model for university enrolment

#%%
import numpy as np
import threshold_model as tm

#%%
# if `save` is `True`, figures are saved in the current directory
save = False
ext = ".pdf"

#%%[markdown]
# ## Draw phase diagram

#%%
def set_param(if_vary: dict, t: float, g: float, b: float, r: float) -> tuple:
    num_param = 151
    _t = np.linspace(0.25, 0.75, num_param) if if_vary["t"] else t
    _g = np.linspace(5, 30, num_param) if if_vary["g"] else g
    _b = np.linspace(5, 30, num_param) if if_vary["b"] else b
    _r = np.linspace(0.5, 1, num_param) if if_vary["r"] else r
    return (_t, _g, _b, _r)


#%%
num_p = 5000
num_p2 = 50000
ms = 0.6

#%% [markdown]
# ### Phase diagram with theta = 1 / 2 and rho = 1

# %%
if_vary = {"t": True, "g": True, "b": False, "r": False}
t, g, b, r = set_param(if_vary, t=0.5, g=15, b=20, r=1)

if_fixed, l_phase = tm.get_phase(num_p=num_p, t=t, g=g, b=b, r=r)
tm.plot_pd(
    t=t,
    g=g,
    b=b,
    r=r,
    if_fixed=if_fixed,
    l_phase=l_phase,
    num_p=num_p,
    ms=ms,
    save=save,
    ext=ext,
    date="",
)

# %%
if_vary = {"t": True, "g": False, "b": True, "r": False}
t, g, b, r = set_param(if_vary, t=0.5, g=15, b=20, r=1)

if_fixed, l_phase = tm.get_phase(num_p=num_p, t=t, g=g, b=b, r=r)
tm.plot_pd(
    t=t,
    g=g,
    b=b,
    r=r,
    if_fixed=if_fixed,
    l_phase=l_phase,
    num_p=num_p,
    ms=ms,
    save=save,
    ext=ext,
    date="",
)

# %%
if_vary = {"t": True, "g": False, "b": False, "r": True}
t, g, b, r = set_param(if_vary, t=0.5, g=15, b=20, r=1)

if_fixed, l_phase = tm.get_phase(num_p=num_p, t=t, g=g, b=b, r=r)
tm.plot_pd(
    t=t,
    g=g,
    b=b,
    r=r,
    if_fixed=if_fixed,
    l_phase=l_phase,
    num_p=num_p,
    ms=ms,
    save=save,
    ext=ext,
    date="",
)
# %%
if_vary = {"t": False, "g": True, "b": True, "r": False}
t, g, b, r = set_param(if_vary, t=0.5, g=15, b=20, r=1)

if_fixed, l_phase = tm.get_phase(num_p=num_p, t=t, g=g, b=b, r=r)
tm.plot_pd(
    t=t,
    g=g,
    b=b,
    r=r,
    if_fixed=if_fixed,
    l_phase=l_phase,
    num_p=num_p,
    ms=ms,
    save=save,
    ext=ext,
    date="",
)

# %%
if_vary = {"t": False, "g": True, "b": False, "r": True}
t, g, b, r = set_param(if_vary, t=0.5, g=15, b=20, r=1)

if_fixed, l_phase = tm.get_phase(num_p=num_p, t=t, g=g, b=b, r=r)
tm.plot_pd(
    t=t,
    g=g,
    b=b,
    r=r,
    if_fixed=if_fixed,
    l_phase=l_phase,
    num_p=num_p,
    ms=ms,
    save=save,
    ext=ext,
    date="",
)

# %%
if_vary = {"t": False, "g": False, "b": True, "r": True}
t, g, b, r = set_param(if_vary, t=0.5, g=15, b=20, r=1)

if_fixed, l_phase = tm.get_phase(num_p=num_p, t=t, g=g, b=b, r=r)
tm.plot_pd(
    t=t,
    g=g,
    b=b,
    r=r,
    if_fixed=if_fixed,
    l_phase=l_phase,
    num_p=num_p,
    ms=ms,
    save=save,
    ext=ext,
    date="",
)

# %%[markdown]
# ### Phase diagram: beta and gamma

# %%
if_vary = {"t": False, "g": True, "b": True, "r": False}
t, g, b, r = set_param(if_vary, t=0.45, g=20, b=15, r=1)

if_fixed, l_phase = tm.get_phase(num_p=num_p2, t=t, g=g, b=b, r=r)
tm.plot_pd(
    t=t,
    g=g,
    b=b,
    r=r,
    if_fixed=if_fixed,
    l_phase=l_phase,
    num_p=num_p,
    ms=ms,
    save=save,
    ext=ext,
    date="t045-r1",
)

# %%
if_vary = {"t": False, "g": True, "b": True, "r": False}
t, g, b, r = set_param(if_vary, t=0.45, g=20, b=15, r=0.9)

if_fixed, l_phase = tm.get_phase(num_p=num_p, t=t, g=g, b=b, r=r)
tm.plot_pd(
    t=t,
    g=g,
    b=b,
    r=r,
    if_fixed=if_fixed,
    l_phase=l_phase,
    num_p=num_p2,
    ms=ms,
    save=save,
    ext=ext,
    date="t045-r09",
)

# %%[markdown]
# ## Plot the map f(p)

#%%
# phase S
(t, g, b, r) = (0.6, 15, 20, 0.9)
a_p, a_f, a_fp, a_dffp, a_stable = tm.find_fp(t, g, b, r, num_p=5000)
tm.plot_f(t, g, b, r, a_p, a_f, a_fp, a_stable, date="S", save=save, ext=ext)

#%%
# phase SU
(t, g, b, r) = (0.7, 15, 13, 1)
a_p, a_f, a_fp, a_dffp, a_stable = tm.find_fp(t, g, b, r, num_p=5000)
tm.plot_f(t, g, b, r, a_p, a_f, a_fp, a_stable, date="SU", save=save, ext=ext)

# %%
# phase US
(t, g, b, r) = (0.3, 15, 17, 1)
a_p, a_f, a_fp, a_dffp, a_stable = tm.find_fp(t, g, b, r, num_p=5000)
tm.plot_f(t, g, b, r, a_p, a_f, a_fp, a_stable, date="US", save=save, ext=ext)

#%%
# phase SUS
(t, g, b, r) = (0.45, 10, 25, 0.9)
a_p, a_f, a_fp, a_dffp, a_stable = tm.find_fp(t, g, b, r, num_p=5000)
tm.plot_f(t, g, b, r, a_p, a_f, a_fp, a_stable, date="SUS", save=save, ext=ext)

#%%
# phase USU
(t, g, b, r) = (0.55, 25, 20, 1)
a_p, a_f, a_fp, a_dffp, a_stable = tm.find_fp(t, g, b, r, num_p=5000)
tm.plot_f(t, g, b, r, a_p, a_f, a_fp, a_stable, date="USU", save=save, ext=ext)

# %%[markdown]
# ## Plot bifurcation diagram

#%%
num = 120
umarker = "s"
ukwargs = {"fillstyle": "none", "alpha": 0.5}

#%%[markdown]
# ### parameter: theta

#%%
t = np.linspace(0.25, 0.75, num, dtype="f8")
#%%
# with gamma < beta
g = 15
b = 25
r = np.array([0.5, 0.9, 1], dtype="f8")
tm.plot_bd(
    const1=g,
    const2=b,
    fix=r,
    b=t,
    param_role=["b", "c1", "c2", "f"],
    num_p=5000,
    legend_loc="center left",
    bbox_to_anchor=(0, 0.5),
    umarker=umarker,
    **ukwargs,
    save=save,
    ext=ext,
    date="large-b",
)
# %%
# with beta < gamma
g = 25
b = 15
r = np.array([0.5, 0.9, 1], dtype="f8")
tm.plot_bd(
    const1=g,
    const2=b,
    fix=r,
    b=t,
    param_role=["b", "c1", "c2", "f"],
    num_p=5000,
    legend_loc="center right",
    bbox_to_anchor=(1, 0.7),
    umarker=umarker,
    **ukwargs,
    save=save,
    ext=ext,
    date="large-g",
)

#%%[markdown]
# ### parameter: rho

#%%
r = np.linspace(0.5, 1, num, dtype="f8")
#%%
# with gamma < beta
g = 15
b = 25
t = np.array([0.45, 0.5, 0.8], dtype="f8")
tm.plot_bd(
    const1=g,
    const2=b,
    fix=t,
    b=r,
    param_role=["f", "c1", "c2", "b"],
    num_p=5000,
    legend_loc="upper left",
    bbox_to_anchor=(0.05, 1.02),
    umarker=umarker,
    **ukwargs,
    save=save,
    ext=ext,
    date="large-b",
)

#%%
# with beta < gamma
g = 25
b = 15
t = np.array([0.45, 0.5, 0.8], dtype="f8")
tm.plot_bd(
    const1=g,
    const2=b,
    fix=t,
    b=r,
    param_role=["f", "c1", "c2", "b"],
    num_p=5000,
    legend_loc="upper left",
    bbox_to_anchor=(0.05, 1.02),
    umarker=umarker,
    **ukwargs,
    save=save,
    ext=ext,
    date="large-g",
)

# %%

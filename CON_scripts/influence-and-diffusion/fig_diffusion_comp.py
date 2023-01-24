#%%[markdown]
# # Figure 4.7
# Chapter 4. Diffusion of enrolment behaviour


#%%
import numpy as np
from matplotlib import pyplot as plt

#%%
a_D = np.linspace(0, 0.11, 12)  # for calculating escape times

# PARAMETERS FOR VISUALIZATION
ext: str = ".pdf"  # extension
save: bool = False  # if True, export figures

#%%
def compare_duration(
    base_er: str,
    base_ba: str,
    a_D: np.ndarray,
    period: str = "b",
    figw: float = 3,
    figh: float = 2.5,
) -> plt.Figure:
    l_data = []
    base = (base_er, base_ba)
    for i in range(2):
        if period == "q":
            a_data = np.loadtxt(base[i] + "first.csv", delimiter=",")
        elif period == "b":
            a_first = np.loadtxt(base[i] + "first.csv", delimiter=",")
            a_start = np.loadtxt(base[i] + "start.csv", delimiter=",")
            a_data = a_start - a_first
        elif period == "t":
            a_start = np.loadtxt(base[i] + "start.csv", delimiter=",")
            a_coll = np.loadtxt(base[i] + "collective.csv", delimiter=",")
            a_data = a_coll - a_start
        else:
            raise ValueError
        l_data.append(a_data)

    colors = ("tab:red", "tab:blue")
    markers = ("s", "x")
    labels = ("ER", "BA")

    fig, ax = plt.subplots(figsize=(figw, figh), layout="constrained", dpi=300)

    for i in range(2):
        data = l_data[i]
        mean = data.mean(axis=1)
        std = data.std(axis=1)

        ax.errorbar(
            x=a_D,
            y=mean,
            yerr=std,
            marker=markers[i],
            ls="none",
            fillstyle="none",
            c=colors[i],
            alpha=0.8,
            elinewidth=0.5,
            capsize=2,
            label=labels[i],
        )

    ax.set_xlabel("$D$")
    ax.grid(axis="y", c="lightgray", zorder=1, lw=0.5)

    if period == "q":
        ax.set_ylabel("Duration of quiescent period")
    elif period == "b":
        ax.set_ylabel("Duration of beginning period")
    elif period == "t":
        ax.set_ylabel("Duration of transient period")

    ax.legend()
    plt.show()

    return fig


#%%
def create_figs(
    _er: str,
    _ba: str,
    _a_D: np.ndarray,
    _save: bool,
    _prefix: str = "",
):
    fig_q = compare_duration(_er, _ba, _a_D, "q")
    fig_b = compare_duration(_er, _ba, _a_D, "b")
    fig_t = compare_duration(_er, _ba, _a_D, "t")
    if _save:
        fig_q.savefig(_prefix + "comp-q" + ext, dpi=300, bbox_inches="tight")
        fig_b.savefig(_prefix + "comp-b" + ext, dpi=300, bbox_inches="tight")
        fig_t.savefig(_prefix + "comp-t" + ext, dpi=300, bbox_inches="tight")


#%%[markdown]
# ## Create figures

#%%
pref = "1_"
er1 = "data_th-0_8/230123-014500_er1_none_"
ba1 = "data_th-0_8/230122-233608_ba1_none_"
create_figs(er1, ba1, a_D, save, pref)

#%%
pref = "2_"
er2 = "data_th-0_8/230123-022947_er2_none_"
ba2 = "data_th-0_8/230123-002042_ba2_none_"
create_figs(er2, ba2, a_D, save, pref)

#%%
pref = "3_"
er3 = "data_th-0_8/230123-031501_er3_none_"
ba3 = "data_th-0_8/230123-005624_ba3_none_"
create_figs(er3, ba3, a_D, save, pref)


# %%

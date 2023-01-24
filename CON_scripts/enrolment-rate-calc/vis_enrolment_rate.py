#%%[markdown]
# # Figure 1.1
# ## Visuallization of university enrolment rate by prefecture
# A figure is created based on the data stored in csv files.

#%%
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt

#%%
# load data
df_enrolment_rate = pd.read_csv(
    "./output/enrolment_rate.csv", index_col=0, encoding="shift-jis"
)

#%%
# reassign columns (translate from Japanese to English)
d_col = {
    "全国": "Japan",
    "北海道": "Hokkaido",
    "青森": "Aomori",
    "岩手": "Iwate",
    "宮城": "Miyagi",
    "秋田": "Akita",
    "山形": "Yamagata",
    "福島": "Fukushima",
    "茨城": "Ibaraki",
    "栃木": "Tochigi",
    "群馬": "Gunma",
    "埼玉": "Saitama",
    "千葉": "Chiba",
    "東京": "Tokyo",
    "神奈川": "Kanagawa",
    "新潟": "Niigata",
    "富山": "Toyama",
    "石川": "Ishikawa",
    "福井": "Fukui",
    "山梨": "Yamanashi",
    "長野": "Nagano",
    "岐阜": "Gifu",
    "静岡": "Shizuoka",
    "愛知": "Aichi",
    "三重": "Mie",
    "滋賀": "Shiga",
    "京都": "Kyoto",
    "大阪": "Osaka",
    "兵庫": "Hyogo",
    "奈良": "Nara",
    "和歌山": "Wakayama",
    "鳥取": "Tottori",
    "島根": "Shimane",
    "岡山": "Okayama",
    "広島": "Hiroshima",
    "山口": "Yamaguchi",
    "徳島": "Tokushima",
    "香川": "Kagawa",
    "愛媛": "Ehime",
    "高知": "Kochi",
    "福岡": "Fukuoka",
    "佐賀": "Saga",
    "長崎": "Nagasaki",
    "熊本": "Kumamoto",
    "大分": "Oita",
    "宮崎": "Miyazaki",
    "鹿児島": "Kagoshima",
    "沖縄": "Okinawa",
}
df_enrolment_rate.rename(columns=d_col, inplace=True)

#%%
def setup_axes(ax, title=None):
    ax.set_xlabel("Year")
    ax.set_ylabel("Enrolment Rate")
    ax.set_ylim((0.3, 0.8))
    ax.legend(
        loc="center left", bbox_to_anchor=(1.05, 0.5), ncols=2, title=title
    )
    return


#%%
def plot_by_region(df):
    fig, ax = plt.subplots(
        nrows=3, figsize=(7, 8), constrained_layout=True, sharex=True
    )
    fig.suptitle("University Enrolment Rate by Prefecture")
    ax1, ax2, ax3 = ax

    df.iloc[:, 1:15].plot(ax=ax1, marker="x", colormap="tab20")
    df.iloc[:, 15:32].plot(ax=ax2, marker="x", colormap="tab20")
    df.iloc[:, 32:].plot(ax=ax3, marker="x", colormap="tab20")

    for a, t in zip(
        ax,
        [
            "Hokkaido, Tohoku, Kanto",
            "Chubu, Kansai, Chugoku",
            "Shikoku, Kyushu & Okinawa",
        ],
    ):
        df["Japan"].plot(ax=a, marker="x", color="k", linestyle="dashed")
        setup_axes(a, t)

    plt.show()
    return fig


#%%
def plot_by_rank(df):
    # df_pref.iloc[:, :9] is the above-average group
    df_pref = df.iloc[:, 1:].sort_values(
        by=2021, axis="columns", ascending=False
    )  # sort by rates in 2021
    fig, ax = plt.subplots(
        nrows=3, figsize=(7, 8), constrained_layout=True, sharex=True
    )
    fig.suptitle("University Enrolment Rate by Prefecture")
    ax1, ax2, ax3 = ax

    df_pref.iloc[:, :16].plot(ax=ax1, marker="x", colormap="tab20")
    df_pref.iloc[:, 16:32].plot(ax=ax2, marker="x", colormap="tab20")
    df_pref.iloc[:, 32:].plot(ax=ax3, marker="x", colormap="tab20")

    for a, t in zip(ax, ["Rank 1 - 16", "Rank 17 - 32", "Rank 33 - 47"]):
        df["Japan"].plot(ax=a, marker="x", color="k", linestyle="dashed")
        setup_axes(a, t)

    plt.show()
    return fig


# %%
# plot the data
# fig = plot_by_region(df_enrolment_rate)
fig = plot_by_rank(df_enrolment_rate)

# %%
fig.savefig("./output/enrolment_rate.png", bbox_inches="tight", dpi=300)


# %%

#%%[markdown]
# # Figure 1.1
# ## Calculation of university enrolment rate by prefecture
# Calculation results will be saved in csv files in `output` directory.

#%%
import pandas as pd
from IPython.display import display


#%%
def load_csv(filename: str):
    df = pd.read_csv(
        "./data/" + filename + ".csv", index_col=1, skiprows=1
    ).iloc[
        :, 1:
    ]  # exclude "年度" column
    df.replace("－", 0, inplace=True)  # replace "－" with 0
    df = df.astype("i4")  # numpy int 32 bit
    return df


#%%
# obtain the number of undergraduate enrolees
df_enrolee = load_csv("enrolee_undergraduate").iloc[:, :-1]  # exclude "その他"

#%%
# display(df_enrolee.head())

#%%
# obtain the size of the 18-year-old population in 3 years (not current)
df_grad_ces = load_csv("grad_ces")  # compulsory education schools
df_grad_lss = load_csv("grad_lss")  # lower secondary schools
df_grad_ses = load_csv(
    "grad_ses_lower"
)  # secondary education shcools (lower div.)

df_candidate = (
    df_grad_ces.add(df_grad_lss, fill_value=0)
    .add(df_grad_ses, fill_value=0)
    .astype("i4")
)

#%%
# display(df_candidate.head())

#%%
# obtain the size of the current 18-year-old population (with 3-year lag)
a_id_original = df_candidate.index.to_numpy(dtype="i4")
# df_candidate[2020] stands for the size of the 18-year-old population in 2023
a_id_lagged = a_id_original + 3
df_candidate.index = a_id_lagged

#%%
# calculate university enrolment rate
df_enrolment_rate = (df_enrolee / df_candidate).dropna(axis=0, how="all")

#%%
# export dataframe as a csv file
df_enrolment_rate.to_csv("./output/enrolment_rate.csv", encoding="shift-jis")

#%%
# display(df_enrolment_rate.tail())


# %%

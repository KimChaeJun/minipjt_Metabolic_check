import pandas as pd
import numpy as np

df23 = pd.read_sas("./HN23_ALL/hn23_all.sas7bdat")
df24 = pd.read_sas("./HN24_ALL/hn24_all.sas7bdat")

df23["year"] = 2023
df24["year"] = 2024

df = pd.concat([df23, df24], ignore_index=True)

cols = [
    "year",
    "sex",
    "age",
    "HE_wc",
    "HE_sbp",
    "HE_dbp",
    "HE_glu",
    "HE_TG",
    "HE_HDL_st2",
    "HE_BMI",
    "sm_presnt",
    "dr_month",
    "pa_aerobic",
]

df = df[cols].copy()

# 숫자형 변환
for col in cols:
    if col != "year":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 대사증후군 5가지 기준
df["abdominal_obesity"] = (
    ((df["sex"] == 1) & (df["HE_wc"] >= 90)) |
    ((df["sex"] == 2) & (df["HE_wc"] >= 85))
).astype(int)

df["high_bp"] = (
    (df["HE_sbp"] >= 130) |
    (df["HE_dbp"] >= 85)
).astype(int)

df["high_glucose"] = (
    df["HE_glu"] >= 100
).astype(int)

df["high_tg"] = (
    df["HE_TG"] >= 150
).astype(int)

df["low_hdl"] = (
    ((df["sex"] == 1) & (df["HE_HDL_st2"] < 40)) |
    ((df["sex"] == 2) & (df["HE_HDL_st2"] < 50))
).astype(int)

# 5개 중 3개 이상이면 대사증후군
criteria_cols = [
    "abdominal_obesity",
    "high_bp",
    "high_glucose",
    "high_tg",
    "low_hdl",
]

df["metabolic_syndrome_score"] = df[criteria_cols].sum(axis=1)
df["metabolic_syndrome"] = (df["metabolic_syndrome_score"] >= 3).astype(int)

# 결측치 제거
df = df.dropna()

print(df.shape)
print(df["metabolic_syndrome"].value_counts())
print(df["metabolic_syndrome"].value_counts(normalize=True))
print(df.head())

eda_cols = [
    "age",
    "HE_BMI",
    "HE_wc",
    "HE_sbp",
    "HE_dbp",
    "HE_glu",
    "HE_TG",
    "HE_HDL_st2"
]
print("==============================================================================================")
print(df.groupby("metabolic_syndrome")[eda_cols].mean().round(2))
print(df.isnull().sum().sort_values(ascending=False).head(20))


print("==============================================================================================")
print(df["sex"].value_counts())
print(df["sm_presnt"].value_counts())
print(df["dr_month"].value_counts())
print(df["pa_aerobic"].value_counts())

print("==============================================================================================")
res_sm_meta = pd.crosstab(
    df["sm_presnt"],
    df["metabolic_syndrome"],
    normalize="index"
)


res_pa_meta = pd.crosstab(
    df["pa_aerobic"],
    df["metabolic_syndrome"],
    normalize="index"
)
print(res_sm_meta)
print(res_pa_meta)
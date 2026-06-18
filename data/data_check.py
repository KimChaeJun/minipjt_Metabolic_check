import pandas as pd

df23 = pd.read_sas("HN23_ALL/hn23_all.sas7bdat")
df24 = pd.read_sas("HN24_ALL/hn24_all.sas7bdat")

pd.Series(df23.columns).to_csv(
    "columns_23.csv",
    index=False
)
pd.Series(df24.columns).to_csv(
    "columns_24.csv",
    index=False
)

cols = pd.read_csv("columns_24.csv")["0"]

keywords = [
    "sm",      # 흡연
    "dr",      # 음주
    "pa",      # 신체활동
    "ex",      # 운동
    "TG",
    "tr",
    "HDL",
    "DM",
    "glu"
]

for kw in keywords:
    print(f"\n===== {kw} =====")
    print([c for c in cols if kw.lower() in c.lower()][:100])
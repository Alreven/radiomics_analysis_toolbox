import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


CSV_FILE = "cv_fbg_ten_68rdmx.csv"

COL_FEATURE = "feature"
COL_ICC = "CV_global"

SELECTION_COLS = ["binWidth", "fbn", "minmax", "n4itk", "normalizeScale", "zscore"]


CONDITION_1 = {
    "binWidth": 1,
    "fbn": 0,
    "minmax": 0,
    "n4itk": 0,
    "normalizeScale": 0,
    "zscore": 0,
}

CONDITION_2 = {
    "binWidth": 1,
    "fbn": 0,
    "minmax": 0,
    "n4itk": 1,
    "normalizeScale": 0,
    "zscore": 0,
}



def apply_filters(df, filters):
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        if col not in df.columns:
            raise ValueError(f"Colonne absente du CSV : {col}")
        mask &= (df[col] == val)
    return df.loc[mask].copy()



df = pd.read_csv(CSV_FILE)


required_cols = [COL_FEATURE, COL_ICC] + SELECTION_COLS
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Colonnes manquantes dans le CSV : {missing_cols}")


df1 = apply_filters(df, CONDITION_1)
df2 = apply_filters(df, CONDITION_2)

if df1.empty:
    raise ValueError("La CONDITION_1 ne retourne aucune ligne.")
if df2.empty:
    raise ValueError("La CONDITION_2 ne retourne aucune ligne.")


if df1[COL_FEATURE].duplicated().any():
    dup = df1.loc[df1[COL_FEATURE].duplicated(), COL_FEATURE].unique()
    raise ValueError(
        f"Certaines features sont dupliquées dans CONDITION_1 : {list(dup)}"
    )

if df2[COL_FEATURE].duplicated().any():
    dup = df2.loc[df2[COL_FEATURE].duplicated(), COL_FEATURE].unique()
    raise ValueError(
        f"Certaines features sont dupliquées dans CONDITION_2 : {list(dup)}"
    )


merged = pd.merge(
    df1[[COL_FEATURE, COL_ICC]].rename(columns={COL_ICC: "icc_1"}),
    df2[[COL_FEATURE, COL_ICC]].rename(columns={COL_ICC: "icc_2"}),
    on=COL_FEATURE,
    how="inner"
)


merged = merged.dropna(subset=["icc_1", "icc_2"])

if merged.empty:
    raise ValueError(
        "Aucune feature commune avec des valeurs ICC non-NaN entre les deux conditions."
    )


icc_1 = merged["icc_1"].to_numpy()
icc_2 = merged["icc_2"].to_numpy()

print("Nombre de features utilisées pour le test :", len(icc_1))

print("\nCondition 1 :")
for k, v in CONDITION_1.items():
    print(f"   {k} = {v}")

print("\nCondition 2 :")
for k, v in CONDITION_2.items():
    print(f"   {k} = {v}")


stat, p_value = wilcoxon(icc_2, icc_1, alternative="less")

print("\nRésumé des CV :")
print("Condition 1 :")
print("   moyenne :", np.mean(icc_1))
print("   médiane :", np.median(icc_1))

print("Condition 2 :")
print("   moyenne :", np.mean(icc_2))
print("   médiane :", np.median(icc_2))

print("\nTest de Wilcoxon pairé (unilatéral : cv_2 < cv_1)")
print("   statistique W :", stat)
print("   p-value        :", p_value)


diff_median = np.median(icc_2 - icc_1)

print("\nConclusion (alpha = 0.05) :")

if p_value < 0.05:
    print("   Les CV de la condition 2 sont significativement PLUS PETITS")
    print("   que ceux de la condition 1 (amélioration significative).")
else:
    print("   Aucune amélioration significative entre les deux conditions.")

print("\nDifférence médiane (icc_2 - icc_1) :", diff_median)
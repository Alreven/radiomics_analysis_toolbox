import pandas as pd
import numpy as np

def coefficient_variation(values: np.ndarray, ddof: int = 1) -> float:
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) < 2:
        return np.nan

    mean_x = np.mean(x)
    if mean_x == 0:
        return np.nan

    std_x = np.std(x, ddof=ddof)
    return std_x / abs(mean_x)

CSV_IN = "radiomics_features.csv"
CSV_OUT = "cv_results.csv"

RATER_COLS = ["date", "acquisition"]
COMBO_COLS = ["binWidth", "fbn", "minmax", "n4itk", "normalizeScale", "zscore"]

MASK_PREFIXES = ["pvp40", "fbg", "fat", "ext"]

df = pd.read_csv(CSV_IN)

required_cols = RATER_COLS + COMBO_COLS + ["mask"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes absentes du CSV : {missing}")

df["rater"] = df[RATER_COLS].astype(str).agg("_".join, axis=1)

def extract_prefix(mask_name: str):
    for prefix in MASK_PREFIXES:
        if str(mask_name).startswith(prefix):
            return prefix
    return np.nan

df["mask_group"] = df["mask"].apply(extract_prefix)

df = df.dropna(subset=["mask_group"])

feature_cols = [c for c in df.columns if c.startswith("original_")]
if not feature_cols:
    raise ValueError("Aucune colonne de feature commençant par 'original_' n'a été trouvée.")

results = []

for combo_values, df_group in df.groupby(COMBO_COLS, dropna=False):
    combo_dict = dict(zip(COMBO_COLS, combo_values))

    print("\n======================================")
    print("Combinaison en cours :")
    for k, v in combo_dict.items():
        print(f"  {k} = {v}")
    print("======================================")

    for mask_group, df_mask_group in df_group.groupby("mask_group"):
        for feat in feature_cols:
            mat = df_mask_group.pivot(index="mask", columns="rater", values=feat)

            mat = mat.dropna(axis=0, how="any")

            n_subjects, n_raters = mat.shape

            if n_subjects < 1 or n_raters < 2:
                cv_global = np.nan
            else:
                all_values = mat.to_numpy().ravel()
                cv_global = coefficient_variation(all_values)

            row = {
                **combo_dict,
                "mask_group": mask_group,
                "feature": feat,
                "CV_global_percent": cv_global,
                "n_subjects": n_subjects,
                "n_raters": n_raters,
                "n_values": n_subjects * n_raters,
            }

            results.append(row)

cv_df = pd.DataFrame(results)
cv_df.to_csv(CSV_OUT, index=False)

print("\nRésultats sauvegardés dans :", CSV_OUT)
print("\nAperçu :\n")
print(cv_df.head())
import pandas as pd
import numpy as np

def icc2_1(data_matrix: np.ndarray) -> float:
    Y = np.asarray(data_matrix, dtype=float)
    n, k = Y.shape

    mean_subject = Y.mean(axis=1, keepdims=True)
    mean_rater = Y.mean(axis=0, keepdims=True)
    grand_mean = Y.mean()

    SSR = k * np.sum((mean_subject - grand_mean) ** 2)
    SSC = n * np.sum((mean_rater - grand_mean) ** 2)
    SSE = np.sum((Y - mean_subject - mean_rater + grand_mean) ** 2)

    dfR = n - 1
    dfC = k - 1
    dfE = dfR * dfC

    if dfR <= 0 or dfC <= 0 or dfE <= 0:
        return np.nan

    MSR = SSR / dfR
    MSC = SSC / dfC
    MSE = SSE / dfE

    denom = MSR + (k - 1) * MSE + (k * (MSC - MSE) / n)
    if denom == 0:
        return np.nan

    icc = (MSR - MSE) / denom
    return icc

CSV_IN = "radiomics_features.csv"
CSV_OUT = "icc_results.csv"

RATER_COLS = ["date", "acquisition"]

COMBO_COLS = ["binWidth", "fbn", "minmax", "n4itk", "normalizeScale", "zscore"]

df = pd.read_csv(CSV_IN)

required_cols = RATER_COLS + COMBO_COLS + ["mask"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes absentes du CSV : {missing}")

df["rater"] = df[RATER_COLS].astype(str).agg("_".join, axis=1)

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

    for feat in feature_cols:
        mat = df_group.pivot(index="mask", columns="rater", values=feat)

        mat = mat.dropna(axis=0, how="any")

        n_subjects, n_raters = mat.shape

        if n_subjects < 2 or n_raters < 2:
            icc_val = np.nan
        else:
            icc_val = icc2_1(mat.values)

        row = {
            **combo_dict,
            "feature": feat,
            "ICC2_1": icc_val,
            "n_subjects": n_subjects,
            "n_raters": n_raters,
        }
        results.append(row)

icc_df = pd.DataFrame(results)

icc_df.to_csv(CSV_OUT, index=False)

print("\nRésultats sauvegardés dans :", CSV_OUT)
print("\nAperçu :\n")
print(icc_df.head())
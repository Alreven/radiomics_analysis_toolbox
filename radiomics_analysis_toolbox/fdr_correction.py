import numpy as np
from statsmodels.stats.multitest import multipletests

# Liste des p-values (converties en notation scientifique Python)
p_values = [
    9e-1, 2e-7, 9e-1, 9e-1, 9e-1, 4e-5, 9e-1, 9e-1, 9e-1, 3e-4, 9e-1, 9e-1,
    9e-1, 1e-6, 9e-1, 2e-1, 9e-1, 2e-7, 6e-1, 1e-3, 9e-1, 2e-10, 4e-1, 4e-8,
    9e-1, 1e-10, 2e-2, 9e-8, 3e-2, 9e-2, 8e-7, 2e-1, 9e-1, 6e-2, 9e-2, 4e-3,
    1e-1, 9e-1, 7e-2, 1e-1, 9e-3, 3e-1, 9e-1, 6e-1, 9e-1, 2e-10, 2e-2, 9e-1,
    1e-1, 4e-1, 5e-10, 1e-5, 9e-1, 4e-3, 3e-1, 3e-9, 5e-7, 9e-1, 2e-2, 4e-1,
    3e-9, 2e-7, 9e-1
]

p_values = np.array(p_values)

# Application FDR (Benjamini-Hochberg)
reject, pvals_corrected, _, _ = multipletests(
    p_values,
    alpha=0.05,
    method='fdr_bh'
)

# Affichage des résultats
print("Index | p-value brute | p-value FDR | significatif")
print("-" * 55)

for i, (p_raw, p_corr, rej) in enumerate(zip(p_values, pvals_corrected, reject)):
    print(f"{i:3d} | {p_raw:.2e}      | {p_corr:.2e}    | {rej}")
"""
Synthetic test data in the original DL56Bandwise .npz layout (95 RBP columns + 20 CPTE columns).

Class sizes match ds004504 (36 AD / 29 CN / 23 FTD). A real signal is planted only in the
alpha band of the RBP block (band-major columns 38-56): AD lower, FTD higher than CN.
Every other column is pure noise, which makes it easy to check that a pipeline finds the
signal and stays at chance elsewhere (used for the bias demonstration in REPORT.md §14.1).

Usage:
    python scripts/make_synthetic_data.py 1 synthetic_1row.npz    # one row per subject
    python scripts/make_synthetic_data.py 3 synthetic_3rows.npz   # three "epochs" per subject
"""
import sys

import numpy as np

rng = np.random.default_rng(0)
rows_per_subject, out = int(sys.argv[1]), sys.argv[2]
counts = {0: 36, 1: 29, 2: 23}  # AD, CN, FTD
X, y, groups = [], [], []
for label, n in counts.items():
    for s in range(n):
        subject_shift = rng.normal(0, 0.3, 115)
        for _ in range(rows_per_subject):
            x = rng.normal(0, 1, 115) + subject_shift
            x[38:57] += {0: -1.0, 1: 0.0, 2: 1.0}[label]
            X.append(x)
            y.append(label)
            groups.append(f"sub-{label}{s:02d}")
X = np.array(X)
X[rng.random(X.shape) < 0.01] = np.nan  # a few missing values
X[0, 3] = np.inf
np.savez(out, features=X, labels=np.array(y), subject_ids=np.array(groups))
print(f"Saved {out}: {X.shape[0]} rows x {X.shape[1]} columns")

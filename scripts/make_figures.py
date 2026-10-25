"""
Regenerate every figure in figures/ from the saved results.

    python scripts/make_figures.py

Reads results/<run>/results.csv and subject_predictions.csv. The peak-alpha figure also needs
features_ds004504.npz (created by `eeg_bandwise_classification.py extract`). It is skipped if
that file is missing.

Colours: the dataviz reference palette (blue, orange, aqua, yellow), validated for
colour-vision deficiency. Values are printed on the bars because aqua and yellow fall
below 3:1 contrast on the light surface.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow (fixed order)
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

TASK_LABELS = {"AD_CN": "AD vs CN", "FTD_CN": "FTD vs CN", "AD_FTD": "AD vs FTD"}
TASK_ORDER = ["AD_CN", "FTD_CN", "AD_FTD"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150,
})


def _subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")


def _save(fig, name):
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def _load(run):
    return pd.read_csv(RESULTS / run / "results.csv").set_index("task")


# --- 1. Results vs literature ----------------------------------------------------

def fig_results_vs_literature():
    nested, loso = _load("headline_auto"), _load("loso_auto")
    series = [
        ("This work: nested 10-fold × 3", [nested.loc[t, "balanced_accuracy"] for t in TASK_ORDER],
         [nested.loc[t, "balanced_accuracy_sd"] for t in TASK_ORDER]),
        ("This work: leave-one-subject-out", [loso.loc[t, "balanced_accuracy"] for t in TASK_ORDER], None),
        ("Rigorous studies, mean (AHEPA review)", [0.821, 0.752, 0.714], [0.069, 0.076, 0.020]),
        ("Leakage-safe benchmark (arXiv 2609.22092)", [0.877, 0.827, 0.780], None),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    width, gap = 0.19, 0.012
    x = np.arange(len(TASK_ORDER))
    for i, (label, vals, err) in enumerate(series):
        pos = x + (i - 1.5) * (width + gap)
        ax.bar(pos, vals, width, color=SERIES[i], label=label, zorder=2)
        if err is not None:
            ax.errorbar(pos, vals, yerr=err, fmt="none", ecolor=INK_2, elinewidth=1, capsize=3, zorder=3)
        for p, v in zip(pos, vals):
            ax.text(p, 0.405, f"{v * 100:.0f}", ha="center", va="bottom", fontsize=9,
                    color="white" if i in (0, 1) else INK, fontweight="bold", zorder=4)
    ax.axhline(0.5, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(x[-1] + 0.48, 0.505, "chance", color=INK_2, fontsize=9, ha="right", va="bottom")
    ax.set_xticks(x, [TASK_LABELS[t] for t in TASK_ORDER], fontsize=11, color=INK)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks(np.arange(0.4, 1.01, 0.1), [f"{v:.0%}" for v in np.arange(0.4, 1.01, 0.1)])
    ax.set_ylabel("Balanced accuracy (subject level)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Honest results sit inside the published range for rigorous studies", pad=26)
    _subtitle(ax, "ds004504, 88 subjects. Error bars: SD over CV repeats (this work) or across studies (review).")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9.5)
    _save(fig, "results_vs_literature.png")


# --- 2. Feature families ---------------------------------------------------------

def fig_feature_families():
    df = pd.read_csv(RESULTS / "families" / "results.csv")
    order = ["riemann", "spectral", "complexity", "connectivity", "all"]
    names = {"riemann": "Riemannian covariance", "spectral": "Spectral", "complexity": "Complexity",
             "connectivity": "wPLI connectivity", "all": "All combined"}
    piv = df.pivot_table(index="feature_set", columns="task", values="balanced_accuracy").loc[order, TASK_ORDER]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    vmin, vmax = 0.45, 0.85
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    ax.imshow(piv.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            dark = (v - vmin) / (vmax - vmin) > 0.55
            best = v == piv.values[:, j].max()
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=11,
                    color="white" if dark else INK, fontweight="bold" if best else "normal")
    ax.set_xticks(range(3), [TASK_LABELS[t] for t in TASK_ORDER], color=INK)
    ax.set_yticks(range(len(order)), [names[o] for o in order], color=INK)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Which features carry the signal? (LDA, balanced accuracy)", pad=26)
    _subtitle(ax, "Bold = best per task. Exploratory: 15 configurations compared, 10-fold × 1.")
    _save(fig, "feature_families.png")


# --- 3. What auto picked ---------------------------------------------------------

def fig_auto_choices():
    df = _load("headline_auto")
    fams = ["riemann", "spectral", "complexity", "connectivity"]
    names = {"riemann": "Riemannian", "spectral": "Spectral", "complexity": "Complexity", "connectivity": "wPLI"}
    n_folds = 30
    data = {}
    for t in TASK_ORDER:
        counts = dict(re.findall(r"(\w+)x(\d+)", df.loc[t, "choices"].split("families:")[-1]))
        data[t] = [int(counts.get(f, 0)) / n_folds for f in fams]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    height, gap = 0.24, 0.02
    y = np.arange(len(fams))
    for i, t in enumerate(TASK_ORDER):
        pos = y + (i - 1) * (height + gap)
        ax.barh(pos, data[t], height, color=SERIES[i], label=TASK_LABELS[t], zorder=2)
        for p, v in zip(pos, data[t]):
            ax.text(v + 0.01, p, f"{v:.0%}", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, [names[f] for f in fams], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xticks(np.arange(0, 1.01, 0.25), [f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
    ax.set_xlabel("Share of the 30 outer folds where the inner loop kept this family")
    ax.grid(axis="y", visible=False)
    ax.set_title("What the unbiased auto mode chose", pad=26)
    _subtitle(ax, "Feature families are picked inside the inner CV loop, separately in every outer fold.")
    ax.legend(loc="lower right", fontsize=9.5)
    _save(fig, "auto_feature_choices.png")


# --- 4. ROC curves ---------------------------------------------------------------

def fig_roc():
    preds = pd.read_csv(RESULTS / "headline_auto" / "subject_predictions.csv")
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=(0, (4, 3)))
    for i, t in enumerate(TASK_ORDER):
        p = preds[preds["task"] == t]
        fpr, tpr, _ = roc_curve(p["y"], p["score"])
        auc = roc_auc_score(p["y"], p["score"])
        ax.plot(fpr, tpr, color=SERIES[i], lw=2, label=f"{TASK_LABELS[t]}  (AUC {auc:.2f})", drawstyle="steps-post")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False-positive rate (1 − specificity)")
    ax.set_ylabel("True-positive rate (sensitivity)")
    ax.set_aspect("equal")
    ax.set_title("Subject-level ROC, out-of-fold", pad=26)
    _subtitle(ax, "Nested 10-fold × 3; each subject's probability averaged over repeats.")
    ax.legend(loc="lower right", fontsize=9.5)
    _save(fig, "roc_curves.png")


# --- 5. Bias demonstration on pure noise -------------------------------------------

def fig_bias_on_noise():
    runs = [("Nested CV\n(new script)", "nested"), ("Tune and score on\nthe same LOSO folds\n(original design)", "not_nested")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rng = np.random.default_rng(0)
    for i, (label, d) in enumerate(runs):
        vals = pd.read_csv(RESULTS / "bias_demo_synthetic" / d / "results.csv")["balanced_accuracy"].to_numpy()
        yj = i + rng.uniform(-0.13, 0.13, len(vals))
        ax.scatter(vals, yj, s=46, color=SERIES[i], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.plot([vals.mean()] * 2, [i - 0.28, i + 0.28], color=INK, lw=2, zorder=4)
        ax.text(vals.mean(), i + 0.33, f"mean {vals.mean():.0%}", ha="center", fontsize=9, color=INK)
        ax.text(vals.max() + 0.012, i, f"max {vals.max():.0%}", va="center", fontsize=9, color=INK_2)
    ax.axvline(0.5, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.505, 1.5, "chance", color=INK_2, fontsize=9, va="center")
    ax.set_yticks([0, 1], [r[0] for r in runs], color=INK)
    ax.set_ylim(-0.55, 1.65)
    ax.set_xlim(0.2, 1.1)
    ax.set_xticks(np.arange(0.2, 1.01, 0.2), [f"{v:.0%}" for v in np.arange(0.2, 1.01, 0.2)])
    ax.set_xlabel("Balanced accuracy on data with NO real signal (24 configurations each)")
    ax.grid(axis="y", visible=False)
    ax.set_title("The original validation design can score 100% on pure noise", pad=26)
    _subtitle(ax, "Synthetic data, ds004504 class sizes, SVM and KNN, 20 Optuna trials. See REPORT.md §14.1.")
    _save(fig, "bias_on_noise.png")


# --- 6. Peak alpha frequency ---------------------------------------------------------

def fig_peak_alpha():
    path = ROOT / "features_ds004504.npz"
    if not path.exists():
        print("skipped peak_alpha_by_group.png (features_ds004504.npz not found; run `extract` first)")
        return
    with np.load(path) as f:
        names = f["feature_names"].astype(str)
        cols = [i for i, n in enumerate(names) if n in {f"spec_paf_{c}" for c in ("O1", "O2", "P3", "P4", "Pz")}]
        X = f["features"][:, cols].mean(axis=1)
        groups, labels = f["subject_ids"].astype(str), f["labels"]
    df = pd.DataFrame({"subject": groups, "label": labels, "paf": X}).groupby("subject").median()
    order = [(0, "AD", SERIES[0]), (2, "FTD", SERIES[2]), (1, "CN", SERIES[1])]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    rng = np.random.default_rng(1)
    for i, (lab, name, color) in enumerate(order):
        v = df.loc[df["label"] == lab, "paf"].to_numpy()
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=40, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=2.2, zorder=4)
        ax.fill_between([i - 0.28, i + 0.28], q1, q3, color=GRID, alpha=0.8, zorder=2, linewidth=0)
        ax.text(i + 0.31, med, f"{med:.2f} Hz", va="center", fontsize=9.5, color=INK)
    ax.set_xticks(range(3), [f"{n}\n(n={int((df['label'] == lab).sum())})" for lab, n, _ in order], color=INK)
    ax.set_xlim(-0.6, 2.85)
    ax.set_ylabel("Posterior peak alpha frequency (Hz)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Sanity check: alpha slows most in Alzheimer's disease", pad=26)
    _subtitle(ax, "One dot per subject (median over epochs, mean of O1/O2/P3/P4/Pz). Bar = median, band = IQR.")
    _save(fig, "peak_alpha_by_group.png")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    fig_results_vs_literature()
    fig_feature_families()
    fig_auto_choices()
    fig_roc()
    fig_bias_on_noise()
    fig_peak_alpha()

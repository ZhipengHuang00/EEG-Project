"""
EEG dementia classification (AD / FTD / CN): from raw EEG to honest,
subject-level results.

Cleaned-up, research-backed rewrite of DL56Bandwise.py. REPORT.md (sections 14-15)
explains each design choice and lists the papers it follows.

Subcommands
-----------
  download  Fetch OpenNeuro ds004504 (88 subjects, preprocessed with ASR+ICA, ~3 GB).
  extract   Raw EEG -> per-epoch features (10 s epochs, 50% overlap):
              spectral     log absolute and relative power (6 bands), theta/alpha and slowing
                           ratios, spectral entropy, peak alpha frequency, aperiodic 1/f fit
              complexity   Hjorth mobility/complexity, permutation and sample entropy,
                           Higuchi and Katz fractal dimension
              connectivity debiased wPLI (171 pairs x 6 bands) and node strength
              riemann      per-band covariance matrices (tangent space is fitted inside the CV)
  classify  Nested, subject-grouped CV with feature selection inside the folds,
            repeated outer CV, an optional permutation test and an `auto` mode that also picks
            the model and feature families inside the inner loop, so best-of-N selection
            can't inflate the score. Also reads the original RBP/CPTE .npz layout.

Quick start
-----------
    python eeg_bandwise_classification.py download
    python eeg_bandwise_classification.py extract --n-jobs -1
    python eeg_bandwise_classification.py classify --data features_ds004504.npz --repeats 3 --n-jobs -1
    python eeg_bandwise_classification.py classify --quick --data features_ds004504.npz
"""
import argparse
import importlib.util
import itertools
import json
import logging
import os
import time
import urllib.request
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger("eeg")

# --- Constants ----------------------------------------------------------------

CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha1": (8, 10), "alpha2": (10, 13),
         "beta": (13, 30), "gamma": (30, 45)}
LEGACY_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]  # layout of the original RBP .npz
BAND_TAGS = set(BANDS) | set(LEGACY_BANDS)
REGIONS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "temporal": ["T3", "T4", "T5", "T6"],
    "central": ["C3", "C4", "Cz"],
    "parietal": ["P3", "P4", "Pz"],
    "occipital": ["O1", "O2"],
}
CH_TO_REGION = {ch: region for region, chs in REGIONS.items() for ch in chs}
FAMILY_PREFIX = {"spec": "spectral", "cplx": "complexity", "conn": "connectivity", "cpte": "cpte"}

CLASS_NAMES = {0: "AD", 1: "CN", 2: "FTD"}
GROUP_LABELS = {"A": 0, "C": 1, "F": 2}  # participants.tsv `Group` column
# Label 1 is always the "positive" class: the disease vs CN, and FTD in AD vs FTD.
TASKS = {
    "AD_CN": {"negative": 1, "positive": 0},
    "FTD_CN": {"negative": 1, "positive": 2},
    "AD_FTD": {"negative": 0, "positive": 2},
}
MODELS = ["LogReg", "LDA", "SVM", "RandomForest", "FamilyVote", "XGBoost", "CatBoost", "KNN", "MLP", "LightGBM"]
AUTO_MODELS = ["LogReg", "LDA", "SVM", "RandomForest", "FamilyVote"]
OPTIONAL_PACKAGES = {"XGBoost": "xgboost", "CatBoost": "catboost", "LightGBM": "lightgbm"}
CLASS_WEIGHT_MODELS = {"LogReg", "RandomForest", "SVM", "XGBoost", "CatBoost", "LightGBM"}

DS004504_URL = "https://s3.amazonaws.com/openneuro.org/ds004504"

# Subject-level, leakage-free results on ds004504 to compare against.
LITERATURE = [
    ("Miltiadous et al. 2023 (dataset paper): RBP + RF/MLP, LOSO", "77.0 acc", "73.1 acc", "-"),
    ("AHEPA review 2026: mean of Validity-1 studies", "82.1 acc", "75.2 acc", "71.4 acc"),
    ("Riemannian FC stacking (PMC12782479), LOSO", "85 acc", "74 acc", "74 acc"),
    ("FC stacking, Sci Rep 2026, LOSO", "81.8 AUC", "71.4 AUC", "65.1 AUC"),
    ("Leakage-safe benchmark (arXiv 2609.22092), LOSO", "87.7 acc", "82.7 acc", "78.0 acc"),
]


# =============================================================================
# download
# =============================================================================

def cmd_download(args):
    subjects = args.subjects or list(range(1, 89))
    files = ["participants.tsv"] + [
        f"derivatives/sub-{i:03d}/eeg/sub-{i:03d}_task-eyesclosed_eeg.set" for i in subjects]

    def fetch(rel):
        dest = args.out / rel
        url = f"{DS004504_URL}/{rel}"
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            size = int(r.headers["Content-Length"])
        if dest.exists() and dest.stat().st_size == size:
            return rel, "cached"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        return rel, "downloaded"

    with ThreadPoolExecutor(args.n_jobs) as pool:
        for rel, status in pool.map(fetch, files):
            logger.info("%-10s %s", status, rel)
    logger.info("ds004504 is in %s", args.out)


# =============================================================================
# extract
# =============================================================================

@dataclass
class ExtractConfig:
    sfreq: int = 250
    epoch_sec: float = 10.0
    step_sec: float = 5.0          # 50% overlap
    ptp_max_uv: float = 150.0      # peak-to-peak rejection threshold ...
    ptp_adaptive_factor: float = 2.0  # ... or this x the subject's median epoch peak-to-peak, if higher
    logvar_z_max: float = 3.5      # robust z of log-variance vs. the subject's median


def _band_power(f, psd, lo, hi):
    m = (f >= lo) & (f < hi)
    return psd[..., m].sum(-1) * (f[1] - f[0])


def spectral_features(ep: np.ndarray, sf: int) -> Tuple[List[np.ndarray], List[str]]:
    """ep: (epochs, channels, samples) in µV. Returns blocks of shape (epochs, channels)."""
    from scipy.signal import welch
    f, psd = welch(ep, fs=sf, nperseg=4 * sf, noverlap=2 * sf, axis=-1)
    power = {b: _band_power(f, psd, lo, hi) for b, (lo, hi) in BANDS.items()}
    total = _band_power(f, psd, 0.5, 45)
    alpha = power["alpha1"] + power["alpha2"]

    blocks, measures = [], []
    for b in BANDS:
        blocks.append(np.log10(power[b]))
        measures.append(f"spec_abspow_{b}")
    for b in BANDS:
        blocks.append(power[b] / total)
        measures.append(f"spec_relpow_{b}")
    blocks.append(np.log10(power["theta"] / alpha))
    measures.append("spec_ratio-theta-alpha")
    blocks.append(np.log10((power["delta"] + power["theta"]) / (alpha + power["beta"])))
    measures.append("spec_ratio-slowing")

    m = (f >= 0.5) & (f < 45)
    p = psd[..., m] / psd[..., m].sum(-1, keepdims=True)
    blocks.append(-(p * np.log2(p + 1e-30)).sum(-1) / np.log2(m.sum()))
    measures.append("spec_entropy")

    ma = (f >= 7) & (f <= 13)  # peak alpha frequency as centre of gravity
    blocks.append((psd[..., ma] * f[ma]).sum(-1) / psd[..., ma].sum(-1))
    measures.append("spec_paf")

    # Aperiodic (1/f) component: least-squares line in log-log space, skipping the alpha peak.
    fit = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    A = np.column_stack([np.log10(f[fit]), np.ones(fit.sum())])
    logp = np.log10(psd[..., fit]).reshape(-1, fit.sum()).T
    coef, *_ = np.linalg.lstsq(A, logp, rcond=None)
    blocks.append(-coef[0].reshape(psd.shape[:2]))
    measures.append("spec_aperiodic-exp")
    blocks.append(coef[1].reshape(psd.shape[:2]))
    measures.append("spec_aperiodic-off")

    names = [f"{m_}_{ch}" for m_ in measures for ch in CHANNELS]
    return blocks, names


def complexity_features(ep: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
    import antropy as ant
    E, C, _ = ep.shape
    mobility, complexity = ant.hjorth_params(ep, axis=-1)
    perm, sampen, higuchi = (np.empty((E, C)) for _ in range(3))
    for e in range(E):
        for c in range(C):
            x = np.ascontiguousarray(ep[e, c], dtype=np.float64)
            perm[e, c] = ant.perm_entropy(x, order=3, delay=1, normalize=True)
            sampen[e, c] = ant.sample_entropy(x, order=2)
            higuchi[e, c] = ant.higuchi_fd(x, kmax=10)
    katz = ant.katz_fd(ep, axis=-1)
    measures = ["cplx_hjorth-mob", "cplx_hjorth-comp", "cplx_permen", "cplx_sampen", "cplx_higuchi", "cplx_katz"]
    names = [f"{m}_{ch}" for m in measures for ch in CHANNELS]
    return [mobility, complexity, perm, sampen, higuchi, katz], names


def connectivity_features(ep: np.ndarray, sf: int, chunk: int = 16) -> Tuple[np.ndarray, List[str]]:
    """Debiased wPLI (Vinck et al. 2011) per band, estimated from 2 s Hann segments
    (1 s step) inside each epoch, averaged over the band's frequency bins.
    Also returns node strength (mean wPLI of each channel)."""
    E, C, T = ep.shape
    seg, step = 2 * sf, sf
    starts = range(0, T - seg + 1, step)
    freqs = np.fft.rfftfreq(seg, 1 / sf)
    iu, ju = np.triu_indices(C, 1)
    win = np.hanning(seg)

    wpli = {b: np.empty((E, len(iu))) for b in BANDS}
    for e0 in range(0, E, chunk):
        segs = np.stack([ep[e0:e0 + chunk, :, s:s + seg] for s in starts], axis=1)  # (e, S, C, seg)
        segs = segs - segs.mean(-1, keepdims=True)
        spec = np.fft.rfft(segs * win, axis=-1)
        for b, (lo, hi) in BANDS.items():
            sb = spec[..., (freqs >= lo) & (freqs < hi)]
            im = np.imag(sb[:, :, iu] * np.conj(sb[:, :, ju]))  # (e, S, P, K)
            s, s_abs, s_sq = im.sum(1), np.abs(im).sum(1), (im ** 2).sum(1)
            wpli[b][e0:e0 + chunk] = ((s ** 2 - s_sq) / (s_abs ** 2 - s_sq + 1e-30)).mean(-1)

    blocks, names = [], []
    for b in BANDS:
        blocks.append(wpli[b])
        names += [f"conn_wpli_{b}_{CHANNELS[i]}-{CHANNELS[j]}" for i, j in zip(iu, ju)]
    for b in BANDS:
        W = np.zeros((E, C, C))
        W[:, iu, ju] = wpli[b]
        W = W + W.transpose(0, 2, 1)
        blocks.append(W.sum(-1) / (C - 1))
        names += [f"conn_strength_{b}_{ch}" for ch in CHANNELS]
    return np.concatenate(blocks, axis=1), names


def band_covariances(data: np.ndarray, starts: np.ndarray, win: int, sf: int) -> np.ndarray:
    """(epochs, bands, C, C) covariance of band-pass filtered signal, in µV²."""
    from scipy.signal import butter, sosfiltfilt
    out = np.empty((len(starts), len(BANDS), data.shape[0], data.shape[0]), dtype=np.float32)
    for k, (lo, hi) in enumerate(BANDS.values()):
        filtered = sosfiltfilt(butter(4, [lo, hi], btype="bandpass", fs=sf, output="sos"), data, axis=-1)
        ep = np.stack([filtered[:, s:s + win] for s in starts])
        ep = ep - ep.mean(-1, keepdims=True)
        out[:, k] = np.einsum("ect,edt->ecd", ep, ep) / win
    return out


def extract_subject(set_path: Path, cfg: ExtractConfig) -> dict:
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_eeglab(set_path, preload=True)
    raw.pick(CHANNELS).reorder_channels(CHANNELS)
    raw.filter(0.5, 45.0)
    raw.set_eeg_reference("average")
    raw.resample(cfg.sfreq)
    data = raw.get_data() * 1e6  # V -> µV

    sf = cfg.sfreq
    win, step = int(cfg.epoch_sec * sf), int(cfg.step_sec * sf)
    starts = np.arange(0, data.shape[1] - win + 1, step)
    ep = np.stack([data[:, s:s + win] for s in starts])

    # Peak-to-peak limit: 150 µV, raised for subjects whose *typical* amplitude is high
    # (otherwise a whole high-amplitude recording would be discarded, not just its artifacts).
    max_ptp = np.ptp(ep, axis=-1).max(axis=1)
    ptp_limit = max(cfg.ptp_max_uv, cfg.ptp_adaptive_factor * np.median(max_ptp))
    ptp_bad = max_ptp > ptp_limit
    logvar = np.log(ep.var(axis=-1)).mean(axis=1)
    mad = 1.4826 * np.median(np.abs(logvar - np.median(logvar))) + 1e-12
    var_bad = np.abs(logvar - np.median(logvar)) / mad > cfg.logvar_z_max
    keep = ~(ptp_bad | var_bad)
    if keep.sum() < 5:
        raise ValueError(f"{set_path.name}: only {keep.sum()} clean epochs")
    ep, starts = ep[keep], starts[keep]

    spec_blocks, spec_names = spectral_features(ep, sf)
    cplx_blocks, cplx_names = complexity_features(ep)
    conn, conn_names = connectivity_features(ep, sf)
    features = np.concatenate(spec_blocks + cplx_blocks + [conn], axis=1).astype(np.float32)
    return {
        "features": features,
        "feature_names": np.array(spec_names + cplx_names + conn_names),
        "covariances": band_covariances(data, starts, win, sf),
        "epoch_start_sec": starts / sf,
        "n_epochs_total": len(keep),
        "ptp_limit_uv": ptp_limit,
    }


def _extract_cached(root: Path, sid: str, cache: Path, cfg: ExtractConfig) -> Tuple[str, dict]:
    out = cache / f"{sid}.npz"
    if out.exists():
        with np.load(out) as f:
            return sid, {k: f[k] for k in f.files}
    path = root / "derivatives" / sid / "eeg" / f"{sid}_task-eyesclosed_eeg.set"
    t0 = time.time()
    res = extract_subject(path, cfg)
    np.savez_compressed(out, **res)
    print(f"{sid}: {len(res['features'])}/{res['n_epochs_total']} epochs kept "
          f"(ptp limit {float(res['ptp_limit_uv']):.0f} µV), {time.time() - t0:.0f}s", flush=True)
    return sid, res


def cmd_extract(args):
    root = args.bids_root
    parts = pd.read_csv(root / "participants.tsv", sep="\t")
    parts = parts[parts["Group"].isin(GROUP_LABELS)]
    if args.subjects:
        parts = parts[parts["participant_id"].isin([f"sub-{i:03d}" for i in args.subjects])]
    available = [(root / "derivatives" / s / "eeg" / f"{s}_task-eyesclosed_eeg.set").exists()
                 for s in parts["participant_id"]]
    if not all(available):
        missing = parts["participant_id"][~np.array(available)].tolist()
        raise SystemExit(f"Missing EEG for {len(missing)} subjects (e.g. {missing[:3]}). Run `download` first.")

    cache = args.cache_dir or root / "features_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cfg = ExtractConfig()
    logger.info("Extracting %d subjects -> %s (cache: %s)", len(parts), args.out, cache)
    results = dict(Parallel(n_jobs=args.n_jobs)(
        delayed(_extract_cached)(root, sid, cache, cfg) for sid in parts["participant_id"]))

    feats, covs, labels, sids, ages, sexes, starts = [], [], [], [], [], [], []
    names = None
    for _, row in parts.iterrows():
        res = results[row["participant_id"]]
        n = len(res["features"])
        if names is None:
            names = res["feature_names"]
        feats.append(res["features"])
        covs.append(res["covariances"])
        starts.append(res["epoch_start_sec"])
        labels += [GROUP_LABELS[row["Group"]]] * n
        sids += [row["participant_id"]] * n
        ages += [row["Age"]] * n
        sexes += [row["Gender"]] * n
    # MMSE is deliberately NOT exported: it is part of the diagnosis, so using it as a feature leaks the label.
    np.savez_compressed(
        args.out, features=np.vstack(feats), feature_names=names, labels=np.array(labels),
        subject_ids=np.array(sids), covariances=np.concatenate(covs), cov_bands=np.array(list(BANDS)),
        age=np.array(ages), sex=np.array(sexes), epoch_start_sec=np.concatenate(starts),
        extract_config=json.dumps(asdict(cfg)))

    X = np.vstack(feats)
    kept = [len(results[s]["features"]) / results[s]["n_epochs_total"] for s in parts["participant_id"]]
    logger.info("Saved %s: %d epochs x %d features (+ %d covariance bands)",
                args.out, X.shape[0], X.shape[1], len(BANDS))
    logger.info("Epochs kept per subject: median %.0f%%, min %.0f%% | NaN features: %d",
                100 * np.median(kept), 100 * np.min(kept), int(np.isnan(X).sum()))
    paf_cols = [i for i, n in enumerate(names) if n.startswith("spec_paf_")]
    for label, cname in CLASS_NAMES.items():
        mask = np.array(labels) == label
        if not mask.any():
            continue
        logger.info("  %s: median peak alpha frequency %.2f Hz", cname, np.nanmedian(X[mask][:, paf_cols]))


# =============================================================================
# classify: data
# =============================================================================

@dataclass
class Dataset:
    X: np.ndarray
    names: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    covs: Optional[np.ndarray] = None  # (rows, n_bands, C, C)
    cov_bands: Tuple[str, ...] = ()

    def __post_init__(self):
        parsed = [parse_name(n) for n in self.names]
        self.col_family = np.array([p[0] for p in parsed])
        self.col_band = np.array([p[1] or "" for p in parsed])

    def families(self) -> List[str]:
        fams = list(dict.fromkeys(self.col_family))
        return fams + (["riemann"] if self.covs is not None else [])


def parse_name(name: str) -> Tuple[str, Optional[str], str]:
    """'spec_relpow_alpha1_Fp1' -> ('spectral', 'alpha1', 'Fp1'). Band is None for band-free features."""
    parts = name.split("_")
    band = parts[2] if len(parts) == 4 and parts[2] in BAND_TAGS else None
    return FAMILY_PREFIX[parts[0]], band, parts[-1]


def band_matches(tag: str, band: str) -> bool:
    return band == "all" or (bool(tag) and tag.startswith(band))  # "alpha" matches alpha1 and alpha2


def _read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        try:
            return {k: f[k] for k in f.files}
        except ValueError:
            pass
    logger.warning("%s contains object arrays; loading with allow_pickle=True "
                   "(only do this for files you trust).", path)
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def _legacy_names(n_cols: int, extra_cpte: int, rbp_layout: str) -> List[str]:
    """Names for the original DL56Bandwise layout: 95 RBP columns, then CPTE columns."""
    chans = [f"ch{i + 1:02d}" for i in range(19)]
    if rbp_layout == "band-major":
        rbp = [f"spec_relpow_{b}_{c}" for b in LEGACY_BANDS for c in chans]
    else:
        rbp = [f"spec_relpow_{b}_{c}" for c in chans for b in LEGACY_BANDS]
    n_cpte = n_cols - 95 + extra_cpte
    if n_cpte and n_cpte % 5 == 0:
        per = n_cpte // 5
        cpte = [f"cpte_val_{b}_{i}" for b in LEGACY_BANDS for i in range(per)]
    else:
        if n_cpte:
            logger.warning("CPTE has %d columns (not divisible by 5 bands), so it is only usable "
                           "with --bands all.", n_cpte)
        cpte = [f"cpte_val_{i}" for i in range(n_cpte)]
    if n_cpte == 105:
        logger.warning("CPTE block has exactly 105 columns, the number of network (edge) features the "
                       "original docstring described. Check that these really are CPTE.")
    return rbp + cpte


def to_regions(X: np.ndarray, names: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Average channel features within brain regions (and channel pairs within region pairs).
    A fixed per-row operation, so it can't leak information between subjects."""
    new = []
    for n in names:
        parts = n.split("_")
        loc = parts[-1]
        if "-" in loc:
            a, b = loc.split("-")
            regions = sorted([CH_TO_REGION.get(a, ""), CH_TO_REGION.get(b, "")])
            region = "-".join(regions) if all(regions) else None
        else:
            region = CH_TO_REGION.get(loc)
        new.append("_".join(parts[:-1] + [region]) if region else n)
    df = pd.DataFrame(X, columns=new)
    df = df.T.groupby(level=0, sort=False).mean().T
    return df.to_numpy(), df.columns.to_numpy()


def load_dataset(path: Path, args) -> Dataset:
    arrays = _read_npz(path)
    for key in ("features", "labels", "subject_ids"):
        if key not in arrays:
            raise KeyError(f"{path} is missing required array '{key}' (found: {sorted(arrays)})")
    X = arrays["features"].astype(np.float64)
    y = arrays["labels"].astype(int)
    groups = arrays["subject_ids"].astype(str)
    unknown = set(np.unique(y)) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Unexpected label values {sorted(unknown)}; expected {CLASS_NAMES}")

    covs, cov_bands = None, ()
    if "feature_names" in arrays and str(arrays["feature_names"][0]).split("_")[0] in FAMILY_PREFIX:
        names = arrays["feature_names"].astype(str)
        if "covariances" in arrays:
            covs = arrays["covariances"].astype(np.float64)
            cov_bands = tuple(arrays["cov_bands"].astype(str))
    else:  # original DL56Bandwise layout
        if X.shape[1] < 95:
            raise ValueError(f"Expected at least 95 RBP columns, got {X.shape[1]}")
        extra = next((arrays[k].astype(np.float64) for k in ("cpte_features", "cpte") if k in arrays), None)
        if extra is not None:
            X = np.hstack([X, extra])
        names = np.array(_legacy_names(X.shape[1] - (0 if extra is None else extra.shape[1]),
                                       0 if extra is None else extra.shape[1], args.rbp_layout))
        logger.info("Original RBP/CPTE layout detected (RBP order: %s)", args.rbp_layout)

    subjects = np.unique(groups)
    mixed = [s for s in subjects if len(np.unique(y[groups == s])) > 1]
    if mixed:
        raise ValueError(f"Subjects with more than one label: {mixed[:5]}")

    if args.unit == "subject" and len(subjects) < len(groups):
        # One row per subject: the median over epochs for features, the mean for covariances.
        order = np.argsort(groups, kind="stable")
        X, y, groups = X[order], y[order], groups[order]
        covs = covs[order] if covs is not None else None
        subjects, first = np.unique(groups, return_index=True)
        X = pd.DataFrame(X).groupby(groups, sort=True).median().to_numpy()
        if covs is not None:
            covs = np.stack([covs[groups == s].mean(0) for s in subjects])
        y, groups = y[first], subjects
    elif args.unit == "epoch" and args.max_epochs:
        rng = np.random.default_rng(args.seed)
        keep = np.concatenate([rng.permutation(np.flatnonzero(groups == s))[:args.max_epochs] for s in subjects])
        keep.sort()
        X, y, groups = X[keep], y[keep], groups[keep]
        covs = covs[keep] if covs is not None else None

    if args.spatial == "regions":
        X, names = to_regions(X, names)

    ds = Dataset(X, names, y, groups, covs, cov_bands)
    logger.info("Rows: %d (%s level) | subjects: %d | columns: %d | families: %s",
                len(y), args.unit, len(np.unique(groups)), X.shape[1], ", ".join(ds.families()))
    for label, name in CLASS_NAMES.items():
        logger.info("  %s: %d subjects", name, len(np.unique(groups[y == label])))
    logger.info("NaN: %d | Inf: %d", np.isnan(X).sum(), np.isinf(X).sum())
    return ds


@dataclass
class Layout:
    """Where each feature family lives in a design matrix."""
    family_cols: Dict[str, np.ndarray]
    cov_cols: Optional[np.ndarray] = None
    n_cov_bands: int = 0
    n_ch: int = 0

    def families(self) -> List[str]:
        return list(self.family_cols) + (["riemann"] if self.cov_cols is not None else [])


def parse_feature_set(spec: str, ds: Dataset) -> List[str]:
    available = ds.families()
    fams = available if spec == "all" else spec.split("+")
    unknown = [f for f in fams if f not in available]
    if unknown:
        raise SystemExit(f"Feature set '{spec}': {unknown} not in data (available: {available})")
    return fams


def make_design(ds: Dataset, families: List[str], band: str):
    """Return (X, Layout, "") or (None, None, reason to skip)."""
    band_ok = np.array([band_matches(t, band) for t in ds.col_band])
    blocks, family_cols, start = [], {}, 0
    for fam in families:
        if fam == "riemann":
            continue
        idx = np.flatnonzero((ds.col_family == fam) & band_ok)
        if len(idx):
            family_cols[fam] = np.arange(start, start + len(idx))
            blocks.append(ds.X[:, idx])
            start += len(idx)
    layout = Layout(family_cols)
    if "riemann" in families and ds.covs is not None:
        bsel = [i for i, b in enumerate(ds.cov_bands) if band_matches(b, band)]
        if bsel:
            C = ds.covs[:, bsel].reshape(len(ds.X), -1)
            layout.cov_cols = np.arange(start, start + C.shape[1])
            layout.n_cov_bands, layout.n_ch = len(bsel), ds.covs.shape[-1]
            blocks.append(C)
    if not blocks:
        return None, None, f"no {'+'.join(families)} features for band '{band}'"
    return np.hstack(blocks), layout, ""


def task_labels(y: np.ndarray, task: str) -> Tuple[np.ndarray, np.ndarray]:
    spec = TASKS[task]
    mask = np.isin(y, [spec["negative"], spec["positive"]])
    return mask, (y[mask] == spec["positive"]).astype(int)


# =============================================================================
# classify: models
# =============================================================================

def _inf_to_nan(X):
    return np.where(np.isinf(X), np.nan, X)


def _tabular_steps() -> Pipeline:
    return Pipeline([
        ("inf_to_nan", FunctionTransformer(_inf_to_nan)),
        ("impute", SimpleImputer(strategy="mean", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])


class TangentSpaceFeatures(TransformerMixin, BaseEstimator):
    """Per-band covariance matrices -> Riemannian tangent-space vectors. The reference
    (Riemannian mean) is fitted on the training fold only."""

    def __init__(self, n_bands=1, n_ch=19, shrinkage=0.05):
        self.n_bands, self.n_ch, self.shrinkage = n_bands, n_ch, shrinkage

    def _covs(self, X):
        C = np.asarray(X, dtype=np.float64).reshape(len(X), self.n_bands, self.n_ch, self.n_ch)
        scale = np.trace(C, axis1=-2, axis2=-1)[..., None, None] / self.n_ch
        return (1 - self.shrinkage) * C + self.shrinkage * scale * np.eye(self.n_ch)

    def fit(self, X, y=None):
        from pyriemann.tangentspace import TangentSpace
        C = self._covs(X)
        self.ts_ = [TangentSpace(metric="riemann").fit(C[:, b]) for b in range(self.n_bands)]
        return self

    def transform(self, X):
        C = self._covs(X)
        return np.hstack([ts.transform(C[:, b]) for b, ts in enumerate(self.ts_)])


def _feature_block(layout: Layout, families: List[str]) -> Tuple[ColumnTransformer, int]:
    tab = [layout.family_cols[f] for f in families if f in layout.family_cols]
    transformers, n_out = [], 0
    if tab:
        tab = np.concatenate(tab)
        transformers.append(("tab", _tabular_steps(), tab))
        n_out += len(tab)
    if "riemann" in families and layout.cov_cols is not None:
        ts = Pipeline([("tangent", TangentSpaceFeatures(layout.n_cov_bands, layout.n_ch)),
                       ("scale", StandardScaler())])
        transformers.append(("riemann", ts, layout.cov_cols))
        n_out += layout.n_cov_bands * layout.n_ch * (layout.n_ch + 1) // 2
    return ColumnTransformer(transformers), n_out


class FamilyVote(ClassifierMixin, BaseEstimator):
    """Soft vote of one regularised logistic regression per feature family. Each member
    selects its own top-k features. Nothing is fitted on held-out data."""

    def __init__(self, layout=None, families=(), C=1.0, k=50):
        self.layout, self.families, self.C, self.k = layout, families, C, k

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.members_ = []
        for fam in self.families:
            block, n_out = _feature_block(self.layout, [fam])
            steps = [("features", block)]
            if self.k < n_out:
                steps.append(("select", SelectKBest(f_classif, k=self.k)))
            steps.append(("model", LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000)))
            self.members_.append(Pipeline(steps).fit(X, y))
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def active_families(p: dict, layout: Layout) -> List[str]:
    return [f for f in layout.families() if p.get(f"use_{f}", True)]


def suggest_params(trial: optuna.Trial, model_name: str, layout: Layout) -> dict:
    """The only place the search space is defined. build_pipeline reads these same keys."""
    p = {}
    if model_name == "auto":
        # Model and feature families are chosen in the inner loop, so the outer score
        # covers the whole "pick the best pipeline" procedure.
        model_name = p["model"] = trial.suggest_categorical("model", AUTO_MODELS)
        if len(layout.families()) > 1:
            for fam in layout.families():
                p[f"use_{fam}"] = trial.suggest_categorical(f"use_{fam}", [True, False])

    if model_name == "FamilyVote":
        p["fv_C"] = trial.suggest_float("fv_C", 1e-3, 10, log=True)
        p["fv_k"] = trial.suggest_categorical("fv_k", [10, 25, 50, 100])
        return p

    p["reduce"] = trial.suggest_categorical("reduce", ["kbest", "none", "pca"])
    if p["reduce"] == "kbest":
        p["k"] = trial.suggest_categorical("k", [10, 25, 50, 100, 200])
        p["select_score"] = trial.suggest_categorical("select_score", ["f", "mi"])
    elif p["reduce"] == "pca":
        p["pca_variance"] = trial.suggest_float("pca_variance", 0.80, 0.99)
    if model_name in CLASS_WEIGHT_MODELS:
        p["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])

    if model_name == "LogReg":
        p["logreg_C"] = trial.suggest_float("logreg_C", 1e-3, 1e2, log=True)
        p["logreg_l1_ratio"] = trial.suggest_float("logreg_l1_ratio", 0.0, 1.0)
    elif model_name == "LDA":
        pass  # shrinkage is set analytically (Ledoit-Wolf)
    elif model_name == "RandomForest":
        p["rf_n_estimators"] = trial.suggest_int("rf_n_estimators", 100, 500, step=50)
        p["rf_max_depth"] = trial.suggest_int("rf_max_depth", 2, 12)
        p["rf_min_samples_leaf"] = trial.suggest_int("rf_min_samples_leaf", 1, 5)
        p["rf_max_features"] = trial.suggest_categorical("rf_max_features", ["sqrt", "log2"])
    elif model_name == "SVM":
        p["svm_C"] = trial.suggest_float("svm_C", 1e-3, 1e2, log=True)
        p["svm_kernel"] = trial.suggest_categorical("svm_kernel", ["linear", "rbf"])
        if p["svm_kernel"] == "rbf":
            p["svm_gamma"] = trial.suggest_float("svm_gamma", 1e-4, 1.0, log=True)
    elif model_name == "XGBoost":
        p["xgb_n_estimators"] = trial.suggest_int("xgb_n_estimators", 50, 300)
        p["xgb_max_depth"] = trial.suggest_int("xgb_max_depth", 2, 6)
        p["xgb_learning_rate"] = trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True)
        p["xgb_subsample"] = trial.suggest_float("xgb_subsample", 0.5, 1.0)
        p["xgb_colsample_bytree"] = trial.suggest_float("xgb_colsample_bytree", 0.3, 1.0)
        p["xgb_min_child_weight"] = trial.suggest_float("xgb_min_child_weight", 0.5, 5.0)
    elif model_name == "CatBoost":
        p["cat_iterations"] = trial.suggest_int("cat_iterations", 100, 400, step=50)
        p["cat_depth"] = trial.suggest_int("cat_depth", 2, 6)
        p["cat_learning_rate"] = trial.suggest_float("cat_learning_rate", 0.01, 0.3, log=True)
        p["cat_l2_leaf_reg"] = trial.suggest_float("cat_l2_leaf_reg", 1.0, 10.0, log=True)
    elif model_name == "KNN":
        p["knn_n_neighbors"] = trial.suggest_int("knn_n_neighbors", 3, 15)
        p["knn_weights"] = trial.suggest_categorical("knn_weights", ["uniform", "distance"])
        p["knn_p"] = trial.suggest_int("knn_p", 1, 2)
    elif model_name == "MLP":
        p["mlp_hidden"] = trial.suggest_categorical("mlp_hidden", ["32", "64", "64-32"])
        p["mlp_activation"] = trial.suggest_categorical("mlp_activation", ["relu", "tanh"])
        p["mlp_alpha"] = trial.suggest_float("mlp_alpha", 1e-5, 1.0, log=True)
        p["mlp_learning_rate_init"] = trial.suggest_float("mlp_learning_rate_init", 1e-4, 1e-2, log=True)
    elif model_name == "LightGBM":
        p["lgb_n_estimators"] = trial.suggest_int("lgb_n_estimators", 50, 300)
        p["lgb_learning_rate"] = trial.suggest_float("lgb_learning_rate", 0.01, 0.3, log=True)
        p["lgb_num_leaves"] = trial.suggest_int("lgb_num_leaves", 4, 31)
        p["lgb_min_child_samples"] = trial.suggest_int("lgb_min_child_samples", 2, 15)
        p["lgb_colsample_bytree"] = trial.suggest_float("lgb_colsample_bytree", 0.3, 1.0)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return p


def _build_model(model_name: str, p: dict, y_train: np.ndarray, seed: int):
    cw = p.get("class_weight")
    if model_name == "LogReg":
        return LogisticRegression(penalty="elasticnet", solver="saga", C=p["logreg_C"],
                                  l1_ratio=p["logreg_l1_ratio"], class_weight=cw,
                                  max_iter=5000, tol=1e-3, random_state=seed)
    if model_name == "LDA":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=p["rf_n_estimators"], max_depth=p["rf_max_depth"],
            min_samples_leaf=p["rf_min_samples_leaf"], max_features=p["rf_max_features"],
            class_weight=cw, random_state=seed, n_jobs=1)
    if model_name == "SVM":
        return SVC(C=p["svm_C"], kernel=p["svm_kernel"], gamma=p.get("svm_gamma", "scale"),
                   class_weight=cw, random_state=seed)
    if model_name == "XGBoost":
        from xgboost import XGBClassifier
        n_pos = max(int(y_train.sum()), 1)
        scale_pos_weight = (len(y_train) - n_pos) / n_pos if cw == "balanced" else 1.0
        return XGBClassifier(
            n_estimators=p["xgb_n_estimators"], max_depth=p["xgb_max_depth"],
            learning_rate=p["xgb_learning_rate"], subsample=p["xgb_subsample"],
            colsample_bytree=p["xgb_colsample_bytree"], min_child_weight=p["xgb_min_child_weight"],
            scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=seed, n_jobs=1)
    if model_name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=p["cat_iterations"], depth=p["cat_depth"],
            learning_rate=p["cat_learning_rate"], l2_leaf_reg=p["cat_l2_leaf_reg"],
            auto_class_weights="Balanced" if cw == "balanced" else None,
            random_seed=seed, thread_count=1, verbose=0, allow_writing_files=False)
    if model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=min(p["knn_n_neighbors"], len(y_train) - 1),
                                    weights=p["knn_weights"], p=p["knn_p"])
    if model_name == "MLP":
        hidden = tuple(int(n) for n in p["mlp_hidden"].split("-"))
        return MLPClassifier(hidden_layer_sizes=hidden, activation=p["mlp_activation"],
                             alpha=p["mlp_alpha"], learning_rate_init=p["mlp_learning_rate_init"],
                             max_iter=1000, random_state=seed)
    if model_name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=p["lgb_n_estimators"], learning_rate=p["lgb_learning_rate"],
            num_leaves=p["lgb_num_leaves"], min_child_samples=p["lgb_min_child_samples"],
            colsample_bytree=p["lgb_colsample_bytree"], class_weight=cw,
            random_state=seed, n_jobs=1, verbose=-1)
    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(model_name: str, p: dict, y_train: np.ndarray, seed: int, layout: Layout):
    """Every preprocessing and selection step lives inside the pipeline, so it is fitted on training rows only."""
    name = p.get("model", model_name)
    fams = active_families(p, layout) or layout.families()
    if name == "FamilyVote":
        return FamilyVote(layout, tuple(fams), C=p["fv_C"], k=p["fv_k"])
    block, n_out = _feature_block(layout, fams)
    steps = [("features", block)]
    if p.get("reduce") == "kbest" and p["k"] < n_out:
        score = f_classif if p["select_score"] == "f" else partial(mutual_info_classif, random_state=seed)
        steps.append(("select", SelectKBest(score, k=p["k"])))
    elif p.get("reduce") == "pca":
        steps.append(("pca", PCA(n_components=p["pca_variance"], svd_solver="full", random_state=seed)))
    steps.append(("model", _build_model(name, p, y_train, seed)))
    return Pipeline(steps)


# =============================================================================
# classify: cross-validation
# =============================================================================

@dataclass
class RunConfig:
    n_trials: int
    inner_folds: int
    outer_folds: int  # 0 = leave-one-subject-out
    repeats: int
    permutations: int
    nested: bool
    seed: int


def _fit_predict(model_name, params, X, y, train, test, seed, layout) -> Optional[np.ndarray]:
    """Fit on `train`; return P(label 1) for `test`, or None if the fold can't be fitted."""
    if len(np.unique(y[train])) < 2:
        return None
    pipe = build_pipeline(model_name, params, y[train], seed, layout)
    try:
        pipe.fit(X[train], y[train])
    except Exception as e:  # keep the run alive, but count the subjects as failed
        logger.warning("%s fit failed: %s", params.get("model", model_name), e)
        return None
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X[test])[:, 1]
    return 1 / (1 + np.exp(-pipe.decision_function(X[test])))  # monotone: same ranking and 0.5 cut-off


def cross_val_scores(model_name, params, X, y, groups, cv, seed, layout):
    score = np.full(len(y), np.nan)
    failed = []
    for train, test in cv.split(X, y, groups):
        out = _fit_predict(model_name, params, X, y, train, test, seed, layout)
        if out is None:
            failed.extend(np.unique(groups[test]).tolist())
        else:
            score[test] = out
    return score, failed


def to_subject_level(y, score, groups) -> pd.DataFrame:
    """Average row (epoch) probabilities per subject; predict label 1 if the mean is >= 0.5."""
    df = pd.DataFrame({"subject": groups, "y": y, "score": score}).dropna(subset=["score"])
    agg = df.groupby("subject").agg(y=("y", "first"), score=("score", "mean"), n_rows=("y", "size"))
    agg["pred"] = (agg["score"] >= 0.5).astype(int)
    return agg.reset_index()


METRICS = ["balanced_accuracy", "accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]


def subject_metrics(agg: pd.DataFrame) -> dict:
    if agg.empty:
        return {"n_subjects": 0, **{k: np.nan for k in METRICS}}
    y, p, s = agg["y"].to_numpy(), agg["pred"].to_numpy(), agg["score"].to_numpy()
    return {
        "n_subjects": len(agg),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "accuracy": accuracy_score(y, p),
        "sensitivity": recall_score(y, p, pos_label=1, zero_division=0),
        "specificity": recall_score(y, p, pos_label=0, zero_division=0),
        "precision": precision_score(y, p, pos_label=1, zero_division=0),
        "f1": f1_score(y, p, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan,
    }


def tune(model_name, X, y, groups, cv, n_trials, seed, layout) -> dict:
    """Optuna search that maximises subject-level balanced accuracy over `cv`."""
    def objective(trial):
        params = suggest_params(trial, model_name, layout)
        if not active_families(params, layout):
            return 0.0
        score, _ = cross_val_scores(model_name, params, X, y, groups, cv, seed, layout)
        agg = to_subject_level(y, score, groups)
        return balanced_accuracy_score(agg["y"], agg["pred"]) if len(agg) else 0.0

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def _outer_cv(cfg: RunConfig, seed: int):
    if cfg.outer_folds == 0:
        return LeaveOneGroupOut()
    return StratifiedGroupKFold(n_splits=cfg.outer_folds, shuffle=True, random_state=seed)


def permute_subject_labels(y, groups, seed):
    """Shuffle labels between subjects (never within), keeping every subject's rows together."""
    subjects, first = np.unique(groups, return_index=True)
    shuffled = dict(zip(subjects, np.random.default_rng(seed).permutation(y[first])))
    return np.array([shuffled[g] for g in groups])


def run_fold(job_key, X, layout, y, groups, train, test, model_name, cfg, seed):
    """One outer fold: tune on the inner folds of `train`, then score `test`."""
    _configure_worker()
    inner = StratifiedGroupKFold(n_splits=cfg.inner_folds, shuffle=True, random_state=seed)
    best = tune(model_name, X[train], y[train], groups[train], inner, cfg.n_trials, seed, layout)
    out = _fit_predict(model_name, best, X, y, train, test, seed, layout)
    failed = np.unique(groups[test]).tolist() if out is None else []
    return {"job": job_key, "test": test, "score": out, "params": best, "failed": failed}


def run_nonnested(job_key, X, layout, y, groups, model_name, cfg, seed):
    """Tune and score on the same outer folds (biased; only for comparison with the original)."""
    _configure_worker()
    outer = _outer_cv(cfg, seed)
    best = tune(model_name, X, y, groups, outer, cfg.n_trials, seed, layout)
    score, failed = cross_val_scores(model_name, best, X, y, groups, outer, seed, layout)
    return {"job": job_key, "test": np.arange(len(y)), "score": score, "params": best, "failed": failed}


# =============================================================================
# classify: main
# =============================================================================

def _configure_worker():
    """Also called inside joblib worker processes, which don't inherit the parent's filters."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    # Constant columns inside a small training fold are expected; SelectKBest handles their NaN scores.
    warnings.filterwarnings("ignore", message="Features .* are constant", category=UserWarning)
    warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)


def _describe_choices(fold_params: List[dict]) -> str:
    """For auto mode: how often each model / family was picked across outer folds."""
    models = pd.Series([p.get("model") for p in fold_params]).dropna().value_counts()
    fams = pd.Series([k[4:] for p in fold_params for k, v in p.items() if k.startswith("use_") and v]).value_counts()
    parts = []
    if len(models):
        parts.append("models: " + ", ".join(f"{m}x{n}" for m, n in models.items()))
    if len(fams):
        parts.append("families: " + ", ".join(f"{f}x{n}" for f, n in fams.items()))
    return "; ".join(parts)


def _write_summary(out_dir: Path, results_df: pd.DataFrame, cfg: RunConfig, args):
    lines = ["# Classification summary", "",
             f"Data: `{args.data}` | unit: {args.unit} | spatial: {args.spatial} | "
             f"outer: {'LOSO' if cfg.outer_folds == 0 else f'{cfg.outer_folds}-fold'} x {cfg.repeats} repeat(s) | "
             f"inner: {cfg.inner_folds}-fold, {cfg.n_trials} trials | nested: {cfg.nested}", "",
             "Subject-level, out-of-fold metrics (mean ± SD over repeats). Positive class = disease "
             "(FTD for AD vs FTD).", ""]
    cols = ["task", "model", "feature_set", "band", "n_features", "balanced_accuracy", "balanced_accuracy_sd",
            "sensitivity", "specificity", "roc_auc"] + (["perm_p"] if "perm_p" in results_df else []) + ["choices"]
    show = results_df[[c for c in cols if c in results_df]].copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].map(lambda v: f"{v:.3f}")
    lines.append("| " + " | ".join(show.columns) + " |")
    lines.append("|" + "---|" * len(show.columns))
    for _, r in show.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    lines += ["", "## Published subject-level results on ds004504", "",
              "| Study | AD vs CN | FTD vs CN | AD vs FTD |", "|---|---|---|---|"]
    lines += [f"| {s} | {a} | {b} | {c} |" for s, a, b, c in LITERATURE]
    lines += ["", f"{len(results_df)} configurations were compared. If more than one, treat the top row per "
              "task as exploratory: only `auto` rows (or a pre-registered single configuration) give an "
              "unbiased estimate."]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def cmd_classify(args):
    out_dir = args.out_dir or Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out_dir / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(handler)

    missing = [m for m in args.models if m in OPTIONAL_PACKAGES and importlib.util.find_spec(OPTIONAL_PACKAGES[m]) is None]
    if missing:
        raise SystemExit(f"Missing packages for {missing}. Run: pip install -r requirements.txt "
                         f"(or drop them with --models)")

    cfg = RunConfig(n_trials=args.n_trials, inner_folds=args.inner_folds, outer_folds=args.outer_folds,
                    repeats=args.repeats, permutations=args.permutations, nested=not args.no_nested,
                    seed=args.seed)
    (out_dir / "run_config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"}, indent=2))
    if not cfg.nested:
        logger.warning("--no-nested: scores are optimistically biased (tuned and scored on the same folds).")
    if cfg.outer_folds == 0:
        logger.warning("Outer LOSO: holding out one subject shifts the training class balance, and a heavily "
                       "regularised model (e.g. SVM with tiny C) can turn that shift into a perfect score on "
                       "pure noise. Prefer k-fold (--outer-folds 10).")

    if args.n_jobs != 1:
        # One BLAS/OpenMP thread per worker; otherwise N workers x M threads oversubscribe the CPU.
        # Worker processes inherit these when joblib starts them.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
            os.environ.setdefault(var, "1")

    ds = load_dataset(args.data, args)
    if "riemann" in ds.families() and importlib.util.find_spec("pyriemann") is None:
        raise SystemExit("The riemann family needs pyriemann: pip install -r requirements.txt")

    # Build every (feature set, band) design once; jobs share them.
    designs = {}
    for fs, band in itertools.product(args.feature_sets, args.bands):
        designs[(fs, band)] = make_design(ds, parse_feature_set(fs, ds), band)

    jobs, runs = [], {}
    for band, task, model_name, fs in itertools.product(args.bands, args.tasks, args.models, args.feature_sets):
        key = (task, model_name, fs, band)
        X_all, layout, reason = designs[(fs, band)]
        if X_all is None:
            logger.info("Skipped %s: %s", key, reason)
            continue
        mask, y = task_labels(ds.y, task)
        X, groups = X_all[mask], ds.groups[mask]
        seed0 = cfg.seed + zlib.crc32("|".join(key).encode()) % 10_000
        run_list = [("real", r) for r in range(cfg.repeats)] + [("perm", r) for r in range(cfg.permutations)]
        for kind, r in run_list:
            y_run = y if kind == "real" else permute_subject_labels(y, groups, seed0 + 7919 * (r + 1))
            rseed = seed0 + 1000 * r + (500_000 if kind == "perm" else 0)
            job_key = (key, kind, r)
            runs[job_key] = {"y": y_run, "groups": groups, "n_features": X.shape[1]}
            if not cfg.nested:
                jobs.append(delayed(run_nonnested)(job_key, X, layout, y_run, groups, model_name, cfg, rseed))
                continue
            for k, (train, test) in enumerate(_outer_cv(cfg, rseed).split(X, y_run, groups)):
                jobs.append(delayed(run_fold)(job_key, X, layout, y_run, groups, train, test,
                                              model_name, cfg, rseed + k))

    logger.info("%d configurations, %d runs, %d fold jobs (n_jobs=%d) -> %s",
                len({k[0] for k in runs}), len(runs), len(jobs), args.n_jobs, out_dir)
    t0 = time.time()
    fold_results = Parallel(n_jobs=args.n_jobs, verbose=10 if args.n_jobs != 1 else 0)(jobs)
    logger.info("Finished in %.1f min", (time.time() - t0) / 60)

    # Collect fold outputs per run.
    for job_key, meta in runs.items():
        meta["score"] = np.full(len(meta["y"]), np.nan)
        meta["params"], meta["failed"] = [], []
    for res in fold_results:
        meta = runs[res["job"]]
        if res["score"] is not None:
            meta["score"][res["test"]] = res["score"]
        meta["params"].append(res["params"])
        meta["failed"] += res["failed"]

    rows, preds = [], []
    with open(out_dir / "best_params.jsonl", "w") as f:
        for key in dict.fromkeys(k[0] for k in runs):
            task, model_name, fs, band = key
            real = [runs[(key, "real", r)] for r in range(cfg.repeats)]
            per_repeat = [subject_metrics(to_subject_level(m["y"], m["score"], m["groups"])) for m in real]
            row = {"task": task, "model": model_name, "feature_set": fs, "band": band,
                   "n_features": real[0]["n_features"], "n_subjects": per_repeat[0]["n_subjects"]}
            for metric in METRICS:
                vals = np.array([m[metric] for m in per_repeat], dtype=float)
                row[metric] = np.nanmean(vals)
                row[f"{metric}_sd"] = np.nanstd(vals) if len(vals) > 1 else np.nan
            row["n_failed_subjects"] = len(set(itertools.chain.from_iterable(m["failed"] for m in real)))
            all_params = [p for m in real for p in m["params"]]
            row["choices"] = _describe_choices(all_params)
            if cfg.permutations:
                null = np.array([subject_metrics(to_subject_level(m["y"], m["score"], m["groups"]))["balanced_accuracy"]
                                 for m in (runs[(key, "perm", r)] for r in range(cfg.permutations))])
                row["perm_null_mean"] = float(np.nanmean(null))
                row["perm_null_values"] = json.dumps(np.round(null, 4).tolist())
                row["perm_p"] = (1 + int(np.sum(null >= row["balanced_accuracy"]))) / (len(null) + 1)
            rows.append(row)

            mean_score = np.nanmean(np.vstack([m["score"] for m in real]), axis=0)
            preds.append(to_subject_level(real[0]["y"], mean_score, real[0]["groups"]).assign(
                task=task, model=model_name, feature_set=fs, band=band))
            f.write(json.dumps({"task": task, "model": model_name, "feature_set": fs, "band": band,
                                "fold_params": all_params}, default=str) + "\n")
            logger.info("%s | %s | %s | %s: bal_acc=%.3f±%.3f sens=%.3f spec=%.3f auc=%.3f%s%s",
                        task, model_name, fs, band, row["balanced_accuracy"],
                        np.nan_to_num(row["balanced_accuracy_sd"]), row["sensitivity"], row["specificity"],
                        row["roc_auc"], f" perm_p={row['perm_p']:.3f}" if cfg.permutations else "",
                        f" [{row['choices']}]" if row["choices"] else "")

    if not rows:
        logger.error("Nothing was evaluated. Check the skip reasons above.")
        return
    results_df = pd.DataFrame(rows).sort_values(["task", "balanced_accuracy"], ascending=[True, False])
    results_df.to_csv(out_dir / "results.csv", index=False)
    pd.concat(preds).to_csv(out_dir / "subject_predictions.csv", index=False)
    _write_summary(out_dir, results_df, cfg, args)
    logger.info("Saved results.csv, summary.md, subject_predictions.csv, best_params.jsonl to %s", out_dir)


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="Fetch OpenNeuro ds004504 (preprocessed EEG, ~3 GB)")
    d.add_argument("--out", type=Path, default=Path("data/ds004504"))
    d.add_argument("--subjects", type=int, nargs="+", help="subject numbers (default: all 88)")
    d.add_argument("--n-jobs", type=int, default=8, help="parallel downloads")
    d.set_defaults(func=cmd_download)

    e = sub.add_parser("extract", help="Raw EEG -> features .npz")
    e.add_argument("--bids-root", type=Path, default=Path("data/ds004504"))
    e.add_argument("--out", type=Path, default=Path("features_ds004504.npz"))
    e.add_argument("--cache-dir", type=Path, default=None, help="default: <bids-root>/features_cache")
    e.add_argument("--subjects", type=int, nargs="+", help="subject numbers (default: all)")
    e.add_argument("--n-jobs", type=int, default=-1)
    e.set_defaults(func=cmd_extract)

    c = sub.add_parser("classify", help="Nested, subject-grouped classification")
    c.add_argument("--data", type=Path, default=Path("features_ds004504.npz"))
    c.add_argument("--out-dir", type=Path, default=None, help="default: results/<timestamp>")
    c.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    c.add_argument("--models", nargs="+", choices=MODELS + ["auto"], default=["auto"],
                   help="'auto' searches model + feature families inside the inner loop (unbiased headline)")
    c.add_argument("--feature-sets", nargs="+", default=["all"],
                   help="'all' or families joined with '+', e.g. spectral connectivity spectral+riemann. "
                        "Families: spectral complexity connectivity riemann (new data), spectral cpte (original .npz)")
    c.add_argument("--bands", nargs="+", choices=sorted(BAND_TAGS) + ["all"], default=["all"],
                   help="'all' uses every feature; a band keeps only that band's features ('alpha' = alpha1+alpha2)")
    c.add_argument("--unit", choices=["subject", "epoch"], default="subject",
                   help="subject: median of each feature over epochs (one row per subject); "
                        "epoch: epoch rows, probabilities averaged per subject")
    c.add_argument("--max-epochs", type=int, default=40, help="epoch unit only: random epochs kept per subject")
    c.add_argument("--spatial", choices=["channels", "regions"], default="channels",
                   help="regions: average features over frontal/temporal/central/parietal/occipital")
    c.add_argument("--rbp-layout", choices=["band-major", "channel-major"], default="band-major",
                   help="original .npz only: column order of the 95 RBP features")
    c.add_argument("--n-trials", type=int, default=30, help="Optuna trials per tuning run")
    c.add_argument("--inner-folds", type=int, default=5)
    c.add_argument("--outer-folds", type=int, default=10, help="0 = leave-one-subject-out")
    c.add_argument("--repeats", type=int, default=1, help="repeated outer CV with different splits (3-5 recommended)")
    c.add_argument("--permutations", type=int, default=0,
                   help="label permutations for a p-value (each costs one full run; 100+ for a paper)")
    c.add_argument("--no-nested", action="store_true",
                   help="Tune and score on the same folds (biased; only for comparison with the original)")
    c.add_argument("--n-jobs", type=int, default=1, help="parallel fold jobs (-1 = all cores)")
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--quick", action="store_true",
                   help="Smoke test: AD_CN, LogReg + SVM, all features, 3 trials, 3x3 folds")
    c.set_defaults(func=cmd_classify)

    args = ap.parse_args(argv)
    if args.command == "classify" and args.quick:
        args.tasks, args.models, args.feature_sets, args.bands = ["AD_CN"], ["LogReg", "SVM"], ["all"], ["all"]
        args.n_trials, args.inner_folds, args.outer_folds, args.repeats = 3, 3, 3, 1
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    _configure_worker()
    args.func(args)


if __name__ == "__main__":
    main()

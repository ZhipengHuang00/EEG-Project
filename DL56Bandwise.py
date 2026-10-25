# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'eeg_classification_ml_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# --- Feature Indexing Configuration ---
# Assuming features are ordered with RBP first.
# RBP: 5 bands * 19 channels = 95 features
# Any additional columns AFTER the first 95 RBP features in `features` will be treated as CPTE.
RBP_FEATURE_COUNT = 95

# --- Part 1: Data Preprocessing ---

class EEGDataPreprocessor:
    """Handles EEG feature loading and initial splitting."""
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load EEG features, labels, and subject IDs from an NPZ file.
        Splits combined features into RBP (first 95) and Network (next 105) components.
        Any additional columns beyond RBP+Network are treated as CPTE features. If the NPZ file
        also contains CPTE arrays (e.g., 'cpte_features' or 'cpte'), those are concatenated to
        the CPTE block.
        """
        try:
            logger.info(f"Loading features from {self.data_path}")
            data = np.load(self.data_path, allow_pickle=True)
            X_combined, y, subject_ids = data['features'], data['labels'], data['subject_ids']

            # Optional CPTE feature blocks from file
            X_cpte_file = None
            if 'cpte_features' in data:
                X_cpte_file = data['cpte_features']
            elif 'cpte' in data:
                X_cpte_file = data['cpte']

            # Ensure data types are consistent and numeric
            X_combined = X_combined.astype(np.float64)
            y = y.astype(int)
            subject_ids = subject_ids.astype(str)
            if X_cpte_file is not None:
                X_cpte_file = X_cpte_file.astype(np.float64)

            # --- RBP slicing + CPTE detection (NO Network) ---
            if X_combined.shape[1] < RBP_FEATURE_COUNT:
                logger.error(
                    f"Insufficient features: expected at least {RBP_FEATURE_COUNT} RBP features but got {X_combined.shape[1]}."
                )
                rbp_features = X_combined
                cpte_from_combined = np.empty((X_combined.shape[0], 0))
            else:
                # First 95 = RBP; everything after is CPTE
                rbp_features = X_combined[:, :RBP_FEATURE_COUNT]
                cpte_from_combined = X_combined[:, RBP_FEATURE_COUNT:]

            # Network features have been REMOVED
            network_features = np.empty((X_combined.shape[0], 0))

            # Merge CPTE from combined tail with any CPTE arrays stored in file
            if X_cpte_file is not None and X_cpte_file.size > 0:
                if cpte_from_combined.size > 0:
                    X_cpte = np.hstack([cpte_from_combined, X_cpte_file])
                else:
                    X_cpte = X_cpte_file
            else:
                X_cpte = cpte_from_combined if cpte_from_combined.size > 0 else np.empty((X_combined.shape[0], 0))

            # Track CPTE column count for logging
            self.last_cpte_cols = X_cpte.shape[1]

            self._log_data_info(X_combined, rbp_features, network_features, y)
            return X_combined, rbp_features, network_features, X_cpte, y, subject_ids
        except Exception as e:
            logger.error(f"Error loading data from {self.data_path}: {e}")
            raise

    def _log_data_info(self, X_combined: np.ndarray, rbp_features: np.ndarray, network_features: np.ndarray, y: np.ndarray) -> None:
        """Log dataset statistics"""
        logger.info(f"Total number of subjects: {X_combined.shape[0]}")
        logger.info(f"Total combined features: {X_combined.shape[1]}")
        logger.info(f"  RBP features: {rbp_features.shape[1]} (expected {RBP_FEATURE_COUNT})")
        logger.info(f"  Network features: {network_features.shape[1]} (REMOVED)")
        logger.info(f"  CPTE features: {getattr(self, 'last_cpte_cols', 'unknown')}")

        labels, counts = np.unique(y, return_counts=True)
        label_names = {0: 'AD', 1: 'CN', 2: 'FTD'} # Assuming these are your class mappings
        logger.info("Label distribution:")
        for label, count in zip(labels, counts):
            logger.info(f"  {label_names.get(label, f'Unknown ({label})')}: {count} subjects")

        # Log NaN/Inf counts *before* imputation in the CV loop
        logger.info(f"NaN values in raw loaded combined features: {np.isnan(X_combined).sum()}")
        logger.info(f"Inf values in raw loaded combined features: {np.isinf(X_combined).sum()}")
        
        # Replace inf with nan for statistics calculation if any are present
        X_for_stats = X_combined.copy()
        X_for_stats[np.isinf(X_for_stats)] = np.nan
        
        logger.info(f"Combined Feature stats (after converting Inf to NaN for stats): "
                    f"Mean={np.nanmean(X_for_stats):.6f}, Std={np.nanstd(X_for_stats):.6f}, "
                    f"Min={np.nanmin(X_for_stats):.6f}, Max={np.nanmax(X_for_stats):.6f}")

# --- Part 2: ML Classification ---

class EEGClassifier:
    """Handles ML classification with Optuna optimization, including feature set selection."""
    def __init__(
        self,
        X_combined: np.ndarray,
        X_rbp: np.ndarray,
        X_network: np.ndarray,
        X_cpte: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        window_seconds: int = 10,
        connectivity_method: str = 'phase_locking',
        threshold: float = 0.5
    ):
        self.X_combined = X_combined
        self.X_rbp = X_rbp
        self.X_network = X_network
        self.X_cpte = X_cpte
        self.y = y
        self.subject_ids = subject_ids
        # Store feature extraction parameters for reproducibility/logging
        self.window_seconds = window_seconds
        self.connectivity_method = connectivity_method
        self.threshold = threshold

        # Infer CPTE band size if per-band (5 bands). If not divisible by 5, disable band slicing for CPTE.
        self.cpte_band_size = None
        if X_cpte is not None and X_cpte.shape[1] > 0 and X_cpte.shape[1] % 5 == 0:
            self.cpte_band_size = X_cpte.shape[1] // 5

        self.model_configurations = [
            ('AD_CN', 'RandomForest'), ('AD_CN', 'SVM'), ('AD_CN', 'XGBoost'), ('AD_CN', 'CatBoost'),
            ('AD_CN', 'KNN'), ('AD_CN', 'MLP'), ('AD_CN', 'LightGBM'),
            ('FTD_CN', 'RandomForest'), ('FTD_CN', 'SVM'), ('FTD_CN', 'XGBoost'), ('FTD_CN', 'CatBoost'),
            ('FTD_CN', 'KNN'), ('FTD_CN', 'MLP'), ('FTD_CN', 'LightGBM'),
            ('AD_FTD', 'RandomForest'), ('AD_FTD', 'SVM'), ('AD_FTD', 'XGBoost'), ('AD_FTD', 'CatBoost'),
            ('AD_FTD', 'KNN'), ('AD_FTD', 'MLP'), ('AD_FTD', 'LightGBM')
        ]
        self.feature_set_choices = ['rbp_only', 'cpte_only', 'network_only', 'combined']
        self.band_choices = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    def create_datasets(self) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Create classification datasets for binary tasks (AD vs CN, FTD vs CN, AD vs FTD),
        for each feature set type (RBP, CPTE, Network, Combined).
        """
        datasets = {}

        # Masks for each task
        ad_cn_mask = (self.y == 0) | (self.y == 1)
        ftd_cn_mask = (self.y == 2) | (self.y == 1)
        ad_ftd_mask = (self.y == 0) | (self.y == 2)

        # Helper to create dataset structure for a given mask and optional label mapping
        def _get_task_data(features, labels, s_ids, mask=None, label_map: Optional[Dict[int, int]] = None):
            if mask is not None:
                features_task = features[mask]
                labels_task = labels[mask]
                s_ids_task = s_ids[mask]
            else:
                features_task = features
                labels_task = labels
                s_ids_task = s_ids

            if label_map is not None:
                labels_task_remapped = labels_task.copy()
                for old, new in label_map.items():
                    labels_task_remapped[labels_task_remapped == old] = new
                labels_task = labels_task_remapped

            return {'features': features_task, 'labels': labels_task, 'subject_ids': s_ids_task}

        task_specs = {
            'AD_CN':   (ad_cn_mask, None),                 # AD=0, CN=1 already matches desired mapping
            'FTD_CN':  (ftd_cn_mask, {2: 0, 1: 1}),        # FTD->0, CN->1
            'AD_FTD':  (ad_ftd_mask, {0: 0, 2: 1})         # AD->0, FTD->1
        }

        for task_name, (current_mask, label_map) in task_specs.items():
            datasets[task_name] = {}
            datasets[task_name]['rbp_only'] = _get_task_data(self.X_rbp, self.y, self.subject_ids, current_mask, label_map)
            datasets[task_name]['cpte_only'] = _get_task_data(self.X_cpte, self.y, self.subject_ids, current_mask, label_map)
            datasets[task_name]['network_only'] = _get_task_data(self.X_network, self.y, self.subject_ids, current_mask, label_map)
            # For 'combined', concatenate RBP, Network, and CPTE features
            combined_features = np.concatenate(
                [self.X_rbp, self.X_network, self.X_cpte], axis=1
            )
            datasets[task_name]['combined'] = _get_task_data(combined_features, self.y, self.subject_ids, current_mask, label_map)

        return datasets

    # --- Band index mapping for slicing (class constant) ---
    BAND_TO_IDX = {'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3, 'gamma': 4}

    def _slice_rbp_by_band(self, X_rbp_task: np.ndarray, band: str) -> np.ndarray:
        """Slice RBP features for a specific frequency band. Assumes 5 bands * 19 channels = 95 features."""
        if X_rbp_task is None or X_rbp_task.shape[1] == 0:
            return np.empty((X_rbp_task.shape[0], 0)) if X_rbp_task is not None else np.empty((0, 0))
        per_band = 19
        band_idx = self.BAND_TO_IDX.get(band, 0)
        start = band_idx * per_band
        end = start + per_band
        end = min(end, X_rbp_task.shape[1])
        if start >= X_rbp_task.shape[1]:
            # No columns for this band
            return np.empty((X_rbp_task.shape[0], 0))
        return X_rbp_task[:, start:end]

    def _slice_cpte_by_band(self, X_cpte_task: np.ndarray, band: str) -> np.ndarray:
        """Return the CPTE features for a specific band if band size is inferable; otherwise return all CPTE."""
        if self.cpte_band_size is None or X_cpte_task.shape[1] == 0:
            if X_cpte_task.shape[1] > 0:
                logger.warning("CPTE per-band size unknown; using all CPTE features for every band.")
            return X_cpte_task
        band_idx = self.BAND_TO_IDX.get(band, 0)
        start = band_idx * self.cpte_band_size
        end = start + self.cpte_band_size
        if X_cpte_task.shape[1] < end:
            logger.warning(f"Requested CPTE band slice [{start}:{end}] exceeds available columns {X_cpte_task.shape[1]}. Using available columns only.")
            end = min(end, X_cpte_task.shape[1])
        return X_cpte_task[:, start:end]


    def evaluate_model_ml(self, model: Any, X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray, use_pca: bool = False, pca_variance: float = 0.95) -> Tuple[float, float, float, float]:
        """
        Evaluate ML model using Leave-One-Subject-Out Cross-Validation.
        Applies imputation, scaling, and (optionally) PCA within each CV fold to prevent data leakage.
        """
        y_true_all, y_pred_all = [], []
        unique_subjects = np.unique(subject_ids)

        if len(unique_subjects) < 2:
            logger.warning("Not enough unique subjects for Leave-One-Subject-Out Cross-Validation. Returning 0 scores.")
            return 0.0, 0.0, 0.0, 0.0

        for test_subject_id in unique_subjects:
            train_mask = (subject_ids != test_subject_id)
            test_mask = (subject_ids == test_subject_id)

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            # --- Critical: Handle edge cases for data splits ---
            if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                logger.warning(f"Skipping fold for subject {test_subject_id}: Empty train or test set after split.")
                y_true_all.extend(y_test.tolist() if y_test.size > 0 else [])
                y_pred_all.extend([np.nan] * y_test.shape[0])
                continue

            if len(np.unique(y_train)) < 2:
                logger.warning(f"Skipping fold for subject {test_subject_id}: Training set has less than 2 classes. Cannot fit model.")
                y_true_all.extend(y_test.tolist())
                y_pred_all.extend([np.nan] * y_test.shape[0])
                continue

            # --- Imputation, Scaling, and Optional PCA within the CV loop ---
            # Ensure no Infs before imputation by converting to NaN first
            X_train_processed = np.nan_to_num(X_train, posinf=np.nan, neginf=np.nan)
            X_test_processed = np.nan_to_num(X_test, posinf=np.nan, neginf=np.nan)

            if X_train_processed.shape[1] == 0:
                logger.warning(f"Skipping fold for subject {test_subject_id}: X_train_processed has 0 features after processing. Cannot impute.")
                y_true_all.extend(y_test.tolist())
                y_pred_all.extend([np.nan] * y_test.shape[0])
                continue

            imputer = SimpleImputer(strategy='mean')
            scaler = StandardScaler()

            X_train_imputed = imputer.fit_transform(X_train_processed)
            X_test_imputed = imputer.transform(X_test_processed)

            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_test_scaled = scaler.transform(X_test_imputed)

            if use_pca:
                pca = PCA(n_components=pca_variance, svd_solver='full', random_state=42)
                X_train_scaled = pca.fit_transform(X_train_scaled)
                X_test_scaled = pca.transform(X_test_scaled)

            # Fit model and predict
            try:
                model.fit(X_train_scaled, y_train)
                y_pred_fold = model.predict(X_test_scaled)
                y_true_all.extend(y_test.tolist())
                y_pred_all.extend(y_pred_fold.tolist())
            except Exception as e:
                logger.error(f"Error during model fitting or prediction for subject {test_subject_id}: {e}. Appending NaN predictions.")
                y_true_all.extend(y_test.tolist())
                y_pred_all.extend([np.nan] * y_test.shape[0])

        # Ensure arrays are 1D before filtering
        y_true_all = np.ravel(y_true_all)
        y_pred_all = np.ravel(y_pred_all)
        # Filter out NaN predictions for metric calculation
        valid_indices = ~np.isnan(y_pred_all)
        y_true_filtered = np.asarray(y_true_all)[valid_indices]
        y_pred_filtered = np.asarray(y_pred_all)[valid_indices]

        if not y_true_filtered.size:  # Check if there are any valid predictions
            logger.warning("No valid predictions to calculate metrics. Returning 0 scores.")
            return 0.0, 0.0, 0.0, 0.0

        # Handle potential single-class predictions in multi-class context for metrics
        unique_true = np.unique(y_true_filtered)
        unique_pred = np.unique(y_pred_filtered)

        average_method = 'weighted'
        if len(unique_true) == 2 and len(unique_pred) == 2 and (unique_true == unique_pred).all():
            average_method = 'binary'
        elif len(unique_true) > 2:
            average_method = 'macro'

        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_filtered, y_pred_filtered, average=average_method, zero_division=0
            )
        except ValueError as e:
            logger.error(f"Error calculating precision/recall/f1 with {average_method} average: {e}. Trying 'macro'.")
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true_filtered, y_pred_filtered, average='macro', zero_division=0
                )
            except Exception as e_macro:
                logger.error(f"Failed to calculate metrics even with 'macro' average: {e_macro}. Returning 0s.")
                precision, recall, f1 = 0.0, 0.0, 0.0

        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)

        return accuracy, precision, recall, f1

    def objective_wrapper(self, trial: optuna.Trial, task_name: str, model_type: str, datasets: Dict) -> float:
        """
        Wrapper objective function for Optuna to select feature type
        and then call the specific model's objective.
        """
        # Suggest which feature set to use for this trial
        feature_set_type = trial.suggest_categorical('feature_set_type', list(self.feature_set_choices))
        # For backward compatibility, use 'delta' as default band
        return self.objective_wrapper_fixed(trial, task_name, model_type, datasets, feature_set_type, band='delta')
    def objective_wrapper_fixed(self, trial: optuna.Trial, task_name: str, model_type: str, datasets: Dict, feature_set_type: str, band: str) -> float:
        """
        Objective function for Optuna with fixed feature_set_type.
        Does NOT call trial.suggest_categorical for feature_set_type.
        Also includes feature extraction parameters as hyperparameters.
        """
        # Suggest feature extraction parameters (even if not used in this pipeline, for logging/repro)
        window_seconds = trial.suggest_categorical('window_seconds', [4, 6, 8, 10, 12])
        # Ensure window_seconds is always an integer
        if not isinstance(window_seconds, int):
            logger.warning(f"window_seconds is not int, got {type(window_seconds)}. Forcing to int.")
            window_seconds = int(window_seconds)
        connectivity_method = trial.suggest_categorical('connectivity_method', ['pearson', 'phase_locking', 'wpli_debiased'])
        threshold = trial.suggest_float('threshold', 0.3, 0.8, step=0.1)
        # Add new Optuna hyperparameters for class_weight, use_pca
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
        use_pca = trial.suggest_categorical('use_pca', [True, False])
        pca_variance = trial.suggest_float('pca_variance', 0.85, 0.999) if use_pca else 0.95
        frequency_band = trial.suggest_categorical('frequency_band', ['delta', 'theta', 'alpha', 'beta', 'gamma'])
        # Robust feature_params dictionary including all relevant params
        feature_params = {
            'window_seconds': window_seconds,
            'connectivity_method': connectivity_method,
            'threshold': threshold,
            'class_weight': class_weight,
            'use_pca': use_pca,
            'pca_variance': pca_variance,
            'frequency_band': frequency_band
        }
        logger.info(f"Trial {trial.number}: Feature params: {feature_params}")

        # Get the corresponding data for the chosen task and feature set
        # For new feature set types, handle band slicing
        datasets_task = datasets[task_name]
        X_rbp_task_full = datasets_task['rbp_only']['features']
        X_cpte_task_full = datasets_task['cpte_only']['features']
        X_network_task_full = datasets_task['network_only']['features']
        y_task = datasets_task['rbp_only']['labels']  # All have same labels
        subject_ids_task = datasets_task['rbp_only']['subject_ids']
        # Feature slicing by type
        if feature_set_type == 'rbp_only':
            X_task = self._slice_rbp_by_band(X_rbp_task_full, band)
        elif feature_set_type == 'cpte_only':
            X_task = self._slice_cpte_by_band(X_cpte_task_full, band)
        elif feature_set_type == 'network_only':
            X_task = X_network_task_full
        elif feature_set_type == 'combined':
            # Concatenate RBP, Network, and CPTE (apply band slicing to RBP and CPTE, use all network)
            X_rbp_band = self._slice_rbp_by_band(X_rbp_task_full, band)
            X_cpte_band = self._slice_cpte_by_band(X_cpte_task_full, band)
            X_task = np.concatenate([X_rbp_band, X_network_task_full, X_cpte_band], axis=1)
        else:
            raise ValueError(f"Unknown feature_set_type: {feature_set_type}")

        if X_task.shape[1] == 0:
            logger.warning(f"Trial {trial.number}: Selected feature set '{feature_set_type}' for task '{task_name}' resulted in 0 features. Returning 0.0.")
            return 0.0

        if model_type == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
                'max_depth': trial.suggest_int('rf_max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 4)
            }
            # Add class_weight param to RandomForestClassifier
            model = RandomForestClassifier(**params, class_weight=class_weight, random_state=42, n_jobs=-1)
        elif model_type == 'SVM':
            C = trial.suggest_float('svm_C', 1e-2, 1e2, log=True)
            kernel = trial.suggest_categorical('svm_kernel', ['rbf', 'linear'])
            params = {
                'C': C,
                'kernel': kernel,
                'random_state': 42
            }
            if kernel == 'rbf':
                params['gamma'] = trial.suggest_categorical('svm_gamma', ['scale', 'auto'])
            # Add class_weight param to SVC
            params['class_weight'] = class_weight
            model = SVC(**params)
        elif model_type == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('cat_iterations', 50, 200),
                'depth': trial.suggest_int('cat_depth', 3, 10),
                'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1.0, 10.0)
            }
            model = CatBoostClassifier(**params, verbose=0, random_seed=42)
        elif model_type == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 200),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0)
            }
            model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
        elif model_type == 'KNN':
            params = {
                'n_neighbors': trial.suggest_int('knn_n_neighbors', 3, 15),
                'weights': trial.suggest_categorical('knn_weights', ['uniform', 'distance']),
                'p': trial.suggest_int('knn_p', 1, 2)
            }
            model = KNeighborsClassifier(**params)
        elif model_type == 'MLP':
            params = {
                'hidden_layer_sizes': trial.suggest_categorical('mlp_hidden_layer_sizes', list([(50,), (100,), (100, 50)])),
                'activation': trial.suggest_categorical('mlp_activation', ['relu', 'tanh']),
                'alpha': trial.suggest_float('mlp_alpha', 1e-5, 1e-1, log=True),
                'learning_rate': trial.suggest_categorical('mlp_learning_rate', ['constant', 'adaptive']),
                'max_iter': 300
            }
            # Remove class_weight from params if present and do not pass to MLPClassifier
            params.pop('class_weight', None)
            model = MLPClassifier(**params, random_state=42)
        elif model_type == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
                'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 100)
            }
            model = lgb.LGBMClassifier(**params, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Evaluate model
        _, _, _, f1 = self.evaluate_model_ml(model, X_task, y_task, subject_ids_task, use_pca=use_pca, pca_variance=pca_variance)
        # Optionally, store feature_params in trial.user_attrs for later retrieval
        trial.set_user_attr('feature_params', feature_params)
        return f1


    def run_pipeline(self, n_trials: int = 20) -> Dict:
        """
        Run classification pipeline for all model-task-feature_set combinations,
        optimizing hyperparameters for each, and compare all combinations.
        """
        datasets = self.create_datasets()
        final_results = {}
        for band in self.band_choices:
            for task_name in ['AD_CN', 'FTD_CN', 'AD_FTD']:
                for model_type in ['RandomForest', 'SVM', 'XGBoost', 'CatBoost', 'KNN', 'MLP', 'LightGBM']:
                    for feature_set_type in self.feature_set_choices:
                        logger.info(f"Running optimization for Task: {task_name}, Model: {model_type}, Feature Set: {feature_set_type}, Band: {band}")
                        # Unique Optuna study name for each combination
                        study_name = f"{task_name}_{model_type}_{feature_set_type}_{band}_study"
                        # Define sampler and pruner for reproducibility and pruning
                        sampler = TPESampler(seed=42, multivariate=True, group=True)
                        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0, interval_steps=1)
                        study = optuna.create_study(direction='maximize', study_name=study_name, storage='sqlite:///eeg_optuna_studies.db', load_if_exists=True, sampler=sampler, pruner=pruner)
                        # Define local objective with fixed feature_set_type and band
                        def fixed_feature_objective(trial):
                            return self.objective_wrapper_fixed(trial, task_name, model_type, datasets, feature_set_type, band)
                        study.optimize(fixed_feature_objective, n_trials=n_trials, show_progress_bar=True)
                        best_params = study.best_params.copy()
                        # Normalize Optuna trial keys to model parameter names
                        if model_type == 'RandomForest':
                            best_params = {
                                'n_estimators': best_params.get('rf_n_estimators', 100),
                                'max_depth': best_params.get('rf_max_depth', None),
                                'min_samples_split': best_params.get('rf_min_samples_split', 2),
                                'min_samples_leaf': best_params.get('rf_min_samples_leaf', 1)
                            }
                        elif model_type == 'SVM':
                            best_params = {
                                'C': best_params.get('svm_C', 1.0),
                                'kernel': best_params.get('svm_kernel', 'rbf'),
                                'gamma': best_params.get('svm_gamma', 'scale')
                            }
                        elif model_type == 'CatBoost':
                            best_params = {
                                'iterations': best_params.get('cat_iterations', 100),
                                'depth': best_params.get('cat_depth', 6),
                                'learning_rate': best_params.get('cat_learning_rate', 0.1),
                                'l2_leaf_reg': best_params.get('cat_l2_leaf_reg', 3.0)
                            }
                        elif model_type == 'XGBoost':
                            best_params = {
                                'n_estimators': best_params.get('xgb_n_estimators', 100),
                                'max_depth': best_params.get('xgb_max_depth', 6),
                                'learning_rate': best_params.get('xgb_learning_rate', 0.1),
                                'subsample': best_params.get('xgb_subsample', 0.8),
                                'colsample_bytree': best_params.get('xgb_colsample_bytree', 0.8)
                            }
                        elif model_type == 'KNN':
                            best_params = {
                                'n_neighbors': best_params.get('knn_n_neighbors', 5),
                                'weights': best_params.get('knn_weights', 'uniform'),
                                'p': best_params.get('knn_p', 2)
                            }
                        elif model_type == 'MLP':
                            best_params = {
                                'hidden_layer_sizes': best_params.get('mlp_hidden_layer_sizes', (100,)),
                                'activation': best_params.get('mlp_activation', 'relu'),
                                'alpha': best_params.get('mlp_alpha', 0.0001),
                                'learning_rate': best_params.get('mlp_learning_rate', 'constant'),
                                'max_iter': 300
                            }
                        elif model_type == 'LightGBM':
                            best_params = {
                                'n_estimators': best_params.get('lgb_n_estimators', 100),
                                'learning_rate': best_params.get('lgb_learning_rate', 0.1),
                                'max_depth': best_params.get('lgb_max_depth', 5),
                                'num_leaves': best_params.get('lgb_num_leaves', 31)
                            }
                        # Re-initialize the model with the best found hyperparameters
                        # Reconstruct band-wise features for final evaluation to match the study's band
                        X_source_rbp = datasets[task_name]['rbp_only']['features']
                        X_source_cpte = datasets[task_name]['cpte_only']['features']
                        X_source_network = datasets[task_name]['network_only']['features']

                        if feature_set_type == 'rbp_only':
                            X_best_features = self._slice_rbp_by_band(X_source_rbp, band)
                        elif feature_set_type == 'cpte_only':
                            X_best_features = self._slice_cpte_by_band(X_source_cpte, band)
                        elif feature_set_type == 'network_only':
                            X_best_features = X_source_network
                        elif feature_set_type == 'combined':
                            X_rbp_band = self._slice_rbp_by_band(X_source_rbp, band)
                            X_cpte_band = self._slice_cpte_by_band(X_source_cpte, band)
                            X_best_features = np.concatenate([X_rbp_band, X_source_network, X_cpte_band], axis=1)
                        else:
                            X_best_features = datasets[task_name][feature_set_type]['features']
                        y_task = datasets[task_name]['rbp_only']['labels']
                        subject_ids_task = datasets[task_name]['rbp_only']['subject_ids']
                        if model_type == 'RandomForest':
                            final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
                        elif model_type == 'SVM':
                            final_model = SVC(**best_params)
                        elif model_type == 'CatBoost':
                            final_model = CatBoostClassifier(**best_params, verbose=0, random_seed=42)
                        elif model_type == 'XGBoost':
                            final_model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
                        elif model_type == 'KNN':
                            final_model = KNeighborsClassifier(**best_params)
                        elif model_type == 'MLP':
                            final_model = MLPClassifier(**best_params, random_state=42)
                        elif model_type == 'LightGBM':
                            final_model = lgb.LGBMClassifier(**best_params, random_state=42)
                        else:
                            raise ValueError(f"Unknown model type: {model_type}")
                        # Retrieve feature extraction hyperparameters from best trial (if present)
                        feature_params = study.best_trial.user_attrs.get('feature_params', {
                            'window_seconds': getattr(self, 'window_seconds', 10),
                            'connectivity_method': getattr(self, 'connectivity_method', 'phase_locking'),
                            'threshold': getattr(self, 'threshold', 0.5),
                            'frequency_band': band,
                            'use_pca': False,
                            'pca_variance': 0.95
                        })
                        use_pca_final = feature_params.get('use_pca', False)
                        pca_variance_final = feature_params.get('pca_variance', 0.95)

                        # Evaluate the best model with the best feature set on the full dataset with LOSO and (optional) PCA
                        accuracy, precision, recall, f1 = self.evaluate_model_ml(
                            final_model, X_best_features, y_task, subject_ids_task,
                            use_pca=use_pca_final, pca_variance=pca_variance_final
                        )
                        logger.info(f"Best feature extraction params for {study_name}: {feature_params}")
                        final_results[(task_name, model_type, feature_set_type, band)] = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'best_params': best_params,
                            'best_feature_set_type': feature_set_type,
                            'hyperparameters': feature_params
                        }
                        logger.info(f"Results for {task_name}_{model_type}_{feature_set_type}_{band}: Accuracy={accuracy:.4f}, F1={f1:.4f}, "
                                    f"Precision={precision:.4f}, Recall={recall:.4f}, Feature Params={feature_params}")
        return final_results

def main():
    """Main execution block"""
    try:
        data_path = 'eeg_features_combined_phase_locking_10s_50.npz'
        preprocessor = EEGDataPreprocessor(data_path)
        # load_data now returns combined, rbp, network, cpte features, labels, and subject_ids
        X_combined, X_rbp, X_network, X_cpte, y, subject_ids = preprocessor.load_data()
        
        # Pass all feature sets to the classifier
        classifier = EEGClassifier(X_combined, X_rbp, X_network, X_cpte, y, subject_ids)
        results = classifier.run_pipeline(n_trials=50)
        
        # Save results to CSV for all combinations (now includes band)
        results_df = pd.DataFrame([
            {
                'Task': task,
                'Model': model_type,
                'Feature_Set': feature_set,
                'Band': band,
                'Accuracy': res['accuracy'],
                'Precision': res['precision'],
                'Recall': res['recall'],
                'F1_Score': res['f1_score'],
                'Best_Params': str(res['best_params']),
                'Feature_Params': str(res.get('hyperparameters', {}))
            }
            for (task, model_type, feature_set, band), res in results.items()
        ])
        results_df.to_csv('eeg_classification_results_with_feature_selection.csv', index=False)
        logger.info("Saved results to eeg_classification_results_with_feature_selection.csv")

        logger.info("\n--- Final Classification Results (All Combinations) ---")
        for (task, model_type, feature_set, band), res in results.items():
            logger.info(f"\nConfiguration: Task={task}, Model={model_type}, Feature Set={feature_set}, Band={band}")
            logger.info(f"  Accuracy: {res['accuracy']:.4f}")
            logger.info(f"  Precision: {res['precision']:.4f}")
            logger.info(f"  Recall: {res['recall']:.4f}")
            logger.info(f"  F1-Score: {res['f1_score']:.4f}")
            logger.info(f"  Best Parameters: {res['best_params']}")
            logger.info(f"  Feature Extraction Hyperparameters: {res.get('hyperparameters', {})}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
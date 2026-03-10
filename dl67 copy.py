# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import logging
from datetime import datetime
import warnings
from typing import Tuple, Dict, Any, Optional
import multiprocessing
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# --- Environment Setup and Logging ---
# Ensure BLAS/OpenMP libraries are limited to 1 thread per worker process for stable parallelism
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'eeg_classification_allbands_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# --- Feature counts ---
RBP_FEATURE_COUNT = 95
NETWORK_FEATURE_COUNT = 110


def sanitize_features(X: np.ndarray) -> np.ndarray:
    """Safeguards against NaN, Inf, and extremely large values."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    return X


# --- Data Preprocessor ---
class EEGDataPreprocessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.last_cpte_cols = 0

    def load_data(self):
        try:
            data = np.load(self.data_path, allow_pickle=True)
            X_combined = data['features'].astype(np.float64)
            y = data['labels'].astype(int)
            subject_ids = data['subject_ids'].astype(str)

            # --- Load CPTE features from SepehrF2.csv for alignment ---
            # NOTE: Assuming 'SepehrF2.csv' is in the current directory
            cpte_df = pd.read_csv('SepehrF2.csv')
            cpte_subject_ids = cpte_df['subject_id'].astype(str).values
            
            # Drop non-feature columns
            cpte_features_df = cpte_df.drop(columns=['subject_id', 'label'], errors='ignore').astype(np.float64)
            
            cpte_features_aligned = []
            for sid in subject_ids:
                mask = cpte_subject_ids == sid
                if np.any(mask):
                    cpte_features_aligned.append(cpte_features_df[mask].values[0])
                else:
                    # If subject id not found, fill with NaNs (Imputer will handle this later)
                    cpte_features_aligned.append(np.full(cpte_features_df.shape[1], np.nan))
            X_cpte = np.array(cpte_features_aligned, dtype=np.float64)
            self.last_cpte_cols = X_cpte.shape[1]

            # Sanitize all data immediately after loading/combining
            X_combined = sanitize_features(X_combined)
            X_cpte = sanitize_features(X_cpte)

            # Split feature sets based on known counts from the .npz file
            X_rbp_all = X_combined[:, :RBP_FEATURE_COUNT]
            X_network_all = X_combined[:, RBP_FEATURE_COUNT:RBP_FEATURE_COUNT + NETWORK_FEATURE_COUNT]
            
            logger.info("Data loaded and CPTE features aligned successfully.")
            return X_rbp_all, X_network_all, X_cpte, y, subject_ids
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None, None, None, None, None


# --- Classifier Class for Nested CV (LOSO + Optuna) ---
class EEGClassifierNestedCV:
    BAND_TO_IDX = {'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3, 'gamma': 4}

    def __init__(self, X_rbp, X_network, X_cpte, y, ids, n_outer_jobs, optuna_n_jobs, optuna_timeout_sec):
        
        self.X_rbp_all = X_rbp
        self.X_network_all = X_network
        self.X_cpte_all = X_cpte
        self.y_all = y
        self.ids = ids
        
        # Determine feature size per band based on total feature count
        self.rbp_band_size = RBP_FEATURE_COUNT // 5
        self.network_band_size = NETWORK_FEATURE_COUNT // 5
        self.cpte_band_size = X_cpte.shape[1] // 5 if X_cpte.shape[1] % 5 == 0 else None
        
        self.n_outer_jobs = n_outer_jobs
        self.optuna_n_jobs = optuna_n_jobs
        self.optuna_timeout_sec = optuna_timeout_sec
        
        self.bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        self.models = ['RandomForest', 'SVM', 'XGBoost', 'CatBoost', 'KNeighbors', 'MLP', 'LightGBM']
        self.feature_sets = ['rbp_only', 'network_only', 'cpte_only', 'rbp_network_only', 'combined']
        self.tasks = ['AD_CN', 'FTD_CN', 'AD_FTD'] 


    def _get_task_data(self, X_rbp, X_network, X_cpte, y, ids, task_name) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Filters data and relabels based on the classification task."""
        if task_name == 'AD_CN':
            mask = (y == 0) | (y == 1)
            label_map = None # AD=0, CN=1
        elif task_name == 'FTD_CN':
            mask = (y == 2) | (y == 1)
            label_map = {2: 0, 1: 1} # FTD=0, CN=1
        elif task_name == 'AD_FTD':
            mask = (y == 0) | (y == 2)
            label_map = {0: 0, 2: 1} # AD=0, FTD=1
        else:
            raise ValueError(f"Unknown task: {task_name}")

        X_rbp_task = X_rbp[mask]
        X_net_task = X_network[mask]
        X_cpte_task = X_cpte[mask]
        y_task = y[mask].copy()
        ids_task = ids[mask]
        
        if label_map:
            for old, new in label_map.items():
                y_task[y_task == old] = new
        
        return X_rbp_task, X_net_task, X_cpte_task, y_task, ids_task
    
    def _slice_by_band(self, X: np.ndarray, band: str, band_size: Optional[int]) -> np.ndarray:
        """Slices feature arrays by frequency band."""
        if band_size is None or X.shape[1] % 5 != 0 or X.shape[1] == 0:
            return X # Return full set if band slicing is not possible/relevant
        
        idx = self.BAND_TO_IDX[band]
        start, end = idx * band_size, (idx + 1) * band_size
        return X[:, start:end]

    def _get_data_for_experiment(self, task: str, feature_set: str, band: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assembles the final feature matrix for a specific experiment."""
        X_rbp, X_net, X_cpte, y, ids = self._get_task_data(
            self.X_rbp_all, self.X_network_all, self.X_cpte_all, self.y_all, self.ids, task
        )

        X_list = []
        if feature_set in ['rbp_only', 'rbp_network_only', 'combined']:
            X_list.append(self._slice_by_band(X_rbp, band, self.rbp_band_size))
        
        if feature_set in ['network_only', 'rbp_network_only', 'combined']:
            X_list.append(self._slice_by_band(X_net, band, self.network_band_size))
            
        if feature_set in ['cpte_only', 'combined']:
            X_list.append(self._slice_by_band(X_cpte, band, self.cpte_band_size))

        if not X_list:
             raise ValueError(f"Invalid feature set '{feature_set}' for band slicing.")

        X = np.hstack(X_list)
        return X, y, ids


    def _get_model(self, model_name: str, trial: Optional[optuna.Trial]) -> Any:
        # Handles model initialization for Optuna (trial is present) or final training (trial is None)
        if trial is None:
            # Default initialization for final model training
            if model_name == 'RandomForest':
                return RandomForestClassifier(random_state=42, n_jobs=1)
            # ... (Add similar default initializations for all models)
            elif model_name == 'SVM':
                return SVC(random_state=42)
            elif model_name == 'XGBoost':
                return XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=1)
            elif model_name == 'CatBoost':
                return CatBoostClassifier(loss_function='Logloss', verbose=0, random_seed=42)
            elif model_name == 'KNeighbors':
                return KNeighborsClassifier()
            elif model_name == 'MLP':
                return MLPClassifier(max_iter=500, random_state=42)
            elif model_name == 'LightGBM':
                return lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=1)
            else:
                raise ValueError(f"Unknown model: {model_name}")

        # Optuna trial specific parameter suggestions
        class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
        
        if model_name == 'RandomForest':
            return RandomForestClassifier(
                n_estimators=trial.suggest_int('rf_n_estimators', 50, 250),
                max_depth=trial.suggest_int('rf_max_depth', 5, 25, log=True),
                min_samples_split=trial.suggest_int('rf_min_samples_split', 2, 10),
                min_samples_leaf=trial.suggest_int('rf_min_samples_leaf', 1, 5),
                class_weight=class_weight,
                random_state=42,
                n_jobs=1
            )
        elif model_name == 'SVM':
            return SVC(
                C=trial.suggest_float('svm_C', 0.01, 10.0, log=True),
                kernel=trial.suggest_categorical('svm_kernel', ['rbf']),
                gamma=trial.suggest_categorical('svm_gamma', ['scale', 'auto']),
                class_weight=class_weight,
                random_state=42
            )
        elif model_name == 'XGBoost':
            return XGBClassifier(
                n_estimators=trial.suggest_int('xgb_n_estimators', 50, 250),
                max_depth=trial.suggest_int('xgb_max_depth', 3, 15),
                learning_rate=trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
                eval_metric='logloss',
                random_state=42,
                n_jobs=1
            )
        elif model_name == 'CatBoost':
            return CatBoostClassifier(
                iterations=trial.suggest_int('cb_iterations', 50, 250),
                learning_rate=trial.suggest_float('cb_learning_rate', 0.01, 0.3, log=True),
                depth=trial.suggest_int('cb_depth', 3, 10),
                loss_function='Logloss',
                verbose=0,
                random_seed=42
            )
        elif model_name == 'KNeighbors':
            return KNeighborsClassifier(
                n_neighbors=trial.suggest_int('knn_n_neighbors', 3, 15),
                weights=trial.suggest_categorical('knn_weights', ['uniform', 'distance']),
                p=trial.suggest_categorical('knn_p', [1, 2]),
                n_jobs=1
            )
        elif model_name == 'MLP':
            return MLPClassifier(
                hidden_layer_sizes=(trial.suggest_int('mlp_hidden_1', 50, 300),),
                activation=trial.suggest_categorical('mlp_activation', ['relu', 'tanh']),
                solver='adam',
                alpha=trial.suggest_float('mlp_alpha', 1e-5, 1e-1, log=True),
                learning_rate_init=trial.suggest_float('mlp_lr_init', 1e-4, 1e-2, log=True),
                max_iter=500,
                random_state=42
            )
        elif model_name == 'LightGBM':
            return lgb.LGBMClassifier(
                n_estimators=trial.suggest_int('lgbm_n_estimators', 50, 250),
                max_depth=trial.suggest_int('lgbm_max_depth', 3, 15),
                learning_rate=trial.suggest_float('lgbm_learning_rate', 0.01, 0.3, log=True),
                objective='binary',
                random_state=42,
                n_jobs=1
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray, model_name: str) -> float:
        """Optuna objective: group-aware inner CV on outer-train only (no test peeking)."""
        logger.info(f"[PATCH ACTIVE] Inner LOGO objective (no test peeking). PID={os.getpid()}")

        # 1) Hyperparameters for preprocessing
        max_features = int(min(X_train.shape[1], 100))
        use_selectkbest = max_features > 0 and trial.suggest_categorical('use_selectkbest', [True, False])
        k_features = trial.suggest_int('k_features', 5, max_features) if use_selectkbest else X_train.shape[1]

        use_pca = trial.suggest_categorical('use_pca', [True, False])
        pca_variance = trial.suggest_float('pca_variance', 0.8, 0.999) if use_pca else None

        # 2) Model with trial params
        classifier = self._get_model(model_name, trial)

        # 3) Inner group-aware CV over OUTER-TRAIN only
        inner_logo = LeaveOneGroupOut()
        f1_scores = []
        for step_idx, (in_idx, val_idx) in enumerate(inner_logo.split(X_train, y_train, groups=groups_train)):
            X_in, X_val = X_train[in_idx], X_train[val_idx]
            y_in, y_val = y_train[in_idx], y_train[val_idx]

            steps = [
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]
            if use_selectkbest:
                steps.append(('selectkbest', SelectKBest(score_func=f_classif, k=k_features)))
            if use_pca:
                steps.append(('pca', PCA(n_components=pca_variance)))
            steps.append(('classifier', classifier))

            pipe = Pipeline(steps)
            pipe.fit(X_in, y_in)
            y_hat = pipe.predict(X_val)
            f1_fold = f1_score(y_val, y_hat, average='weighted', zero_division=0)
            f1_scores.append(f1_fold)

            # Optional: report to pruner
            trial.report(float(np.mean(f1_scores)), step=step_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(f1_scores))

    def _train_and_evaluate_inner(self, model_name: str, X_outer_train: np.ndarray, y_outer_train: np.ndarray, groups_train: np.ndarray, X_outer_test: np.ndarray, y_outer_test: np.ndarray, subject_id: str) -> Dict[str, float]:
        """Runs the inner Optuna cross-validation to find best hyperparameters."""
        study_name = f"NestedLOSO_{model_name}_{subject_id}"
        try:
            sampler = TPESampler(seed=42)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                pruner=pruner,
                study_name=study_name
            )

            objective_with_data = lambda trial: self._objective(
                trial,
                X_outer_train,
                y_outer_train,
                groups_train,
                model_name
            )

            study.optimize(
                objective_with_data,
                n_trials=getattr(self, 'optuna_n_trials', 40),
                n_jobs=self.optuna_n_jobs,
                timeout=self.optuna_timeout_sec,
                show_progress_bar=False
            )

            best_f1 = study.best_value
            best_params = study.best_params
            logger.info(f"Study {study_name} finished. Best F1: {best_f1:.4f} with params: {best_params}")

            # --- Final Evaluation of the BEST Model ---
            # 1. Get the final classifier instance and map best params to estimator kwargs
            final_classifier = self._get_model(model_name, trial=None)

            def map_final_params(model_name: str, bp: Dict[str, Any]) -> Dict[str, Any]:
                m = {}
                if model_name == 'RandomForest':
                    if 'rf_n_estimators' in bp: m['n_estimators'] = bp['rf_n_estimators']
                    if 'rf_max_depth' in bp: m['max_depth'] = bp['rf_max_depth']
                    if 'rf_min_samples_split' in bp: m['min_samples_split'] = bp['rf_min_samples_split']
                    if 'rf_min_samples_leaf' in bp: m['min_samples_leaf'] = bp['rf_min_samples_leaf']
                    if 'class_weight' in bp: m['class_weight'] = bp['class_weight']
                elif model_name == 'SVM':
                    if 'svm_C' in bp: m['C'] = bp['svm_C']
                    if 'svm_kernel' in bp: m['kernel'] = bp['svm_kernel']
                    if 'svm_gamma' in bp: m['gamma'] = bp['svm_gamma']
                    if 'class_weight' in bp: m['class_weight'] = bp['class_weight']
                elif model_name == 'XGBoost':
                    if 'xgb_n_estimators' in bp: m['n_estimators'] = bp['xgb_n_estimators']
                    if 'xgb_max_depth' in bp: m['max_depth'] = bp['xgb_max_depth']
                    if 'xgb_learning_rate' in bp: m['learning_rate'] = bp['xgb_learning_rate']
                elif model_name == 'CatBoost':
                    if 'cb_iterations' in bp: m['iterations'] = bp['cb_iterations']
                    if 'cb_learning_rate' in bp: m['learning_rate'] = bp['cb_learning_rate']
                    if 'cb_depth' in bp: m['depth'] = bp['cb_depth']
                elif model_name == 'KNeighbors':
                    if 'knn_n_neighbors' in bp: m['n_neighbors'] = bp['knn_n_neighbors']
                    if 'knn_weights' in bp: m['weights'] = bp['knn_weights']
                    if 'knn_p' in bp: m['p'] = bp['knn_p']
                elif model_name == 'MLP':
                    if 'mlp_hidden_1' in bp: m['hidden_layer_sizes'] = (bp['mlp_hidden_1'],)
                    if 'mlp_activation' in bp: m['activation'] = bp['mlp_activation']
                    if 'mlp_alpha' in bp: m['alpha'] = bp['mlp_alpha']
                    if 'mlp_lr_init' in bp: m['learning_rate_init'] = bp['mlp_lr_init']
                elif model_name == 'LightGBM':
                    if 'lgbm_n_estimators' in bp: m['n_estimators'] = bp['lgbm_n_estimators']
                    if 'lgbm_max_depth' in bp: m['max_depth'] = bp['lgbm_max_depth']
                    if 'lgbm_learning_rate' in bp: m['learning_rate'] = bp['lgbm_learning_rate']
                return m

            classifier_params = map_final_params(model_name, best_params)
            if classifier_params:
                final_classifier.set_params(**classifier_params)

            # 2. Rebuild the FINAL pipeline using the best params
            steps = [
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]

            if best_params.get('use_selectkbest'):
                steps.append(('selectkbest', SelectKBest(score_func=f_classif, k=best_params.get('k_features'))))

            if best_params.get('use_pca'):
                steps.append(('pca', PCA(n_components=best_params.get('pca_variance'))))

            steps.append(('classifier', final_classifier))
            final_pipeline = Pipeline(steps)

            # 3. Train the FINAL model on the entire outer training set
            final_pipeline.fit(X_outer_train, y_outer_train)

            # 4. Predict and calculate metrics for the final held-out subject
            y_pred = final_pipeline.predict(X_outer_test)

            f1_final = f1_score(y_outer_test, y_pred, average='weighted', zero_division=0)
            acc_final = accuracy_score(y_outer_test, y_pred)
            prec_final, recall_final, fscore, _ = precision_recall_fscore_support(y_outer_test, y_pred, average='weighted', zero_division=0)

            return {
                "F1": f1_final,
                "Accuracy": acc_final,
                "Precision": prec_final,
                "Recall": recall_final,
                "Sensitivity": recall_final,
                "Best_Params": best_params
            }

        except Exception as e:
            logger.error(f"Error in inner CV for {study_name}: {e}")
            return {}

    def _run_single_experiment(self, task: str, model: str, feat: str, band: str) -> Optional[Tuple[str, str, str, str, Dict[str, float]]]:
        """Runs the outer Leave-One-Subject-Out (LOSO) cross-validation."""
        try:
            X_all, y_all, groups = self._get_data_for_experiment(task, feat, band)
        except ValueError as e:
            logger.error(f"Skipping experiment {task}/{model}/{feat}/{band}: {e}")
            return None
            
        logo = LeaveOneGroupOut()
        fold_results = []

        for fold_idx, (train_index, test_index) in enumerate(logo.split(X_all, y_all, groups=groups)):
            X_train, X_test = X_all[train_index], X_all[test_index]
            y_train, y_test = y_all[train_index], y_all[test_index]
            groups_train = groups[train_index]
            subject_id = groups[test_index[0]]
            subject_id_with_experiment = f"{subject_id}_exp-{model}-{feat}-{band}"
            fold_metrics = self._train_and_evaluate_inner(
                model,
                X_train,
                y_train,
                groups_train,
                X_test,
                y_test,
                subject_id_with_experiment
            )
            if fold_metrics:
                fold_results.append(fold_metrics)
                
        if not fold_results:
            logger.error(f"Experiment {task}, {model}, {feat}, {band} failed all folds.")
            return None
            
        # Average results across all held-out subjects
        avg_results = {
            metric: np.mean([res[metric] for res in fold_results if metric in res])
            for metric in ["F1", "Accuracy", "Precision", "Recall", "Sensitivity"]
        }
        # Best params are logged from the final fold's best trial
        avg_results["Best_Params"] = fold_results[-1].get("Best_Params", {})


        logger.info(
            f"Avg Results - Task: {task}, Model: {model}, Feature_Set: {feat}, Band: {band} | "
            f"F1: {avg_results['F1']:.4f}, Accuracy: {avg_results['Accuracy']:.4f}"
        )

        return task, model, feat, band, avg_results

    def run_pipeline(self, n_trials: int) -> Dict[Tuple[str, str, str, str], Dict[str, float]]:
        """Main function to run all experiments in parallel."""
        exps = [
            (task, model, feat, band)
            for task in self.tasks
            for model in self.models
            for feat in self.feature_sets
            for band in self.bands
        ]

        logger.info(f"Total experiments to run: {len(exps)}")

        self.optuna_n_trials = n_trials

        # Use joblib to parallelize the outer subject-out cross-validation
        # This is the correct use of parallelism for nested CV.
        parallel_outputs = Parallel(n_jobs=self.n_outer_jobs, verbose=5)(
            delayed(self._run_single_experiment)(task, model, feat, band)
            for (task, model, feat, band) in exps
        )
        
        results = {}
        for output in parallel_outputs:
            if output is not None:
                task, model, feat, band, avg_results = output
                results[(task, model, feat, band)] = avg_results
        
        return results


# --- Main Execution ---

def main():
    logger.info(f"Launching dl66 from {os.path.abspath(__file__)} (PID={os.getpid()})")
    data_path = 'eeg_features_combined_phase_locking_10s_50.npz'
    pre = EEGDataPreprocessor(data_path)
    # X_rbp, X_network, X_cpte, y, ids
    Xr, Xn, Xcpt, y, ids = pre.load_data()

    if Xr is None:
        logger.error("Data loading failed. Exiting.")
        return

    # Parallel settings: adjust as needed
    n_cpus = os.cpu_count() or 4
    # Run the main experiments (outer LOSO folds) in parallel. Up to 16 workers.
    outer_jobs = max(1, min(16, n_cpus))
    # Keep inner Optuna trials single-threaded for stability inside each worker process.
    optuna_jobs = 1
    optuna_timeout_sec = 300  # cap inner study per subject to 5 minutes

    clf = EEGClassifierNestedCV(Xr, Xn, Xcpt, y, ids, n_outer_jobs=outer_jobs, optuna_n_jobs=optuna_jobs, optuna_timeout_sec=optuna_timeout_sec)
    results = clf.run_pipeline(n_trials=40)

    # Save results
    df = pd.DataFrame([
        {
            'Task': t,
            'Model': m,
            'Feature_Set': f,
            'Band': b,
            'F1': v["F1"],
            'Accuracy': v["Accuracy"],
            'Precision': v["Precision"],
            'Recall': v["Recall"],
            'Sensitivity': v["Sensitivity"],
            'Best_Params': str(v["Best_Params"])
        }
        for (t, m, f, b), v in results.items()
    ])

    csv_filename = f'final_results_nested_cv_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(csv_filename, index=False)
    logger.info(f"Final results saved to {csv_filename}")


if __name__ == '__main__':
    main()
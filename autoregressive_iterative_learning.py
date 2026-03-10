"""
Autoregressive Iterative Learning Module for AR-SSRF Framework

This module implements the core autoregressive semi-supervised learning
framework described in Section 2.5, which progressively expands the
training set through iterative pseudo-labeling and model refinement.

Key Components:
1. Iterative pseudo-label generation and validation
2. Progressive training set expansion
3. Model performance monitoring and convergence detection
4. Early stopping based on validation metrics
5. Integration with dual-end sampling strategy

The framework iteratively:
- Trains model on current labeled set
- Predicts on unlabeled pool
- Selects high-confidence samples via dual-end sampling
- Expands training set with pseudo-labeled samples
- Monitors performance and detects convergence

Author: [HE YAN]
Date: [2026.3.10]
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import json
import warnings
from copy import deepcopy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)

warnings.filterwarnings('ignore')


class ConvergenceMonitor:
    """
    Monitor model convergence during iterative training.
    
    This class tracks performance metrics across iterations and
    detects convergence based on metric stability or degradation.
    """
    
    def __init__(self,
                 patience: int = 3,
                 min_delta: float = 0.001,
                 monitor_metric: str = 'ROC-AUC'):
        """
        Initialize convergence monitor.
        
        Args:
            patience: Number of iterations to wait for improvement (default: 3)
            min_delta: Minimum change to qualify as improvement (default: 0.001)
            monitor_metric: Metric to monitor for convergence (default: 'ROC-AUC')
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_metric = monitor_metric
        
        self.best_score = -np.inf
        self.best_iteration = 0
        self.wait_count = 0
        self.history = []
    
    def update(self, iteration: int, metrics: Dict[str, float]) -> bool:
        """
        Update monitor with new iteration metrics.
        
        Args:
            iteration: Current iteration number
            metrics: Dictionary of performance metrics
            
        Returns:
            True if training should stop, False otherwise
        """
        current_score = metrics.get(self.monitor_metric, -np.inf)
        self.history.append({
            'iteration': iteration,
            'score': current_score,
            **metrics
        })
        
        # Check for improvement
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.best_iteration = iteration
            self.wait_count = 0
            print(f"  ✓ New best {self.monitor_metric}: {current_score:.4f}")
            return False
        else:
            self.wait_count += 1
            print(f"  No improvement for {self.wait_count}/{self.patience} iterations")
            
            if self.wait_count >= self.patience:
                print(f"  Early stopping triggered at iteration {iteration}")
                return True
            
            return False
    
    def get_best_iteration(self) -> int:
        """Get iteration number with best performance."""
        return self.best_iteration
    
    def get_history(self) -> pd.DataFrame:
        """Get complete training history."""
        return pd.DataFrame(self.history)
    
    def reset(self):
        """Reset monitor state."""
        self.best_score = -np.inf
        self.best_iteration = 0
        self.wait_count = 0
        self.history = []


class AutoregressiveIterativeLearner:
    """
    Core autoregressive semi-supervised learning framework.
    
    This class implements the iterative pseudo-labeling strategy with
    progressive training set expansion and convergence monitoring.
    """
    
    def __init__(self,
                 base_classifier: BaseEstimator,
                 dual_end_sampler: Any,
                 max_iterations: int = 50,
                 min_unlabeled_samples: int = 10,
                 convergence_monitor: Optional[ConvergenceMonitor] = None,
                 validation_split: float = 0.2,
                 random_state: int = 42):
        """
        Initialize autoregressive iterative learner.
        
        Args:
            base_classifier: Base classifier instance (must support fit/predict_proba)
            dual_end_sampler: Dual-end sampling strategy instance
            max_iterations: Maximum number of iterations (default: 50)
            min_unlabeled_samples: Minimum unlabeled samples to continue (default: 10)
            convergence_monitor: Convergence monitor instance (optional)
            validation_split: Fraction of labeled data for validation (default: 0.2)
            random_state: Random seed for reproducibility
        """
        self.base_classifier = base_classifier
        self.dual_end_sampler = dual_end_sampler
        self.max_iterations = max_iterations
        self.min_unlabeled_samples = min_unlabeled_samples
        self.validation_split = validation_split
        self.random_state = random_state
        
        if convergence_monitor is None:
            self.convergence_monitor = ConvergenceMonitor(
                patience=3,
                min_delta=0.001,
                monitor_metric='ROC-AUC'
            )
        else:
            self.convergence_monitor = convergence_monitor
        
        self.model = None
        self.training_history = []
        self.best_model = None
        self.best_iteration = 0
    
    def fit(self,
            X_labeled: pd.DataFrame,
            y_labeled: pd.Series,
            X_unlabeled: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'AutoregressiveIterativeLearner':
        """
        Train model using autoregressive iterative learning.
        
        Args:
            X_labeled: Initial labeled training features
            y_labeled: Initial labeled training targets
            X_unlabeled: Unlabeled feature pool
            X_val: Validation features (optional, will split from labeled if not provided)
            y_val: Validation targets (optional)
            
        Returns:
            Self (fitted model)
        """
        print(f"\n{'='*80}")
        print(f"Starting Autoregressive Iterative Learning")
        print(f"{'='*80}")
        print(f"Initial labeled samples: {len(X_labeled)}")
        print(f"Unlabeled pool size: {len(X_unlabeled)}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"{'='*80}\n")
        
        # Create validation set if not provided
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_labeled, y_labeled,
                test_size=self.validation_split,
                random_state=self.random_state,
                stratify=y_labeled
            )
        else:
            X_train = X_labeled.copy()
            y_train = y_labeled.copy()
        
        # Initialize training set and unlabeled pool
        X_current = X_train.copy()
        y_current = y_train.copy()
        X_pool = X_unlabeled.copy()
        pool_indices = np.arange(len(X_pool))
        
        # Reset convergence monitor
        self.convergence_monitor.reset()
        
        # Iterative learning loop
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*80}")
            print(f"Iteration {iteration}/{self.max_iterations}")
            print(f"{'='*80}")
            print(f"Current training set size: {len(X_current)}")
            print(f"Remaining unlabeled pool: {len(X_pool)}")
            
            # Check stopping conditions
            if len(X_pool) < self.min_unlabeled_samples:
                print(f"\nStopping: Unlabeled pool exhausted (< {self.min_unlabeled_samples} samples)")
                break
            
            # Step 1: Train model on current labeled set
            print(f"\nStep 1: Training model...")
            self.model = deepcopy(self.base_classifier)
            self.model.fit(X_current, y_current)
            
            # Step 2: Evaluate on validation set
            print(f"\nStep 2: Evaluating on validation set...")
            val_metrics = self._evaluate_model(X_val, y_val)
            self._print_metrics(val_metrics, "Validation Performance")
            
            # Step 3: Check convergence
            should_stop = self.convergence_monitor.update(iteration, val_metrics)
            if should_stop:
                print(f"\nStopping: Convergence detected")
                break
            
            # Update best model if current is better
            if iteration == self.convergence_monitor.get_best_iteration():
                self.best_model = deepcopy(self.model)
                self.best_iteration = iteration
            
            # Step 4: Predict on unlabeled pool
            print(f"\nStep 3: Predicting on unlabeled pool...")
            probabilities = self.model.predict_proba(X_pool)[:, 1]
            
            # Step 5: Select high-confidence samples using dual-end sampling
            print(f"\nStep 4: Selecting high-confidence samples...")
            current_distribution = {
                0: np.sum(y_current == 0),
                1: np.sum(y_current == 1)
            }
            
            selected_indices, pseudo_labels = self.dual_end_sampler.select_samples(
                probabilities,
                pool_indices,
                current_distribution
            )
            
            # Check if any samples were selected
            if len(selected_indices) == 0:
                print(f"\nStopping: No samples selected (thresholds too strict)")
                break
            
            # Step 6: Expand training set with pseudo-labeled samples
            print(f"\nStep 5: Expanding training set...")
            X_selected = X_pool.iloc[selected_indices]
            y_selected = pd.Series(pseudo_labels, index=X_selected.index)
            
            X_current = pd.concat([X_current, X_selected], axis=0)
            y_current = pd.concat([y_current, y_selected], axis=0)
            
            # Remove selected samples from pool
            mask = np.ones(len(X_pool), dtype=bool)
            mask[selected_indices] = False
            X_pool = X_pool.iloc[mask].reset_index(drop=True)
            pool_indices = np.arange(len(X_pool))
            
            # Record iteration statistics
            self._record_iteration(
                iteration=iteration,
                n_labeled=len(X_current),
                n_unlabeled=len(X_pool),
                n_selected=len(selected_indices),
                class_distribution=current_distribution,
                metrics=val_metrics
            )
            
            print(f"\nIteration {iteration} complete:")
            print(f"  Added {len(selected_indices)} pseudo-labeled samples")
            print(f"  New training set size: {len(X_current)}")
            print(f"  Remaining unlabeled: {len(X_pool)}")
        
        # Use best model
        if self.best_model is not None:
            self.model = self.best_model
            print(f"\n{'='*80}")
            print(f"Training Complete - Using model from iteration {self.best_iteration}")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(f"Training Complete - Using final model")
            print(f"{'='*80}\n")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.predict_proba(X)
    
    def _evaluate_model(self,
                       X: pd.DataFrame,
                       y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on given dataset.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of performance metrics
        """
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
        # Calculate metrics
        metrics = {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "F1-Score": f1_score(y, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y, y_pred),
            "ROC-AUC": roc_auc_score(y, y_pred_proba)
        }
        
        return metrics
    
    def _print_metrics(self, metrics: Dict[str, float], title: str = "Performance"):
        """
        Print evaluation metrics.
        
        Args:
            metrics: Dictionary of metrics
            title: Title for the output
        """
        print(f"\n{title}:")
        print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1-Score:  {metrics['F1-Score']:.4f}")
        print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
        print(f"  MCC:       {metrics['MCC']:.4f}")
    
    def _record_iteration(self,
                         iteration: int,
                         n_labeled: int,
                         n_unlabeled: int,
                         n_selected: int,
                         class_distribution: Dict[int, int],
                         metrics: Dict[str, float]):
        """
        Record statistics for current iteration.
        
        Args:
            iteration: Iteration number
            n_labeled: Current size of labeled set
            n_unlabeled: Current size of unlabeled pool
            n_selected: Number of samples selected this iteration
            class_distribution: Current class distribution
            metrics: Performance metrics
        """
        record = {
            'iteration': iteration,
            'n_labeled': n_labeled,
            'n_unlabeled': n_unlabeled,
            'n_selected': n_selected,
            'n_positive': class_distribution.get(1, 0),
            'n_negative': class_distribution.get(0, 0),
            **metrics
        }
        
        self.training_history.append(record)
    
    def get_training_history(self) -> pd.DataFrame:
        """
        Get complete training history.
        
        Returns:
            DataFrame containing iteration statistics
        """
        return pd.DataFrame(self.training_history)
    
    def save_model(self, filepath: str):
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        import pickle
        
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved to: {filepath}")
    
    def save_training_history(self, filepath: str):
        """
        Save training history to CSV file.
        
        Args:
            filepath: Path to save history
        """
        history_df = self.get_training_history()
        history_df.to_csv(filepath, index=False)
        
        print(f"Training history saved to: {filepath}")


class ARSSRFFramework:
    """
    Complete AR-SSRF framework integrating all components.
    
    This class provides a high-level interface for the entire
    autoregressive semi-supervised random forest framework.
    """
    
    def __init__(self,
                 base_classifier: BaseEstimator,
                 threshold_high: float = 0.9,
                 threshold_low: float = 0.1,
                 max_iterations: int = 50,
                 patience: int = 3,
                 min_delta: float = 0.001,
                 balance_ratio: float = 1.0,
                 random_state: int = 42):
        """
        Initialize AR-SSRF framework.
        
        Args:
            base_classifier: Base classifier instance
            threshold_high: Upper probability threshold for positive class
            threshold_low: Lower probability threshold for negative class
            max_iterations: Maximum number of iterations
            patience: Patience for early stopping
            min_delta: Minimum improvement for early stopping
            balance_ratio: Target ratio of positive to negative samples
            random_state: Random seed
        """
        from four_imbalance_sampling import AdaptiveSampler
        
        # Initialize dual-end sampler
        self.sampler = AdaptiveSampler(
            initial_threshold_high=threshold_high,
            initial_threshold_low=threshold_low,
            balance_ratio=balance_ratio,
            auto_adjust=True
        )
        
        # Initialize convergence monitor
        self.monitor = ConvergenceMonitor(
            patience=patience,
            min_delta=min_delta,
            monitor_metric='ROC-AUC'
        )
        
        # Initialize iterative learner
        self.learner = AutoregressiveIterativeLearner(
            base_classifier=base_classifier,
            dual_end_sampler=self.sampler,
            max_iterations=max_iterations,
            convergence_monitor=self.monitor,
            random_state=random_state
        )
    
    def fit(self,
            X_labeled: pd.DataFrame,
            y_labeled: pd.Series,
            X_unlabeled: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'ARSSRFFramework':
        """
        Train AR-SSRF model.
        
        Args:
            X_labeled: Initial labeled training features
            y_labeled: Initial labeled training targets
            X_unlabeled: Unlabeled feature pool
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Self (fitted framework)
        """
        self.learner.fit(X_labeled, y_labeled, X_unlabeled, X_val, y_val)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self.learner.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self.learner.predict_proba(X)
    
    def get_training_history(self) -> pd.DataFrame:
        """Get training history."""
        return self.learner.get_training_history()
    
    def save(self, model_path: str, history_path: str):
        """
        Save model and training history.
        
        Args:
            model_path: Path to save model
            history_path: Path to save training history
        """
        self.learner.save_model(model_path)
        self.learner.save_training_history(history_path)


# Example usage
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    
    # Initialize base classifier
    base_clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    
    # Initialize AR-SSRF framework
    ar_ssrf = ARSSRFFramework(
        base_classifier=base_clf,
        threshold_high=0.9,
        threshold_low=0.1,
        max_iterations=50,
        patience=3,
        random_state=42
    )
    
    # Train model
    # ar_ssrf.fit(X_labeled, y_labeled, X_unlabeled, X_val, y_val)
    
    # Make predictions
    # y_pred = ar_ssrf.predict(X_test)
    # y_pred_proba = ar_ssrf.predict_proba(X_test)
    
    # Get training history
    # history = ar_ssrf.get_training_history()
    
    # Save model and history
    # ar_ssrf.save('ar_ssrf_model.pkl', 'training_history.csv')
    
    print("Autoregressive Iterative Learning Module Ready")
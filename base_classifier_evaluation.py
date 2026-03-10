"""
Base Classifier Evaluation Module for AR-SSRF Framework

This module implements the initial base classifier evaluation phase, where
5 machine learning algorithms are paired with 17 molecular representation
approaches, resulting in 85 model-feature combinations.

The best-performing configuration is selected as the foundation for the
subsequent autoregressive semi-supervised learning framework.

Key Components:
1. Five base classifiers with fixed hyperparameters (Text S5)
2. Model-feature combination evaluation
3. Performance comparison and selection

Author: [Your Name]
Date: [Date]
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import warnings
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


class BaseClassifierFactory:
    """
    Factory class for creating base classifiers with fixed hyperparameters.
    
    Hyperparameter settings are fixed to prevent overfitting to pseudo-label
    noise during the initial stage (as described in Text S5 of the paper).
    
    All hyperparameters below match the actual code implementation.
    """
    
    SUPPORTED_MODELS = {
        'logistic': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'svc': 'Support Vector Classification',
        'knn': 'K-Nearest Neighbors'
    }
    
    @staticmethod
    def create_classifier(model_name: str, random_state: int = 42) -> Any:
        """
        Create a classifier instance with fixed hyperparameters.
        
        Hyperparameter Configuration (matching code implementation):
        
        Logistic Regression:
            - max_iter: 1000 (ensures convergence on complex features)
            - solver: 'lbfgs' (efficient for L2 regularization)
            - C: 1.0 (regularization strength)
            - penalty: 'l2' (Ridge regularization)
        
        Random Forest:
            - n_estimators: 100 (number of trees)
            - criterion: 'gini' (split quality measure)
            - max_depth: None (trees grow until pure leaves)
            - min_samples_split: 2 (minimum samples to split node)
            - min_samples_leaf: 1 (minimum samples at leaf)
        
        XGBoost:
            - objective: 'binary:logistic'
            - n_estimators: 100 (number of boosting rounds)
            - learning_rate: 0.1 (step size shrinkage)
            - max_depth: 6 (maximum tree depth)
            - min_child_weight: 1 (minimum sum of instance weight)
        
        SVC:
            - kernel: 'rbf' (Radial Basis Function)
            - C: 1.0 (penalty parameter)
            - gamma: 'scale' (kernel coefficient = 1/(n_features * X.var()))
            - probability: True (enable probability estimates)
        
        KNN:
            - n_neighbors: 5 (number of neighbors)
            - metric: 'minkowski' with p=2 (Euclidean distance)
            - weights: 'uniform' (all neighbors weighted equally)
        
        Args:
            model_name: Name of the model (see SUPPORTED_MODELS)
            random_state: Random seed for reproducibility
            
        Returns:
            Initialized classifier instance
            
        Raises:
            ValueError: If model_name is not supported
        """
        model_name = model_name.lower()
        
        if model_name == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                C=1.0,
                penalty='l2',
                random_state=random_state
            )
        
        elif model_name == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                criterion='gini',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=random_state
            )
        
        elif model_name == 'xgboost':
            return XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                random_state=random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
        
        elif model_name == 'svc':
            return SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=random_state
            )
        
        elif model_name == 'knn':
            return KNeighborsClassifier(
                n_neighbors=5,
                metric='minkowski',
                p=2,
                weights='uniform'
            )
        
        else:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(BaseClassifierFactory.SUPPORTED_MODELS.keys())}"
            )
    
    @staticmethod
    def get_model_description(model_name: str) -> str:
        """Get human-readable description of model."""
        return BaseClassifierFactory.SUPPORTED_MODELS.get(
            model_name.lower(), 
            "Unknown Model"
        )


class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics.
    
    Computes classification metrics including:
    - Confusion matrix components (TP, TN, FP, FN)
    - Accuracy, Precision, Recall, Specificity
    - F1-Score, Matthews Correlation Coefficient (MCC)
    - ROC-AUC score
    """
    
    @staticmethod
    def evaluate(y_true: np.ndarray,
                y_pred: np.ndarray,
                y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate model performance with comprehensive metrics.
        
        Args:
            y_true: True labels (binary: 0 or 1)
            y_pred: Predicted labels (binary: 0 or 1)
            y_pred_proba: Predicted probabilities for positive class (optional)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        metrics = {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0,
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred)
        }
        
        # Add ROC-AUC if probabilities provided
        if y_pred_proba is not None:
            try:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics["ROC-AUC"] = np.nan
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Model Performance"):
        """
        Print evaluation metrics in a formatted table.
        
        Args:
            metrics: Dictionary of metric names and values
            title: Title for the metrics table
        """
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        print(f"  TP: {metrics['TP']:>6}  |  FN: {metrics['FN']:>6}")
        print(f"  FP: {metrics['FP']:>6}  |  TN: {metrics['TN']:>6}")
        
        # Performance metrics
        print(f"\nPerformance Metrics:")
        metric_order = ['Accuracy', 'Precision', 'Recall', 'Specificity', 
                       'NPV', 'F1-Score', 'MCC', 'ROC-AUC']
        
        for metric_name in metric_order:
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, np.integer)):
                    print(f"  {metric_name:<15}: {value:>6}")
                else:
                    print(f"  {metric_name:<15}: {value:>6.4f}")
        
        print(f"{'='*60}\n")


class BaseClassifierEvaluator:
    """
    Evaluate 85 model-feature combinations (5 algorithms × 17 feature sets).
    
    This class systematically evaluates all combinations to identify the
    best-performing configuration for the AR-SSRF framework.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize base classifier evaluator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.results = []
        self.best_config = None
    
    def evaluate_single_combination(self,
                                    model_name: str,
                                    feature_name: str,
                                    X_train: pd.DataFrame,
                                    y_train: pd.Series,
                                    X_test: pd.DataFrame,
                                    y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate a single model-feature combination.
        
        Args:
            model_name: Name of the classifier
            feature_name: Name of the feature set
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation results
        """
        # Create and train classifier
        classifier = BaseClassifierFactory.create_classifier(
            model_name, 
            self.random_state
        )
        classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = classifier.predict(X_test)
        
        # Get probabilities if available
        if hasattr(classifier, 'predict_proba'):
            y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = None
        
        # Evaluate performance
        metrics = ModelEvaluator.evaluate(y_test, y_pred, y_pred_proba)
        
        # Compile results
        result = {
            'model': model_name,
            'feature_set': feature_name,
            'classifier': classifier,
            **metrics
        }
        
        return result
    
    def evaluate_all_combinations(self,
                                  feature_dict: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                                  test_data: Tuple[pd.DataFrame, pd.Series],
                                  model_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Evaluate all model-feature combinations.
        
        Args:
            feature_dict: Dictionary mapping feature names to (X_train, y_train) tuples
            test_data: Tuple of (X_test, y_test)
            model_list: List of model names to evaluate (default: all 5 models)
            
        Returns:
            DataFrame containing all evaluation results
        """
        if model_list is None:
            model_list = list(BaseClassifierFactory.SUPPORTED_MODELS.keys())
        
        X_test, y_test = test_data
        
        print(f"\nEvaluating {len(model_list)} models × {len(feature_dict)} feature sets")
        print(f"Total combinations: {len(model_list) * len(feature_dict)}")
        print(f"{'='*60}\n")
        
        # Evaluate all combinations
        for model_name in model_list:
            for feature_name, (X_train, y_train) in tqdm(
                feature_dict.items(),
                desc=f"Evaluating {model_name}"
            ):
                try:
                    result = self.evaluate_single_combination(
                        model_name, feature_name,
                        X_train, y_train,
                        X_test, y_test
                    )
                    self.results.append(result)
                except Exception as e:
                    print(f"Error with {model_name} + {feature_name}: {str(e)}")
                    continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Identify best configuration
        self.best_config = results_df.loc[results_df['ROC-AUC'].idxmax()]
        
        print(f"\n{'='*60}")
        print(f"Evaluation Complete")
        print(f"{'='*60}")
        print(f"\nBest Configuration:")
        print(f"  Model: {self.best_config['model']}")
        print(f"  Feature Set: {self.best_config['feature_set']}")
        print(f"  ROC-AUC: {self.best_config['ROC-AUC']:.4f}")
        print(f"  Accuracy: {self.best_config['Accuracy']:.4f}")
        print(f"  F1-Score: {self.best_config['F1-Score']:.4f}")
        
        return results_df
    
    def save_results(self, output_file: str):
        """
        Save evaluation results to CSV file.
        
        Args:
            output_file: Path to output CSV file
        """
        results_df = pd.DataFrame(self.results)
        
        # Remove classifier objects before saving
        results_df_save = results_df.drop(columns=['classifier'], errors='ignore')
        results_df_save.to_csv(output_file, index=False)
        
        print(f"\nResults saved to: {output_file}")
    
    def get_top_configurations(self, n: int = 10, metric: str = 'ROC-AUC') -> pd.DataFrame:
        """
        Get top N configurations based on specified metric.
        
        Args:
            n: Number of top configurations to return
            metric: Metric to rank by (default: 'ROC-AUC')
            
        Returns:
            DataFrame with top N configurations
        """
        results_df = pd.DataFrame(self.results)
        results_df_save = results_df.drop(columns=['classifier'], errors='ignore')
        
        top_configs = results_df_save.nlargest(n, metric)
        
        print(f"\nTop {n} Configurations by {metric}:")
        print(f"{'='*80}")
        for idx, row in top_configs.iterrows():
            print(f"{row['model']:<20} + {row['feature_set']:<30} | {metric}: {row[metric]:.4f}")
        print(f"{'='*80}\n")
        
        return top_configs


# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = BaseClassifierEvaluator(random_state=42)
    
    # Prepare feature dictionary (example structure)
    # feature_dict = {
    #     'rdkit_descriptors': (X_train_rdkit, y_train),
    #     'morgan_fingerprints': (X_train_morgan, y_train),
    #     'maccs_keys': (X_train_maccs, y_train),
    #     # ... 14 more feature sets
    # }
    
    # Prepare test data
    # test_data = (X_test, y_test)
    
    # Evaluate all combinations
    # results_df = evaluator.evaluate_all_combinations(feature_dict, test_data)
    
    # Save results
    # evaluator.save_results('base_classifier_evaluation_results.csv')
    
    # Get top 10 configurations
    # top_configs = evaluator.get_top_configurations(n=10, metric='ROC-AUC')
    
    print("Base Classifier Evaluation Module Ready")
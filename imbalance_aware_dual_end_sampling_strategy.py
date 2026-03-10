"""
Imbalance-aware Dual-end Sampling Module for AR-SSRF Framework

This module implements the dual-end sampling strategy described in Section 2.5.3,
which addresses severe class imbalance in semi-supervised learning through
probability-based sample selection.

Key Components:
1. Dual-end probability-based sampling (high and low confidence)
2. Dynamic sample ratio adjustment based on class distribution
3. Confidence threshold management
4. Sample quality control and validation

The strategy selects samples from both ends of the probability distribution
to maintain balanced representation while expanding the training set.

Author: [HE YAN]
Date: [2026.3.10]
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class DualEndSampler:
    """
    Dual-end sampling strategy for imbalanced semi-supervised learning.
    
    This class implements the probability-based sample selection approach
    that selects high-confidence samples from both positive and negative
    classes to maintain class balance during iterative training.
    
    The sampling strategy uses two thresholds:
    - Upper threshold (P_high): for positive class selection
    - Lower threshold (P_low): for negative class selection
    """
    
    def __init__(self,
                 initial_threshold_high: float = 0.9,
                 initial_threshold_low: float = 0.1,
                 min_samples_per_class: int = 10,
                 max_samples_per_iteration: Optional[int] = None,
                 balance_ratio: float = 1.0):
        """
        Initialize dual-end sampler.
        
        Args:
            initial_threshold_high: Initial upper probability threshold (default: 0.9)
            initial_threshold_low: Initial lower probability threshold (default: 0.1)
            min_samples_per_class: Minimum samples required per class (default: 10)
            max_samples_per_iteration: Maximum samples to select per iteration (optional)
            balance_ratio: Target ratio of positive to negative samples (default: 1.0)
        """
        self.threshold_high = initial_threshold_high
        self.threshold_low = initial_threshold_low
        self.min_samples_per_class = min_samples_per_class
        self.max_samples_per_iteration = max_samples_per_iteration
        self.balance_ratio = balance_ratio
        
        self.selection_history = []
    
    def select_samples(self,
                      probabilities: np.ndarray,
                      indices: np.ndarray,
                      current_class_distribution: Optional[Dict[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select samples using dual-end probability thresholds.
        
        Selection criteria:
        - Positive samples: P(y=1) >= threshold_high
        - Negative samples: P(y=1) <= threshold_low
        
        Args:
            probabilities: Predicted probabilities for positive class
            indices: Indices of unlabeled samples
            current_class_distribution: Current distribution of labeled samples (optional)
            
        Returns:
            Tuple of (selected_indices, pseudo_labels)
        """
        # Identify high-confidence positive samples
        positive_mask = probabilities >= self.threshold_high
        positive_indices = indices[positive_mask]
        positive_labels = np.ones(len(positive_indices), dtype=int)
        
        # Identify high-confidence negative samples
        negative_mask = probabilities <= self.threshold_low
        negative_indices = indices[negative_mask]
        negative_labels = np.zeros(len(negative_indices), dtype=int)
        
        print(f"\nDual-end Sampling:")
        print(f"  Threshold High: {self.threshold_high:.3f} -> {len(positive_indices)} positive samples")
        print(f"  Threshold Low:  {self.threshold_low:.3f} -> {len(negative_indices)} negative samples")
        
        # Apply balance ratio adjustment if needed
        if current_class_distribution is not None:
            positive_indices, negative_indices = self._balance_selection(
                positive_indices, negative_indices,
                current_class_distribution
            )
            positive_labels = np.ones(len(positive_indices), dtype=int)
            negative_labels = np.zeros(len(negative_indices), dtype=int)
        
        # Combine selected samples
        selected_indices = np.concatenate([positive_indices, negative_indices])
        pseudo_labels = np.concatenate([positive_labels, negative_labels])
        
        # Apply maximum samples constraint if specified
        if self.max_samples_per_iteration is not None:
            if len(selected_indices) > self.max_samples_per_iteration:
                # Randomly sample to meet constraint while maintaining balance
                selected_indices, pseudo_labels = self._apply_max_constraint(
                    selected_indices, pseudo_labels
                )
        
        # Record selection statistics
        self._record_selection(len(positive_indices), len(negative_indices))
        
        print(f"  Final selection: {len(selected_indices)} samples "
              f"(Pos: {np.sum(pseudo_labels)}, Neg: {len(pseudo_labels) - np.sum(pseudo_labels)})")
        
        return selected_indices, pseudo_labels
    
    def _balance_selection(self,
                          positive_indices: np.ndarray,
                          negative_indices: np.ndarray,
                          current_distribution: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance sample selection based on current class distribution.
        
        This method adjusts the number of selected samples from each class
        to maintain the target balance ratio.
        
        Args:
            positive_indices: Indices of positive samples
            negative_indices: Indices of negative samples
            current_distribution: Current class distribution {0: count_neg, 1: count_pos}
            
        Returns:
            Tuple of (balanced_positive_indices, balanced_negative_indices)
        """
        current_pos = current_distribution.get(1, 0)
        current_neg = current_distribution.get(0, 0)
        
        # Calculate current ratio
        if current_neg > 0:
            current_ratio = current_pos / current_neg
        else:
            current_ratio = float('inf')
        
        # Determine how many samples to select from each class
        n_pos_available = len(positive_indices)
        n_neg_available = len(negative_indices)
        
        if current_ratio < self.balance_ratio:
            # Need more positive samples
            n_pos_select = min(n_pos_available, 
                             int((self.balance_ratio * current_neg - current_pos)))
            n_neg_select = min(n_neg_available,
                             int(n_pos_select / self.balance_ratio))
        else:
            # Need more negative samples
            n_neg_select = min(n_neg_available,
                             int((current_pos / self.balance_ratio - current_neg)))
            n_pos_select = min(n_pos_available,
                             int(n_neg_select * self.balance_ratio))
        
        # Ensure minimum samples per class
        n_pos_select = max(n_pos_select, self.min_samples_per_class)
        n_neg_select = max(n_neg_select, self.min_samples_per_class)
        
        # Randomly sample if needed
        if n_pos_select < n_pos_available:
            positive_indices = np.random.choice(
                positive_indices, n_pos_select, replace=False
            )
        
        if n_neg_select < n_neg_available:
            negative_indices = np.random.choice(
                negative_indices, n_neg_select, replace=False
            )
        
        return positive_indices, negative_indices
    
    def _apply_max_constraint(self,
                            selected_indices: np.ndarray,
                            pseudo_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply maximum samples per iteration constraint while maintaining balance.
        
        Args:
            selected_indices: All selected sample indices
            pseudo_labels: Corresponding pseudo labels
            
        Returns:
            Tuple of (constrained_indices, constrained_labels)
        """
        n_total = len(selected_indices)
        n_max = self.max_samples_per_iteration
        
        if n_total <= n_max:
            return selected_indices, pseudo_labels
        
        # Separate by class
        pos_mask = pseudo_labels == 1
        pos_indices = selected_indices[pos_mask]
        neg_indices = selected_indices[~pos_mask]
        
        # Calculate balanced selection
        n_pos = min(len(pos_indices), n_max // 2)
        n_neg = min(len(neg_indices), n_max - n_pos)
        
        # Adjust if one class has fewer samples
        if n_pos < n_max // 2:
            n_neg = min(len(neg_indices), n_max - n_pos)
        if n_neg < n_max // 2:
            n_pos = min(len(pos_indices), n_max - n_neg)
        
        # Random sampling
        selected_pos = np.random.choice(pos_indices, n_pos, replace=False)
        selected_neg = np.random.choice(neg_indices, n_neg, replace=False)
        
        # Combine
        constrained_indices = np.concatenate([selected_pos, selected_neg])
        constrained_labels = np.concatenate([
            np.ones(n_pos, dtype=int),
            np.zeros(n_neg, dtype=int)
        ])
        
        return constrained_indices, constrained_labels
    
    def _record_selection(self, n_positive: int, n_negative: int):
        """
        Record selection statistics for monitoring.
        
        Args:
            n_positive: Number of positive samples selected
            n_negative: Number of negative samples selected
        """
        self.selection_history.append({
            'iteration': len(self.selection_history) + 1,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'threshold_high': self.threshold_high,
            'threshold_low': self.threshold_low
        })
    
    def adjust_thresholds(self,
                         n_positive_selected: int,
                         n_negative_selected: int,
                         adjustment_factor: float = 0.05):
        """
        Dynamically adjust thresholds based on selection results.
        
        If insufficient samples are selected, thresholds are relaxed.
        If too many samples are selected, thresholds are tightened.
        
        Args:
            n_positive_selected: Number of positive samples selected
            n_negative_selected: Number of negative samples selected
            adjustment_factor: Amount to adjust thresholds (default: 0.05)
        """
        # Adjust high threshold based on positive sample count
        if n_positive_selected < self.min_samples_per_class:
            self.threshold_high = max(0.5, self.threshold_high - adjustment_factor)
            print(f"  Relaxing high threshold to {self.threshold_high:.3f}")
        
        # Adjust low threshold based on negative sample count
        if n_negative_selected < self.min_samples_per_class:
            self.threshold_low = min(0.5, self.threshold_low + adjustment_factor)
            print(f"  Relaxing low threshold to {self.threshold_low:.3f}")
    
    def get_selection_summary(self) -> pd.DataFrame:
        """
        Get summary of selection history.
        
        Returns:
            DataFrame containing selection statistics across iterations
        """
        return pd.DataFrame(self.selection_history)
    
    def reset_thresholds(self,
                        threshold_high: Optional[float] = None,
                        threshold_low: Optional[float] = None):
        """
        Reset thresholds to specified or initial values.
        
        Args:
            threshold_high: New high threshold (optional)
            threshold_low: New low threshold (optional)
        """
        if threshold_high is not None:
            self.threshold_high = threshold_high
        if threshold_low is not None:
            self.threshold_low = threshold_low
        
        print(f"Thresholds reset to: High={self.threshold_high:.3f}, Low={self.threshold_low:.3f}")


class AdaptiveSampler(DualEndSampler):
    """
    Adaptive dual-end sampler with automatic threshold adjustment.
    
    This extended version automatically adjusts thresholds based on
    selection success rate and class distribution evolution.
    """
    
    def __init__(self,
                 initial_threshold_high: float = 0.9,
                 initial_threshold_low: float = 0.1,
                 min_samples_per_class: int = 10,
                 max_samples_per_iteration: Optional[int] = None,
                 balance_ratio: float = 1.0,
                 auto_adjust: bool = True,
                 adjustment_factor: float = 0.05):
        """
        Initialize adaptive sampler.
        
        Args:
            initial_threshold_high: Initial upper probability threshold
            initial_threshold_low: Initial lower probability threshold
            min_samples_per_class: Minimum samples required per class
            max_samples_per_iteration: Maximum samples to select per iteration
            balance_ratio: Target ratio of positive to negative samples
            auto_adjust: Enable automatic threshold adjustment
            adjustment_factor: Amount to adjust thresholds
        """
        super().__init__(
            initial_threshold_high,
            initial_threshold_low,
            min_samples_per_class,
            max_samples_per_iteration,
            balance_ratio
        )
        
        self.auto_adjust = auto_adjust
        self.adjustment_factor = adjustment_factor
    
    def select_samples(self,
                      probabilities: np.ndarray,
                      indices: np.ndarray,
                      current_class_distribution: Optional[Dict[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select samples with automatic threshold adjustment.
        
        Args:
            probabilities: Predicted probabilities for positive class
            indices: Indices of unlabeled samples
            current_class_distribution: Current distribution of labeled samples
            
        Returns:
            Tuple of (selected_indices, pseudo_labels)
        """
        # Perform selection
        selected_indices, pseudo_labels = super().select_samples(
            probabilities, indices, current_class_distribution
        )
        
        # Auto-adjust thresholds if enabled
        if self.auto_adjust:
            n_positive = np.sum(pseudo_labels)
            n_negative = len(pseudo_labels) - n_positive
            
            self.adjust_thresholds(
                n_positive, n_negative, self.adjustment_factor
            )
        
        return selected_indices, pseudo_labels


class SampleQualityMonitor:
    """
    Monitor and validate quality of selected pseudo-labeled samples.
    
    This class tracks selection statistics and provides quality metrics
    to ensure the semi-supervised learning process maintains high standards.
    """
    
    def __init__(self):
        """Initialize sample quality monitor."""
        self.iteration_stats = []
    
    def record_iteration(self,
                        iteration: int,
                        n_selected: int,
                        class_distribution: Dict[int, int],
                        confidence_scores: np.ndarray,
                        thresholds: Dict[str, float]):
        """
        Record statistics for current iteration.
        
        Args:
            iteration: Current iteration number
            n_selected: Number of samples selected
            class_distribution: Distribution of selected samples by class
            confidence_scores: Confidence scores of selected samples
            thresholds: Current threshold values
        """
        stats = {
            'iteration': iteration,
            'n_selected': n_selected,
            'n_positive': class_distribution.get(1, 0),
            'n_negative': class_distribution.get(0, 0),
            'mean_confidence': np.mean(confidence_scores),
            'std_confidence': np.std(confidence_scores),
            'min_confidence': np.min(confidence_scores),
            'max_confidence': np.max(confidence_scores),
            'threshold_high': thresholds.get('high', np.nan),
            'threshold_low': thresholds.get('low', np.nan)
        }
        
        self.iteration_stats.append(stats)
    
    def get_quality_report(self) -> pd.DataFrame:
        """
        Generate quality report across all iterations.
        
        Returns:
            DataFrame containing quality metrics
        """
        return pd.DataFrame(self.iteration_stats)
    
    def print_summary(self):
        """Print summary of sample quality across iterations."""
        df = self.get_quality_report()
        
        print(f"\n{'='*60}")
        print(f"Sample Quality Summary")
        print(f"{'='*60}")
        print(f"Total iterations: {len(df)}")
        print(f"Total samples selected: {df['n_selected'].sum()}")
        print(f"Average samples per iteration: {df['n_selected'].mean():.1f}")
        print(f"Average confidence: {df['mean_confidence'].mean():.3f}")
        print(f"Class balance (Pos/Neg): {df['n_positive'].sum()}/{df['n_negative'].sum()}")
        print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Initialize dual-end sampler
    sampler = DualEndSampler(
        initial_threshold_high=0.9,
        initial_threshold_low=0.1,
        min_samples_per_class=10,
        balance_ratio=1.0
    )
    
    # Example: Select samples from unlabeled pool
    # probabilities = model.predict_proba(X_unlabeled)[:, 1]
    # indices = np.arange(len(X_unlabeled))
    # current_distribution = {0: 100, 1: 50}  # 100 negative, 50 positive
    
    # selected_indices, pseudo_labels = sampler.select_samples(
    #     probabilities, indices, current_distribution
    # )
    
    # Initialize adaptive sampler
    adaptive_sampler = AdaptiveSampler(
        initial_threshold_high=0.9,
        initial_threshold_low=0.1,
        auto_adjust=True,
        adjustment_factor=0.05
    )
    
    # Initialize quality monitor
    monitor = SampleQualityMonitor()
    
    print("Imbalance-aware Dual-end Sampling Module Ready")
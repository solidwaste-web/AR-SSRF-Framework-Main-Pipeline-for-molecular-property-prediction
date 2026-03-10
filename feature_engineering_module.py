"""
Feature Engineering Module for AR-SSRF Framework

This module implements feature selection and optimization procedures including:
1. Missing value filtering (row and column level)
2. Feature redundancy removal based on correlation analysis
3. Low variance feature filtering
4. Feature quality assessment and validation

The module processes 17 molecular representation approaches as described
in the paper's methodology section.

Author: [HE YAN]
Date: [2026.3.10]
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from scipy.stats import spearmanr
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


class FeatureQualityFilter:
    """
    Filter features based on missing values and variance thresholds.
    
    This class implements the feature quality control procedures described
    in the paper, including row-level and column-level missing value filtering,
    and low variance feature removal.
    """
    
    def __init__(self,
                 row_nan_ratio_threshold: float = 0.3,
                 column_nan_ratio_threshold: float = 0.10,
                 binary_variance_threshold: float = 0.01):
        """
        Initialize feature quality filter.
        
        Args:
            row_nan_ratio_threshold: Maximum allowed missing value ratio per row (default: 0.3)
            column_nan_ratio_threshold: Maximum allowed missing value ratio per column (default: 0.10)
            binary_variance_threshold: Minimum variance for binary features (default: 0.01)
        """
        self.row_nan_ratio_threshold = row_nan_ratio_threshold
        self.column_nan_ratio_threshold = column_nan_ratio_threshold
        self.binary_variance_threshold = binary_variance_threshold
    
    def filter_rows_by_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with excessive missing values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with high-missing-value rows removed
        """
        initial_rows = len(df)
        
        # Calculate missing value ratio for each row
        row_nan_ratios = df.isnull().sum(axis=1) / len(df.columns)
        
        # Keep rows below threshold
        valid_rows = row_nan_ratios <= self.row_nan_ratio_threshold
        df_filtered = df[valid_rows]
        
        removed_rows = initial_rows - len(df_filtered)
        print(f"Removed {removed_rows} rows with >{self.row_nan_ratio_threshold*100}% missing values")
        print(f"Retained {len(df_filtered)} rows")
        
        return df_filtered
    
    def filter_columns_by_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns with excessive missing values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with high-missing-value columns removed
        """
        initial_cols = len(df.columns)
        
        # Calculate missing value ratio for each column
        col_nan_ratios = df.isnull().sum(axis=0) / len(df)
        
        # Keep columns below threshold
        valid_cols = col_nan_ratios <= self.column_nan_ratio_threshold
        df_filtered = df.loc[:, valid_cols]
        
        removed_cols = initial_cols - len(df_filtered.columns)
        print(f"Removed {removed_cols} columns with >{self.column_nan_ratio_threshold*100}% missing values")
        print(f"Retained {len(df_filtered.columns)} columns")
        
        return df_filtered
    
    def filter_low_variance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove binary features with variance below threshold.
        
        Low variance binary features provide little discriminative information
        and can be safely removed to reduce dimensionality.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with low-variance features removed
        """
        initial_cols = len(df.columns)
        
        # Identify binary columns (only 0 and 1 values)
        binary_cols = []
        for col in df.columns:
            unique_values = df[col].dropna().unique()
            if len(unique_values) <= 2 and set(unique_values).issubset({0, 1, 0.0, 1.0}):
                binary_cols.append(col)
        
        # Calculate variance for binary columns
        low_variance_cols = []
        for col in binary_cols:
            variance = df[col].var()
            if variance < self.binary_variance_threshold:
                low_variance_cols.append(col)
        
        # Remove low variance columns
        df_filtered = df.drop(columns=low_variance_cols)
        
        print(f"Identified {len(binary_cols)} binary features")
        print(f"Removed {len(low_variance_cols)} low-variance binary features (var < {self.binary_variance_threshold})")
        print(f"Retained {len(df_filtered.columns)} columns")
        
        return df_filtered


class FeatureRedundancyRemover:
    """
    Remove redundant features based on correlation analysis.
    
    This class identifies and removes highly correlated features to reduce
    multicollinearity and improve model interpretability.
    """
    
    def __init__(self,
                 correlation_threshold: float = 0.9,
                 correlation_method: str = 'spearman'):
        """
        Initialize feature redundancy remover.
        
        Args:
            correlation_threshold: Threshold for identifying redundant features (default: 0.9)
            correlation_method: Method for correlation calculation ('spearman' or 'pearson')
        """
        self.correlation_threshold = correlation_threshold
        self.correlation_method = correlation_method
    
    def calculate_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix using specified method.
        
        Args:
            df: Input DataFrame with numeric features
            
        Returns:
            Correlation matrix
        """
        if self.correlation_method == 'spearman':
            # Spearman correlation for non-linear relationships
            corr_matrix = df.corr(method='spearman')
        elif self.correlation_method == 'pearson':
            # Pearson correlation for linear relationships
            corr_matrix = df.corr(method='pearson')
        else:
            raise ValueError(f"Unsupported correlation method: {self.correlation_method}")
        
        return corr_matrix
    
    def identify_redundant_features(self, df: pd.DataFrame) -> Set[str]:
        """
        Identify redundant features based on correlation threshold.
        
        For each pair of highly correlated features, the feature with
        more missing values is marked for removal.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Set of column names to remove
        """
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(df)
        
        # Get upper triangle of correlation matrix (avoid duplicates)
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_corr = corr_matrix.where(upper_triangle)
        
        # Find highly correlated pairs
        redundant_features = set()
        
        for col in upper_corr.columns:
            # Find features highly correlated with current column
            high_corr_features = upper_corr.index[
                abs(upper_corr[col]) >= self.correlation_threshold
            ].tolist()
            
            for corr_feature in high_corr_features:
                # Compare missing value counts
                col_missing = df[col].isnull().sum()
                corr_missing = df[corr_feature].isnull().sum()
                
                # Remove feature with more missing values
                if col_missing > corr_missing:
                    redundant_features.add(col)
                else:
                    redundant_features.add(corr_feature)
        
        return redundant_features
    
    def remove_redundant_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Set[str]]:
        """
        Remove redundant features from DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (filtered DataFrame, set of removed column names)
        """
        initial_cols = len(df.columns)
        
        # Identify redundant features
        redundant_features = self.identify_redundant_features(df)
        
        # Remove redundant features
        df_filtered = df.drop(columns=list(redundant_features))
        
        print(f"Identified {len(redundant_features)} redundant features (correlation >= {self.correlation_threshold})")
        print(f"Retained {len(df_filtered.columns)} columns")
        
        return df_filtered, redundant_features


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline integrating all processing steps.
    
    This class orchestrates the entire feature selection and optimization
    workflow including missing value filtering, variance filtering, and
    redundancy removal.
    """
    
    def __init__(self,
                 row_nan_ratio_threshold: float = 0.3,
                 column_nan_ratio_threshold: float = 0.10,
                 binary_variance_threshold: float = 0.01,
                 correlation_threshold: float = 0.9,
                 correlation_method: str = 'spearman'):
        """
        Initialize feature engineering pipeline.
        
        Args:
            row_nan_ratio_threshold: Maximum allowed missing value ratio per row
            column_nan_ratio_threshold: Maximum allowed missing value ratio per column
            binary_variance_threshold: Minimum variance for binary features
            correlation_threshold: Threshold for identifying redundant features
            correlation_method: Method for correlation calculation
        """
        self.quality_filter = FeatureQualityFilter(
            row_nan_ratio_threshold,
            column_nan_ratio_threshold,
            binary_variance_threshold
        )
        
        self.redundancy_remover = FeatureRedundancyRemover(
            correlation_threshold,
            correlation_method
        )
        
        self.processing_log = []
    
    def process_features(self, 
                        df: pd.DataFrame,
                        feature_prefix: Optional[str] = None) -> pd.DataFrame:
        """
        Apply complete feature engineering pipeline to DataFrame.
        
        Processing steps:
        1. Filter rows with excessive missing values
        2. Filter columns with excessive missing values
        3. Remove low variance binary features
        4. Remove redundant features based on correlation
        
        Args:
            df: Input DataFrame
            feature_prefix: Optional prefix to filter feature columns
            
        Returns:
            Processed DataFrame with optimized features
        """
        print(f"\n{'='*60}")
        print(f"Starting Feature Engineering Pipeline")
        print(f"Initial shape: {df.shape}")
        print(f"{'='*60}\n")
        
        # Filter by prefix if specified
        if feature_prefix:
            feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]
            df_features = df[feature_cols]
            df_other = df.drop(columns=feature_cols)
        else:
            df_features = df
            df_other = pd.DataFrame()
        
        # Step 1: Filter rows by missing values
        print("Step 1: Filtering rows by missing values")
        df_features = self.quality_filter.filter_rows_by_missing_values(df_features)
        self.processing_log.append(('row_filter', df_features.shape))
        
        # Step 2: Filter columns by missing values
        print("\nStep 2: Filtering columns by missing values")
        df_features = self.quality_filter.filter_columns_by_missing_values(df_features)
        self.processing_log.append(('column_filter', df_features.shape))
        
        # Step 3: Remove low variance features
        print("\nStep 3: Removing low variance features")
        df_features = self.quality_filter.filter_low_variance_features(df_features)
        self.processing_log.append(('variance_filter', df_features.shape))
        
        # Step 4: Remove redundant features
        print("\nStep 4: Removing redundant features")
        df_features, removed_features = self.redundancy_remover.remove_redundant_features(df_features)
        self.processing_log.append(('redundancy_removal', df_features.shape))
        
        # Combine with non-feature columns if applicable
        if not df_other.empty:
            # Align indices
            df_features = df_features.loc[df_other.index]
            df_result = pd.concat([df_other, df_features], axis=1)
        else:
            df_result = df_features
        
        print(f"\n{'='*60}")
        print(f"Feature Engineering Complete")
        print(f"Final shape: {df_result.shape}")
        print(f"Reduction: {df.shape[1]} -> {df_result.shape[1]} columns "
              f"({(1 - df_result.shape[1]/df.shape[1])*100:.1f}% reduction)")
        print(f"{'='*60}\n")
        
        return df_result
    
    def process_dataset_folder(self,
                              folder_path: str,
                              output_file: str,
                              file_pattern: str = '*.csv') -> Dict[str, List[str]]:
        """
        Process multiple feature files in a folder and save retained columns.
        
        This method is useful for processing 17 different molecular representation
        approaches stored in separate files.
        
        Args:
            folder_path: Path to folder containing feature files
            output_file: Path to save JSON file with retained column names
            file_pattern: Pattern to match feature files (default: '*.csv')
            
        Returns:
            Dictionary mapping file names to lists of retained column names
        """
        folder = Path(folder_path)
        feature_files = list(folder.glob(file_pattern))
        
        print(f"Found {len(feature_files)} feature files in {folder_path}")
        
        retained_columns = {}
        
        for file_path in feature_files:
            print(f"\n{'='*60}")
            print(f"Processing: {file_path.name}")
            print(f"{'='*60}")
            
            # Load feature file
            df = pd.read_csv(file_path)
            
            # Process features
            df_processed = self.process_features(df)
            
            # Store retained column names
            retained_columns[file_path.name] = df_processed.columns.tolist()
            
            # Save processed file
            output_path = folder / f"processed_{file_path.name}"
            df_processed.to_csv(output_path, index=False)
            print(f"Saved processed file to: {output_path}")
        
        # Save retained columns to JSON
        with open(output_file, 'w') as f:
            json.dump(retained_columns, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Saved retained column information to: {output_file}")
        print(f"{'='*60}")
        
        return retained_columns
    
    def get_processing_summary(self) -> pd.DataFrame:
        """
        Get summary of processing steps and their impact.
        
        Returns:
            DataFrame summarizing each processing step
        """
        summary_data = []
        for step_name, shape in self.processing_log:
            summary_data.append({
                'Step': step_name,
                'Rows': shape[0],
                'Columns': shape[1]
            })
        
        return pd.DataFrame(summary_data)


# Example usage
if __name__ == "__main__":
    # Initialize feature engineering pipeline
    pipeline = FeatureEngineeringPipeline(
        row_nan_ratio_threshold=0.3,
        column_nan_ratio_threshold=0.10,
        binary_variance_threshold=0.01,
        correlation_threshold=0.9,
        correlation_method='spearman'
    )
    
    # Example 1: Process single DataFrame
    df = pd.read_csv('features.csv')
    df_processed = pipeline.process_features(df, feature_prefix='features.')
    df_processed.to_csv('features_processed.csv', index=False)
    
    # Example 2: Process multiple feature files in a folder
    retained_cols = pipeline.process_dataset_folder(
        folder_path='dataset/',
        output_file='retained_columns.json',
        file_pattern='*.csv'
    )
    
    # Get processing summary
    summary = pipeline.get_processing_summary()
    print("\nProcessing Summary:")
    print(summary)
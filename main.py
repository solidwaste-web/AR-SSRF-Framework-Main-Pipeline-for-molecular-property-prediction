"""
AR-SSRF Framework Main Pipeline

This script is the main entry point for the
Autoregressive Semi-Supervised Random Forest (AR-SSRF) framework.

Workflow
--------
1. Data preprocessing
2. Molecular feature extraction
3. Feature engineering
4. Base classifier evaluation
5. Autoregressive semi-supervised training

Author: He Yan
Date: 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import custom modules
from data_preprocessing import DataPreprocessor
from comprehensive_molecular_feature_extraction import ComprehensiveFeatureExtractor
from feature_engineering_module import FeatureEngineeringPipeline
from base_classifier_evaluation import BaseClassifierEvaluator, BaseClassifierFactory
from imbalance_aware_dual_end_sampling_strategy import AdaptiveSampler
from autoregressive_iterative_learning import ARSSRFFramework


class ARSSRFPipeline:
    """Complete AR-SSRF framework pipeline."""
    
    def __init__(self, config: dict):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary containing all parameters
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', './output'))
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.preprocessor = None
        self.feature_extractor = None
        self.feature_engineer = None
        self.base_evaluator = None
        self.ar_ssrf_model = None
        
    def run_preprocessing(self, input_file: str) -> pd.DataFrame:
        """
        Step 1: Data preprocessing.
        
        Args:
            input_file: Path to raw data CSV
            
        Returns:
            Preprocessed DataFrame
        """
        print(f"\n{'='*80}")
        print("STEP 1: DATA PREPROCESSING")
        print(f"{'='*80}\n")
        
        self.preprocessor = DataPreprocessor(
            iqr_multiplier=self.config.get('iqr_multiplier', 1.5)
        )
        
        processed_df = self.preprocessor.process_dataset(
            input_file=input_file,
            output_file=str(self.output_dir / 'preprocessed_data.csv'),
            smiles_col=self.config.get('smiles_col', 'smiles'),
            solvent_col=self.config.get('solvent_col', 'solvent'),
            group_columns=self.config.get('group_columns', ['std_smiles', 'std_solvent']),
            label_columns=self.config.get('label_columns', ['label'])
        )
        
        return processed_df
    
    def run_feature_extraction(self, df: pd.DataFrame) -> dict:
        """
        Step 2: Extract molecular features.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Dictionary of feature DataFrames by type
        """
        print(f"\n{'='*80}")
        print("STEP 2: MOLECULAR FEATURE EXTRACTION")
        print(f"{'='*80}\n")
        
        feature_types = self.config.get('feature_types', 
                                       ComprehensiveFeatureExtractor.ALL_FEATURE_TYPES)
        
        self.feature_extractor = ComprehensiveFeatureExtractor(
            feature_types=feature_types,
            n_bits=self.config.get('n_bits', 2048),
            morgan_radius=self.config.get('morgan_radius', 2)
        )
        
        # Extract all feature types separately
        feature_output_dir = self.output_dir / 'features'
        feature_dfs = self.feature_extractor.batch_extract_all_types(
            df=df,
            smiles_column=self.config.get('smiles_col', 'smiles'),
            output_dir=str(feature_output_dir)
        )
        
        return feature_dfs
    
    def run_feature_engineering(self, feature_dfs: dict) -> dict:
        """
        Step 3: Feature engineering and selection.
        
        Args:
            feature_dfs: Dictionary of feature DataFrames
            
        Returns:
            Dictionary of processed feature DataFrames
        """
        print(f"\n{'='*80}")
        print("STEP 3: FEATURE ENGINEERING")
        print(f"{'='*80}\n")
        
        self.feature_engineer = FeatureEngineeringPipeline(
            row_nan_ratio_threshold=self.config.get('row_nan_threshold', 0.3),
            column_nan_ratio_threshold=self.config.get('col_nan_threshold', 0.10),
            binary_variance_threshold=self.config.get('variance_threshold', 0.01),
            correlation_threshold=self.config.get('correlation_threshold', 0.9),
            correlation_method=self.config.get('correlation_method', 'spearman')
        )
        
        processed_features = {}
        
        for feat_type, feat_df in feature_dfs.items():
            print(f"\nProcessing {feat_type}...")
            processed_df = self.feature_engineer.process_features(feat_df)
            processed_features[feat_type] = processed_df
            
            # Save processed features
            output_file = self.output_dir / 'features' / f'processed_{feat_type}'
            processed_df.to_csv(output_file, index=False)
        
        return processed_features
    
    def run_base_classifier_evaluation(self, 
                                      processed_features: dict,
                                      test_size: float = 0.2) -> tuple:
        """
        Step 4: Evaluate base classifiers with all feature sets.
        
        Args:
            processed_features: Dictionary of processed feature DataFrames
            test_size: Test set proportion
            
        Returns:
            Tuple of (best_model_name, best_feature_type, results_df)
        """
        print(f"\n{'='*80}")
        print("STEP 4: BASE CLASSIFIER EVALUATION")
        print(f"{'='*80}\n")
        
        self.base_evaluator = BaseClassifierEvaluator(
            random_state=self.config.get('random_state', 42)
        )
        
        # Prepare feature dictionary for evaluation
        feature_dict = {}
        label_col = self.config.get('label_columns', ['label'])[0]
        
        for feat_type, feat_df in processed_features.items():
            if label_col in feat_df.columns:
                X = feat_df.drop(columns=[label_col])
                y = feat_df[label_col]
                
                # Split into train and test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size,
                    random_state=self.config.get('random_state', 42),
                    stratify=y
                )
                
                feature_dict[feat_type] = (X_train, y_train)
        
        # Use first feature set's test data (all should have same samples)
        test_data = (X_test, y_test)
        
        # Evaluate all combinations
        model_list = self.config.get('base_models', 
                                    list(BaseClassifierFactory.SUPPORTED_MODELS.keys()))
        
        results_df = self.base_evaluator.evaluate_all_combinations(
            feature_dict=feature_dict,
            test_data=test_data,
            model_list=model_list
        )
        
        # Save results
        self.base_evaluator.save_results(
            str(self.output_dir / 'base_classifier_results.csv')
        )
        
        # Get best configuration
        best_config = self.base_evaluator.best_config
        
        return best_config['model'], best_config['feature_set'], results_df
    
    def run_ar_ssrf_training(self,
                            best_model_name: str,
                            best_feature_type: str,
                            processed_features: dict) -> ARSSRFFramework:
        """
        Step 5: Train AR-SSRF model with best configuration.
        
        Args:
            best_model_name: Name of best base classifier
            best_feature_type: Name of best feature set
            processed_features: Dictionary of processed features
            
        Returns:
            Trained AR-SSRF framework
        """
        print(f"\n{'='*80}")
        print("STEP 5: AR-SSRF TRAINING")
        print(f"{'='*80}\n")
        
        print(f"Using best configuration:")
        print(f"  Model: {best_model_name}")
        print(f"  Features: {best_feature_type}")
        
        # Get feature data
        feat_df = processed_features[best_feature_type]
        label_col = self.config.get('label_columns', ['label'])[0]
        
        # Separate labeled and unlabeled data
        labeled_df = feat_df[feat_df[label_col].notna()]
        unlabeled_df = feat_df[feat_df[label_col].isna()]
        
        X_labeled = labeled_df.drop(columns=[label_col])
        y_labeled = labeled_df[label_col]
        X_unlabeled = unlabeled_df.drop(columns=[label_col])
        
        # Split labeled data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_labeled, y_labeled,
            test_size=self.config.get('validation_split', 0.2),
            random_state=self.config.get('random_state', 42),
            stratify=y_labeled
        )
        
        # Create base classifier
        base_clf = BaseClassifierFactory.create_classifier(
            best_model_name,
            random_state=self.config.get('random_state', 42)
        )
        
        # Initialize AR-SSRF framework
        self.ar_ssrf_model = ARSSRFFramework(
            base_classifier=base_clf,
            threshold_high=self.config.get('threshold_high', 0.9),
            threshold_low=self.config.get('threshold_low', 0.1),
            max_iterations=self.config.get('max_iterations', 50),
            patience=self.config.get('patience', 3),
            min_delta=self.config.get('min_delta', 0.001),
            balance_ratio=self.config.get('balance_ratio', 1.0),
            random_state=self.config.get('random_state', 42)
        )
        
        # Train model
        self.ar_ssrf_model.fit(
            X_labeled=X_train,
            y_labeled=y_train,
            X_unlabeled=X_unlabeled,
            X_val=X_val,
            y_val=y_val
        )
        
        # Save model and history
        self.ar_ssrf_model.save(
            model_path=str(self.output_dir / 'ar_ssrf_model.pkl'),
            history_path=str(self.output_dir / 'training_history.csv')
        )
        
        return self.ar_ssrf_model
    
    def run_complete_pipeline(self, input_file: str):
        """
        Execute complete AR-SSRF pipeline.
        
        Args:
            input_file: Path to raw data CSV
        """
        print(f"\n{'='*80}")
        print("AR-SSRF FRAMEWORK - COMPLETE PIPELINE")
        print(f"{'='*80}\n")
        
        # Step 1: Preprocessing
        processed_df = self.run_preprocessing(input_file)
        
        # Step 2: Feature extraction
        feature_dfs = self.run_feature_extraction(processed_df)
        
        # Step 3: Feature engineering
        processed_features = self.run_feature_engineering(feature_dfs)
        
        # Step 4: Base classifier evaluation
        best_model, best_features, results = self.run_base_classifier_evaluation(
            processed_features
        )
        
        # Step 5: AR-SSRF training
        ar_ssrf_model = self.run_ar_ssrf_training(
            best_model, best_features, processed_features
        )
        
        print(f"\n{'='*80}")
        print("PIPELINE COMPLETE")
        print(f"{'='*80}\n")
        print(f"All outputs saved to: {self.output_dir}")


def load_config(config_file: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def create_default_config() -> dict:
    """Create default configuration."""
    return {
        'output_dir': './output',
        'random_state': 42,
        
        # Preprocessing
        'smiles_col': 'smiles',
        'solvent_col': 'solvent',
        'label_columns': ['label'],
        'group_columns': ['std_smiles', 'std_solvent'],
        'iqr_multiplier': 1.5,
        
        # Feature extraction
        'n_bits': 2048,
        'morgan_radius': 2,
        
        # Feature engineering
        'row_nan_threshold': 0.3,
        'col_nan_threshold': 0.10,
        'variance_threshold': 0.01,
        'correlation_threshold': 0.9,
        'correlation_method': 'spearman',
        
        # Base classifier evaluation
        'base_models': ['logistic', 'random_forest', 'xgboost', 'svc', 'knn'],
        
        # AR-SSRF training
        'threshold_high': 0.9,
        'threshold_low': 0.1,
        'max_iterations': 50,
        'patience': 3,
        'min_delta': 0.001,
        'balance_ratio': 1.0,
        'validation_split': 0.2
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AR-SSRF Framework - Autoregressive Semi-Supervised Random Forest'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration JSON file (optional)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output directory (default: ./output)'
    )
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
    
    # Override output directory if specified
    config['output_dir'] = args.output
    
    # Initialize and run pipeline
    pipeline = ARSSRFPipeline(config)
    pipeline.run_complete_pipeline(args.input)


if __name__ == "__main__":
    main()
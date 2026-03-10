"""
Data Preprocessing Module for AR-SSRF Framework

This module handles the initial data cleaning, SMILES standardization,
and duplicate sample aggregation as described in the paper's methodology.

Main functionalities:
1. SMILES standardization using RDKit
2. Duplicate sample identification and aggregation
3. Outlier removal using IQR method
4. Data quality control and validation

Author: [HE YAN]
Date: [2026.3.10]
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
import warnings
from rdkit import Chem

warnings.filterwarnings('ignore')


def get_std_smiles(smiles: str) -> Optional[str]:
    """
    Standardize SMILES string using RDKit canonical representation.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Standardized canonical SMILES string, or None if invalid
        
    Example:
        >>> get_std_smiles('CC(=O)O')
        'CC(=O)O'
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def aggregate_duplicate_data(df: pd.DataFrame, 
                            label_columns: List[str],
                            iqr_multiplier: float = 1.5) -> Dict:
    """
    Aggregate duplicate data entries using IQR-based outlier removal.
    
    Strategy for handling duplicates:
    - Single value: return as-is
    - Two values: return mean
    - Three or more values: remove outliers using IQR method, then return mean
    
    Args:
        df: DataFrame containing duplicate entries for the same compound
        label_columns: List of label column names to aggregate
        iqr_multiplier: Multiplier for IQR-based outlier detection (default: 1.5)
        
    Returns:
        Dictionary containing aggregated values and data IDs
        
    Example:
        For a compound with 4 measurements [10, 12, 11, 50]:
        - Q1 = 10.5, Q3 = 13.5, IQR = 3
        - Lower bound = 10.5 - 1.5*3 = 6
        - Upper bound = 13.5 + 1.5*3 = 18
        - Filtered values: [10, 12, 11]
        - Final value: mean([10, 12, 11]) = 11
    """
    data_dict = {}
    
    # Identify valid label columns present in the DataFrame
    valid_label_columns = [col for col in label_columns if col in df.columns]
    
    for col in df.columns:
        # Generate corresponding data_id column name
        data_id_name = col.replace('value', 'data_id')
        
        # Filter out missing values
        valid_df = df.dropna(subset=[col])
        valid_values = valid_df[col]
        
        # Skip non-label columns - keep first valid value
        if col not in valid_label_columns:
            if len(valid_values) > 0:
                data_dict[col] = valid_values.iloc[0]
            continue
        
        # Aggregate label values based on count
        count = len(valid_values)
        
        if count == 0:
            # No valid values - skip this column
            continue
            
        elif count == 1:
            # Single value - use as-is
            data_dict[col] = valid_values.iloc[0]
            
        elif count == 2:
            # Two values - use mean
            data_dict[col] = valid_values.mean()
            
        else:
            # Three or more values - use IQR method for outlier removal
            q1 = valid_values.quantile(0.25)
            q3 = valid_values.quantile(0.75)
            iqr = q3 - q1
            
            # Calculate bounds
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            
            # Filter outliers
            filtered_values = valid_values[
                (valid_values >= lower_bound) & (valid_values <= upper_bound)
            ]
            
            # Use mean of filtered values
            data_dict[col] = filtered_values.mean()
        
        # Store data ID mapping for traceability
        data_id_dict = {
            row['data_id']: row[col] 
            for _, row in valid_df.iterrows()
        }
        if data_id_dict:
            data_dict[data_id_name] = data_id_dict
    
    return data_dict


def get_dfs_dict(df: pd.DataFrame, prefix: str) -> Dict[str, pd.DataFrame]:
    """
    Group DataFrame columns by hierarchical prefix structure.
    
    This function organizes columns with multi-level naming conventions
    (e.g., 'features.descriptor.property') into separate DataFrames.
    
    Args:
        df: Input DataFrame with hierarchically named columns
        prefix: Column prefix for grouping (must contain at least 2 dots)
        
    Returns:
        Dictionary mapping group names to sub-DataFrames
        
    Example:
        For columns like 'features.rdkit.MolWt', 'features.rdkit.LogP',
        'features.mordred.TPSA':
        >>> dfs = get_dfs_dict(df, 'features.')
        >>> dfs.keys()
        dict_keys(['rdkit', 'mordred'])
    """
    # Filter columns starting with the specified prefix
    cols = [col for col in df.columns if col.startswith(prefix)]
    dfs = {}
    
    # Extract first level of prefix
    prefix1 = prefix.split('.')[0]
    
    # Special handling for 'features' prefix - use two-level grouping
    if prefix1 == 'features':
        # Extract second and third level prefixes (e.g., 'rdkit.MolWt')
        prefix2and3_list = list(set([
            '.'.join(col.split('.')[1:3]) for col in cols
        ]))
        
        for prefix2and3 in prefix2and3_list:
            dfs[prefix2and3] = df[cols].filter(
                regex=f'{prefix1}.{prefix2and3}', axis=1
            )
        return dfs
    
    # Standard grouping by second level prefix
    prefix2_list = list(set([col.split('.')[1] for col in cols]))
    
    for prefix2 in prefix2_list:
        dfs[prefix2] = df[cols].filter(
            regex=f'{prefix1}.{prefix2}', axis=1
        )
    
    return dfs


class DataPreprocessor:
    """
    Main data preprocessing pipeline for fluorescence dataset.
    
    This class implements the complete data cleaning and standardization
    procedures including:
    - SMILES standardization for both solute and solvent
    - Duplicate sample identification and aggregation
    - Data quality filtering and validation
    """
    
    def __init__(self, iqr_multiplier: float = 1.5):
        """
        Initialize the data preprocessor.
        
        Args:
            iqr_multiplier: Multiplier for IQR-based outlier detection
        """
        self.iqr_multiplier = iqr_multiplier
    
    def standardize_smiles_columns(self, 
                                   df: pd.DataFrame,
                                   smiles_col: str = 'smiles',
                                   solvent_col: str = 'solvent') -> pd.DataFrame:
        """
        Standardize SMILES strings for both solute and solvent molecules.
        
        Args:
            df: Input DataFrame
            smiles_col: Name of solute SMILES column
            solvent_col: Name of solvent SMILES column
            
        Returns:
            DataFrame with standardized SMILES columns added
        """
        print("Standardizing SMILES strings...")
        
        # Standardize solute SMILES
        tqdm.pandas(desc="Processing solute SMILES")
        df['std_smiles'] = df[smiles_col].progress_apply(get_std_smiles)
        
        # Standardize solvent SMILES
        tqdm.pandas(desc="Processing solvent SMILES")
        df['std_solvent'] = df[solvent_col].progress_apply(get_std_smiles)
        
        # Remove rows with invalid SMILES
        initial_count = len(df)
        df = df.dropna(subset=['std_smiles', 'std_solvent'])
        removed_count = initial_count - len(df)
        
        print(f"Removed {removed_count} samples with invalid SMILES")
        print(f"Retained {len(df)} valid samples")
        
        return df
    
    def merge_duplicates(self,
                        df: pd.DataFrame,
                        group_columns: List[str],
                        label_columns: List[str]) -> pd.DataFrame:
        """
        Identify and merge duplicate samples using aggregation strategy.
        
        Args:
            df: Input DataFrame
            group_columns: Columns to identify duplicates (e.g., ['std_smiles', 'std_solvent'])
            label_columns: Label columns to aggregate
            
        Returns:
            DataFrame with duplicates merged
        """
        print("\nMerging duplicate samples...")
        
        # Group by identifier columns
        grouped = df.groupby(group_columns)
        
        # Aggregate duplicates
        aggregated_data = []
        for name, group in tqdm(grouped, desc="Processing groups"):
            if len(group) > 1:
                # Multiple entries - aggregate using IQR method
                agg_dict = aggregate_duplicate_data(
                    group, 
                    label_columns, 
                    self.iqr_multiplier
                )
            else:
                # Single entry - use as-is
                agg_dict = group.iloc[0].to_dict()
            
            aggregated_data.append(agg_dict)
        
        result_df = pd.DataFrame(aggregated_data)
        
        print(f"Original samples: {len(df)}")
        print(f"After merging duplicates: {len(result_df)}")
        print(f"Merged {len(df) - len(result_df)} duplicate entries")
        
        return result_df
    
    def process_dataset(self,
                       input_file: str,
                       output_file: str,
                       smiles_col: str = 'smiles',
                       solvent_col: str = 'solvent',
                       group_columns: Optional[List[str]] = None,
                       label_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Complete preprocessing pipeline from raw data to clean dataset.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to save processed CSV file
            smiles_col: Name of solute SMILES column
            solvent_col: Name of solvent SMILES column
            group_columns: Columns to identify duplicates
            label_columns: Label columns to aggregate
            
        Returns:
            Processed DataFrame
        """
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} samples")
        
        # Standardize SMILES
        df = self.standardize_smiles_columns(df, smiles_col, solvent_col)
        
        # Merge duplicates if specified
        if group_columns and label_columns:
            df = self.merge_duplicates(df, group_columns, label_columns)
        
        # Save processed data
        print(f"\nSaving processed data to {output_file}...")
        df.to_csv(output_file, index=False)
        print("Preprocessing complete!")
        
        return df


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor(iqr_multiplier=1.5)
    
    # Process dataset
    processed_df = preprocessor.process_dataset(
        input_file='raw_data.csv',
        output_file='processed_data.csv',
        smiles_col='smiles',
        solvent_col='solvent',
        group_columns=['std_smiles', 'std_solvent'],
        label_columns=['fluorescence_value', 'quantum_yield']
    )
    
    print(f"\nFinal dataset shape: {processed_df.shape}")
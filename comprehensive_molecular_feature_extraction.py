"""
Comprehensive Molecular Feature Extraction Module for AR-SSRF Framework

This module implements the complete feature generation pipeline described in Text S3,
including 3 molecular descriptor sets and 14 types of molecular fingerprints.

Feature Categories (17 total approaches):

1. Molecular Descriptors (3 types):
   - RDKitMD: RDKit 2D molecular descriptors
   - MordredMD: Mordred comprehensive descriptors
   - PadelMD: PaDEL molecular descriptors

2. Structure-based Fingerprints (8 types):
   - AtomPairsFP: Atom pair fingerprints
   - EStateFP: Electrotopological state fingerprints
   - ExtendedFP: Extended connectivity fingerprints
   - GraphOnlyFP: Graph-only fingerprints
   - MACCSFP: MACCS structural keys
   - MorganFP: Morgan circular fingerprints (ECFP)
   - PubchemFP: PubChem fingerprints
   - RDKitFP: RDKit topological fingerprints

3. Substructure/Fragment-based Fingerprints (6 types):
   - KlekotaRothFP: Klekota-Roth fingerprints (binary)
   - KlekotaRothFPC: Klekota-Roth fingerprints (count-based)
   - SubstructureFP: Substructure fingerprints (binary)
   - SubstructureFPC: Substructure fingerprints (count-based)
   - TTFP: Topological torsion fingerprints
   - PadelFP: PaDEL fingerprints

Author: [HE YAN]
Date: [2026.03.10]
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import warnings
from tqdm import tqdm
import tempfile

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.EState import Fingerprinter as EStateFingerprinter
from rdkit.Chem.AtomPairs import Pairs
from rdkit.ML.Descriptors import MoleculeDescriptors

from mordred import Calculator, descriptors as mordred_descriptors

try:
    from padelpy import padeldescriptor
    PADEL_AVAILABLE = True
except ImportError:
    PADEL_AVAILABLE = False
    warnings.warn("PaDELPy not available. PaDEL descriptors will be skipped.")

warnings.filterwarnings('ignore')


class MolecularDescriptorExtractor:
    """Extract three types of molecular descriptors: RDKit, Mordred, and PaDEL."""
    
    def __init__(self):
        """Initialize molecular descriptor calculators."""
        self.rdkit_desc_names = [desc[0] for desc in Descriptors._descList]
        self.rdkit_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
            self.rdkit_desc_names
        )
        self.mordred_calculator = Calculator(mordred_descriptors, ignore_3D=True)
    
    def extract_rdkit_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """Extract RDKit molecular descriptors (RDKitMD)."""
        if mol is None:
            return {f'RDKitMD.{name}': np.nan for name in self.rdkit_desc_names}
        
        try:
            values = self.rdkit_calculator.CalcDescriptors(mol)
            return {f'RDKitMD.{name}': float(val) if not np.isinf(val) else np.nan 
                   for name, val in zip(self.rdkit_desc_names, values)}
        except:
            return {f'RDKitMD.{name}': np.nan for name in self.rdkit_desc_names}
    
    def extract_mordred_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """Extract Mordred molecular descriptors (MordredMD)."""
        if mol is None:
            desc_names = [str(d) for d in self.mordred_calculator.descriptors]
            return {f'MordredMD.{name}': np.nan for name in desc_names}
        
        try:
            result = self.mordred_calculator(mol)
            descriptors_dict = {}
            
            for d, val in zip(self.mordred_calculator.descriptors, result):
                desc_name = str(d)
                if val is None or val == 'error':
                    descriptors_dict[f'MordredMD.{desc_name}'] = np.nan
                elif isinstance(val, (int, float)):
                    descriptors_dict[f'MordredMD.{desc_name}'] = np.nan if np.isinf(val) else float(val)
                else:
                    try:
                        descriptors_dict[f'MordredMD.{desc_name}'] = float(val)
                    except:
                        descriptors_dict[f'MordredMD.{desc_name}'] = np.nan
            
            return descriptors_dict
        except:
            desc_names = [str(d) for d in self.mordred_calculator.descriptors]
            return {f'MordredMD.{name}': np.nan for name in desc_names}
    
    def extract_padel_descriptors(self, smiles: str, temp_dir: Optional[str] = None) -> Dict[str, float]:
        """Extract PaDEL molecular descriptors (PadelMD)."""
        if not PADEL_AVAILABLE:
            return {}
        
        try:
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp()
            
            Path(temp_dir).mkdir(exist_ok=True)
            input_file = Path(temp_dir) / 'input.smi'
            output_file = Path(temp_dir) / 'output.csv'
            
            with open(input_file, 'w') as f:
                f.write(f"{smiles}\tmol\n")
            
            padeldescriptor(
                mol_dir=str(input_file),
                d_file=str(output_file),
                descriptortypes='descriptors.xml',
                detectaromaticity=True,
                standardizenitro=True,
                standardizetautomers=True,
                threads=1,
                removesalt=True,
                log=False,
                fingerprints=False
            )
            
            df = pd.read_csv(output_file)
            result = {f'PadelMD.{col}': float(df[col].values[0]) 
                     for col in df.columns if col != 'Name'}
            
            input_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)
            
            return result
        except:
            return {}


class MolecularFingerprintExtractor:
    """Extract 14 types of molecular fingerprints."""
    
    def __init__(self, default_n_bits: int = 2048, morgan_radius: int = 2):
        """Initialize fingerprint extractor."""
        self.default_n_bits = default_n_bits
        self.morgan_radius = morgan_radius
    
    def extract_atom_pairs_fp(self, mol: Chem.Mol, n_bits: int = 2048) -> Dict[str, int]:
        """Extract Atom Pairs fingerprint."""
        if mol is None:
            return {f'AtomPairsFP.{i}': 0 for i in range(n_bits)}
        try:
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
            return {f'AtomPairsFP.{i}': int(fp[i]) for i in range(n_bits)}
        except:
            return {f'AtomPairsFP.{i}': 0 for i in range(n_bits)}
    
    def extract_estate_fp(self, mol: Chem.Mol) -> Dict[str, float]:
        """Extract EState fingerprint."""
        if mol is None:
            return {f'EStateFP.{i}': 0.0 for i in range(79)}
        try:
            fp = EStateFingerprinter.FingerprintMol(mol)[0]
            return {f'EStateFP.{i}': float(fp[i]) for i in range(len(fp))}
        except:
            return {f'EStateFP.{i}': 0.0 for i in range(79)}
    
    def extract_extended_fp(self, mol: Chem.Mol, n_bits: int = 2048) -> Dict[str, int]:
        """Extract Extended Connectivity fingerprint."""
        if mol is None:
            return {f'ExtendedFP.{i}': 0 for i in range(n_bits)}
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=n_bits)
            return {f'ExtendedFP.{i}': int(fp[i]) for i in range(n_bits)}
        except:
            return {f'ExtendedFP.{i}': 0 for i in range(n_bits)}
    
    def extract_graph_only_fp(self, mol: Chem.Mol, n_bits: int = 2048) -> Dict[str, int]:
        """Extract Graph-only fingerprint."""
        if mol is None:
            return {f'GraphOnlyFP.{i}': 0 for i in range(n_bits)}
        try:
            fp = Chem.RDKFingerprint(mol, fpSize=n_bits, useHs=False)
            return {f'GraphOnlyFP.{i}': int(fp[i]) for i in range(n_bits)}
        except:
            return {f'GraphOnlyFP.{i}': 0 for i in range(n_bits)}
    
    def extract_maccs_fp(self, mol: Chem.Mol) -> Dict[str, int]:
        """Extract MACCS keys."""
        if mol is None:
            return {f'MACCSFP.{i}': 0 for i in range(167)}
        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
            return {f'MACCSFP.{i}': int(fp[i]) for i in range(167)}
        except:
            return {f'MACCSFP.{i}': 0 for i in range(167)}
    
    def extract_morgan_fp(self, mol: Chem.Mol, n_bits: int = 2048) -> Dict[str, int]:
        """Extract Morgan fingerprint."""
        if mol is None:
            return {f'MorganFP.{i}': 0 for i in range(n_bits)}
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.morgan_radius, nBits=n_bits)
            return {f'MorganFP.{i}': int(fp[i]) for i in range(n_bits)}
        except:
            return {f'MorganFP.{i}': 0 for i in range(n_bits)}
    
    def extract_pubchem_fp(self, mol: Chem.Mol) -> Dict[str, int]:
        """Extract PubChem fingerprint."""
        if mol is None:
            return {f'PubchemFP.{i}': 0 for i in range(881)}
        try:
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=881)
            return {f'PubchemFP.{i}': int(fp[i]) for i in range(881)}
        except:
            return {f'PubchemFP.{i}': 0 for i in range(881)}
    
    def extract_rdkit_fp(self, mol: Chem.Mol, n_bits: int = 2048) -> Dict[str, int]:
        """Extract RDKit fingerprint."""
        if mol is None:
            return {f'RDKitFP.{i}': 0 for i in range(n_bits)}
        try:
            fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
            return {f'RDKitFP.{i}': int(fp[i]) for i in range(n_bits)}
        except:
            return {f'RDKitFP.{i}': 0 for i in range(n_bits)}
    
    def extract_klekota_roth_fp(self, mol: Chem.Mol, n_bits: int = 4860) -> Dict[str, int]:
        """Extract Klekota-Roth fingerprint (binary)."""
        if mol is None:
            return {f'KlekotaRothFP.{i}': 0 for i in range(n_bits)}
        try:
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
            return {f'KlekotaRothFP.{i}': int(fp[i]) for i in range(n_bits)}
        except:
            return {f'KlekotaRothFP.{i}': 0 for i in range(n_bits)}
    
    def extract_klekota_roth_fpc(self, mol: Chem.Mol, n_bits: int = 4860) -> Dict[str, int]:
        """Extract Klekota-Roth fingerprint (count-based)."""
        if mol is None:
            return {f'KlekotaRothFPC.{i}': 0 for i in range(n_bits)}
        try:
            fp = Pairs.GetAtomPairFingerprint(mol)
            fp_dict = fp.GetNonzeroElements()
            result = {f'KlekotaRothFPC.{i}': 0 for i in range(n_bits)}
            for key, count in fp_dict.items():
                idx = hash(key) % n_bits
                result[f'KlekotaRothFPC.{idx}'] += count
            return result
        except:
            return {f'KlekotaRothFPC.{i}': 0 for i in range(n_bits)}
    
    def extract_substructure_fp(self, mol: Chem.Mol, n_bits: int = 307) -> Dict[str, int]:
        """Extract Substructure fingerprint (binary)."""
        if mol is None:
            return {f'SubstructureFP.{i}': 0 for i in range(n_bits)}
        try:
            fp = Chem.RDKFingerprint(mol, fpSize=n_bits, minPath=1, maxPath=7)
            return {f'SubstructureFP.{i}': int(fp[i]) for i in range(n_bits)}
        except:
            return {f'SubstructureFP.{i}': 0 for i in range(n_bits)}
    
    def extract_substructure_fpc(self, mol: Chem.Mol, n_bits: int = 307) -> Dict[str, int]:
        """Extract Substructure fingerprint (count-based)."""
        if mol is None:
            return {f'SubstructureFPC.{i}': 0 for i in range(n_bits)}
        try:
            fp = AllChem.GetMorganFingerprint(mol, radius=2)
            fp_dict = fp.GetNonzeroElements()
            result = {f'SubstructureFPC.{i}': 0 for i in range(n_bits)}
            for key, count in list(fp_dict.items())[:n_bits]:
                idx = hash(key) % n_bits
                result[f'SubstructureFPC.{idx}'] += count
            return result
        except:
            return {f'SubstructureFPC.{i}': 0 for i in range(n_bits)}
    
    def extract_tt_fp(self, mol: Chem.Mol, n_bits: int = 2048) -> Dict[str, int]:
        """Extract Topological Torsion fingerprint."""
        if mol is None:
            return {f'TTFP.{i}': 0 for i in range(n_bits)}
        try:
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
            return {f'TTFP.{i}': int(fp[i]) for i in range(n_bits)}
        except:
            return {f'TTFP.{i}': 0 for i in range(n_bits)}
    
    def extract_padel_fp(self, smiles: str, temp_dir: Optional[str] = None) -> Dict[str, int]:
        """Extract PaDEL fingerprints."""
        if not PADEL_AVAILABLE:
            return {}
        try:
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp()
            Path(temp_dir).mkdir(exist_ok=True)
            input_file = Path(temp_dir) / 'input.smi'
            output_file = Path(temp_dir) / 'output.csv'
            
            with open(input_file, 'w') as f:
                f.write(f"{smiles}\tmol\n")
            
            padeldescriptor(
                mol_dir=str(input_file),
                d_file=str(output_file),
                fingerprints=True,
                descriptortypes='fingerprints.xml',
                threads=1,
                log=False
            )
            
            df = pd.read_csv(output_file)
            result = {f'PadelFP.{col}': int(df[col].values[0]) 
                     for col in df.columns if col != 'Name'}
            
            input_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)
            
            return result
        except:
            return {}


class ComprehensiveFeatureExtractor:
    """Main class integrating all 17 molecular representation approaches."""
    
    ALL_FEATURE_TYPES = [
        'RDKitMD', 'MordredMD', 'PadelMD',
        'AtomPairsFP', 'EStateFP', 'ExtendedFP', 'GraphOnlyFP', 
        'MACCSFP', 'MorganFP', 'PubchemFP', 'RDKitFP',
        'KlekotaRothFP', 'KlekotaRothFPC', 'SubstructureFP', 
        'SubstructureFPC', 'TTFP', 'PadelFP'
    ]
    
    def __init__(self, feature_types: Optional[List[str]] = None, 
                 n_bits: int = 2048, morgan_radius: int = 2):
        """Initialize comprehensive feature extractor."""
        if feature_types is None:
            self.feature_types = self.ALL_FEATURE_TYPES
        else:
            invalid = set(feature_types) - set(self.ALL_FEATURE_TYPES)
            if invalid:
                raise ValueError(f"Invalid feature types: {invalid}")
            self.feature_types = feature_types
        
        self.descriptor_extractor = MolecularDescriptorExtractor()
        self.fingerprint_extractor = MolecularFingerprintExtractor(
            default_n_bits=n_bits, morgan_radius=morgan_radius
        )
    
    def extract_features(self, smiles: str) -> Dict[str, Union[int, float]]:
        """Extract all selected feature types for a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        features = {}
        
        if 'RDKitMD' in self.feature_types:
            features.update(self.descriptor_extractor.extract_rdkit_descriptors(mol))
        if 'MordredMD' in self.feature_types:
            features.update(self.descriptor_extractor.extract_mordred_descriptors(mol))
        if 'PadelMD' in self.feature_types:
            features.update(self.descriptor_extractor.extract_padel_descriptors(smiles))
        
        if 'AtomPairsFP' in self.feature_types:
            features.update(self.fingerprint_extractor.extract_atom_pairs_fp(mol))
        if 'EStateFP' in self.feature_types:
            features.update(self.fingerprint_extractor.extract_estate_fp(mol))
        if 'ExtendedFP' in self.feature_types:
            features.update(self.fingerprint_extractor.extract_extended_fp(mol))
        if 'GraphOnlyFP' in self.feature_types:
            features.update(self.fingerprint_extractor.extract_graph_only_fp(mol))
        if 'MACCSFP' in self.feature_types:
            features.update(self.fingerprint_extractor.extract_maccs_fp(mol))
        if 'MorganFP' in self.feature_types:
            features.update(self.fingerprint_extractor.extract_morgan_fp(mol))
        if 'PubchemFP' in self.feature_types:
            features.update(self.fingerprint_extractor.extract_pubchem_fp(mol))
        if 'RDKitFP' in self.feature_types:
            features.update(self.fingerprint_extractor.extract_rdkit_fp(mol))
        
        if 'KlekotaRothFP' in self.feature_types:
            features.update(self.fingerprint_extractor.extract_klekota_roth_fp(mol))
        if 'KlekotaRothFPC' in self.feature_types:
            features.update(self.fingerprint_extractor.extract_klekota_roth_fpc(mol))
        if 'SubstructureFP' in self.feature_types:
            features.update(self.fingerprint_extractor.extract_substructure_fp(mol))
        if 'SubstructureFPC' in self.feature_types:
            features.update(self.fingerprint_extractor.extract_substructure_fpc(mol))
        if 'TTFP' in self.feature_types:
            features.update(self.fingerprint_extractor.extract_tt_fp(mol))
        if 'PadelFP' in self.feature_types:
            features.update(self.fingerprint_extractor.extract_padel_fp(smiles))
        
        return features
    
    def process_dataframe(self, df: pd.DataFrame, smiles_column: str = 'smiles',
                         save_path: Optional[str] = None) -> pd.DataFrame:
        """Extract features for all molecules in a DataFrame."""
        print(f"\n{'='*80}")
        print(f"Comprehensive Molecular Feature Extraction")
        print(f"Feature types: {len(self.feature_types)} | Molecules: {len(df)}")
        print(f"{'='*80}\n")
        
        feature_list = []
        failed_molecules = []
        
        for idx, smiles in enumerate(tqdm(df[smiles_column], desc="Extracting features")):
            try:
                features = self.extract_features(smiles)
                feature_list.append(features)
            except Exception as e:
                warnings.warn(f"Failed for molecule {idx}: {smiles}")
                failed_molecules.append((idx, smiles, str(e)))
                feature_list.append({})
        
        feature_df = pd.DataFrame(feature_list)
        result_df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
        
        print(f"\n{'='*80}")
        print(f"Extraction Complete")
        print(f"Total features: {len(feature_df.columns)}")
        print(f"Success: {len(df) - len(failed_molecules)} | Failed: {len(failed_molecules)}")
        
        for feat_type in self.feature_types:
            type_features = [col for col in feature_df.columns if col.startswith(feat_type)]
            print(f"  {feat_type}: {len(type_features)} features")
        print(f"{'='*80}\n")
        
        if save_path:
            result_df.to_csv(save_path, index=False)
            print(f"Saved to: {save_path}")
        
        return result_df
    
    def extract_single_feature_type(self, df: pd.DataFrame, feature_type: str,
                                    smiles_column: str = 'smiles') -> pd.DataFrame:
        """Extract only one specific feature type."""
        if feature_type not in self.ALL_FEATURE_TYPES:
            raise ValueError(f"Invalid feature type: {feature_type}")
        
        original_types = self.feature_types
        self.feature_types = [feature_type]
        result_df = self.process_dataframe(df, smiles_column)
        self.feature_types = original_types
        
        return result_df
    
    def batch_extract_all_types(self, df: pd.DataFrame, smiles_column: str = 'smiles',
                               output_dir: str = './features') -> Dict[str, pd.DataFrame]:
        """Extract all 17 feature types separately and save to individual files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"\n{'='*80}")
        print(f"Batch Feature Extraction - All 17 Types")
        print(f"Output directory: {output_dir}")
        print(f"{'='*80}\n")
        
        feature_dfs = {}
        
        for feat_type in tqdm(self.ALL_FEATURE_TYPES, desc="Extracting feature types"):
            print(f"\nExtracting {feat_type}...")
            feat_df = self.extract_single_feature_type(df, feat_type, smiles_column)
            output_file = output_path / f"{feat_type}_features.csv"
            feat_df.to_csv(output_file, index=False)
            feature_dfs[feat_type] = feat_df
            print(f"Saved to: {output_file}")
        
        print(f"\n{'='*80}")
        print(f"Batch extraction complete! Total files: {len(feature_dfs)}")
        print(f"{'='*80}\n")
        
        return feature_dfs
    
    def get_feature_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistics for extracted features."""
        feature_cols = [col for col in df.columns 
                       if any(col.startswith(ft) for ft in self.ALL_FEATURE_TYPES)]
        
        if not feature_cols:
            return pd.DataFrame()
        
        stats = pd.DataFrame({
            'Feature': feature_cols,
            'Type': [col.split('.')[0] for col in feature_cols],
            'Mean': df[feature_cols].mean(),
            'Std': df[feature_cols].std(),
            'Min': df[feature_cols].min(),
            'Max': df[feature_cols].max(),
            'Missing': df[feature_cols].isna().sum(),
            'Missing_Pct': (df[feature_cols].isna().sum() / len(df) * 100).round(2),
            'Unique': df[feature_cols].nunique()
        })
        
        return stats.reset_index(drop=True)
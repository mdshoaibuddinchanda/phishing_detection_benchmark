"""Data splitting with deterministic random seed."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify_col: str = 'label'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test with stratification.
    
    Args:
        df: Input DataFrame
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        stratify_col: Column to stratify on
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    np.random.seed(random_seed)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Stratified split
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for label in df[stratify_col].unique():
        label_df = df[df[stratify_col] == label]
        n = len(label_df)
        
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_dfs.append(label_df.iloc[:n_train])
        val_dfs.append(label_df.iloc[n_train:n_train + n_val])
        test_dfs.append(label_df.iloc[n_train + n_val:])
    
    train_df = pd.concat(train_dfs).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    val_df = pd.concat(val_dfs).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    test_df = pd.concat(test_dfs).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"Split complete:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Save train/val/test splits to CSV files.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Directory to save splits
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    print(f"Saved splits to {output_dir}/")

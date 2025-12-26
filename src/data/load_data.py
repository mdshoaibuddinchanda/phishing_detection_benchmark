"""Data loading utilities for phishing detection dataset."""

import pandas as pd
from pathlib import Path
from typing import Tuple


def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Load raw phishing email dataset.
    
    Args:
        data_path: Path to raw CSV file
        
    Returns:
        DataFrame with text and labels
    """
    df = pd.read_csv(data_path)
    
    # Validate required columns
    required_columns = ['text', 'label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Basic sanity checks
    print(f"Loaded {len(df)} samples")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    return df


def load_processed_data(train_path: str, val_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed train/val/test splits.
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def validate_dataset(df: pd.DataFrame, text_col: str = 'text', label_col: str = 'label') -> bool:
    """
    Validate dataset integrity.
    
    Args:
        df: Dataset DataFrame
        text_col: Name of text column
        label_col: Name of label column
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check for missing values
    if df[text_col].isnull().any():
        raise ValueError("Found missing values in text column")
    
    if df[label_col].isnull().any():
        raise ValueError("Found missing values in label column")
    
    # Check labels are binary
    unique_labels = df[label_col].unique()
    if len(unique_labels) != 2:
        raise ValueError(f"Expected binary labels, found: {unique_labels}")
    
    # Check for empty strings
    empty_texts = (df[text_col].str.strip() == '').sum()
    if empty_texts > 0:
        print(f"Warning: Found {empty_texts} empty text entries")
    
    return True

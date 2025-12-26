"""Text preprocessing for phishing emails."""

import re
import pandas as pd
from typing import List


def clean_text(text: str) -> str:
    """
    Clean email text with minimal preprocessing.
    
    Args:
        text: Raw email text
        
    Returns:
        Cleaned text
    """
    # Handle missing/NaN values
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s\.\,\!\?\-]', '', text)
    
    return text


def preprocess_dataframe(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """
    Preprocess entire DataFrame.
    
    Args:
        df: Input DataFrame
        text_col: Name of text column
        
    Returns:
        DataFrame with cleaned text
    """
    df = df.copy()
    
    print(f"Preprocessing {len(df)} samples...")
    
    # Clean text
    df[text_col] = df[text_col].apply(clean_text)
    
    # Remove empty texts after cleaning
    initial_count = len(df)
    df = df[df[text_col].str.strip() != '']
    final_count = len(df)
    
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} empty samples after cleaning")
    
    return df


def preprocess_dataset(input_path: str, output_path: str, text_col: str = 'text') -> None:
    """
    Preprocess dataset and save to file.
    
    Args:
        input_path: Path to raw CSV
        output_path: Path to save preprocessed CSV
        text_col: Name of text column
    """
    df = pd.read_csv(input_path)
    df_clean = preprocess_dataframe(df, text_col)
    df_clean.to_csv(output_path, index=False)
    print(f"Saved preprocessed data to {output_path}")

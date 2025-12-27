"""Model evaluation utilities."""

import torch
import numpy as np
from typing import Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import pandas as pd
from tqdm import tqdm


def evaluate_model(
    model_dir: str,
    test_df: pd.DataFrame,
    text_col: str = 'text',
    label_col: str = 'label',
    max_length: int = 256,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate trained model on test set.
    
    Args:
        model_dir: Directory containing saved model
        test_df: Test DataFrame
        text_col: Name of text column
        label_col: Name of label column
        max_length: Maximum sequence length
        batch_size: Batch size for inference
        device: Device to run inference on
        
    Returns:
        Tuple of (predictions, true_labels)
    """
    print(f"Loading model from {model_dir}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # Prepare data
    texts = test_df[text_col].tolist()
    labels = test_df[label_col].tolist()
    
    predictions = []
    
    print(f"Running inference on {len(texts)} samples...")
    
    # Batch inference
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get predictions
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(batch_preds)
    
    predictions = np.array(predictions)
    true_labels = np.array(labels)
    
    return predictions, true_labels


def get_model_size(model_dir: str) -> float:
    """
    Calculate model size in MB.
    
    Args:
        model_dir: Directory containing saved model
        
    Returns:
        Model size in MB
    """
    from pathlib import Path
    
    model_path = Path(model_dir)
    total_size = sum(f.stat().st_size for f in model_path.glob('**/*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    return size_mb

"""Model training with HuggingFace Transformers."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset
import pandas as pd


def prepare_dataset(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    text_col: str = 'text',
    label_col: str = 'label',
    max_length: int = 256
) -> Dataset:
    """
    Tokenize and prepare dataset for training.
    
    Args:
        df: Input DataFrame
        tokenizer: HuggingFace tokenizer
        text_col: Name of text column
        label_col: Name of label column
        max_length: Maximum sequence length
        
    Returns:
        HuggingFace Dataset object
    """
    def tokenize_function(examples):
        return tokenizer(
            examples[text_col],
            padding=False,
            truncation=True,
            max_length=max_length
        )
    
    dataset = Dataset.from_pandas(df[[text_col, label_col]])
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column(label_col, "labels")
    
    return tokenized_dataset


def train_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str,
    config: Dict[str, Any],
    random_seed: int = 42,
    allow_resume: bool = True,
    text_col: str = 'text',
    label_col: str = 'label'
) -> None:
    """
    Fine-tune transformer model for phishing detection.
    
    Args:
        model_name: HuggingFace model identifier
        train_df: Training DataFrame
        val_df: Validation DataFrame
        output_dir: Directory to save model checkpoints
        config: Training configuration
        random_seed: Random seed for reproducibility
        allow_resume: If True, resume from latest checkpoint in output_dir
        text_col: Name of text column
        label_col: Name of label column
    """
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"\nTraining {model_name}")
    print("=" * 50)
    
    # Extract training config first
    training_config = config.get('training', {})
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        use_safetensors=True  # Force safetensors to avoid torch.load vulnerability
    )
    
    # Prepare datasets
    max_length = config.get('max_length', 256)
    train_dataset = prepare_dataset(train_df, tokenizer, text_col=text_col, label_col=label_col, max_length=max_length)
    val_dataset = prepare_dataset(val_df, tokenizer, text_col=text_col, label_col=label_col, max_length=max_length)
    
    # Training arguments with performance optimizations
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get('epochs', 3),
        per_device_train_batch_size=training_config.get('batch_size', 16),
        per_device_eval_batch_size=training_config.get('batch_size', 16),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        learning_rate=training_config.get('learning_rate', 2e-5),
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_steps=training_config.get('warmup_steps', 500),
        logging_steps=training_config.get('logging_steps', 100),
        eval_strategy=training_config.get('eval_strategy', 'epoch'),
        save_strategy=training_config.get('save_strategy', 'epoch'),
        save_steps=training_config.get('save_steps', 500),
        save_total_limit=training_config.get('save_total_limit', 2),
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        seed=random_seed,
        push_to_hub=False,
        # Performance optimizations
        fp16=training_config.get('fp16', False),
        bf16=training_config.get('bf16', False),

        dataloader_num_workers=training_config.get('dataloader_num_workers', 0),
        dataloader_pin_memory=training_config.get('dataloader_pin_memory', True),
        remove_unused_columns=True,
        report_to='none',
        disable_tqdm=False
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Train
    print("\nStarting training...")
    # Attempt to resume from the latest checkpoint if present
    checkpoint_path = None
    out_path = Path(output_dir)
    if allow_resume:
        if out_path.exists():
            last_ckpt = get_last_checkpoint(str(out_path))
            if last_ckpt is not None:
                checkpoint_path = last_ckpt
                print(f"Found checkpoint: {last_ckpt}. Resuming training.")
    else:
        print("Resume disabled: starting training without resuming.")
    trainer.train(resume_from_checkpoint=checkpoint_path)
    
    # Save final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training complete for {model_name}")

#!/usr/bin/env python3
"""
Fine-tune a Hugging Face model to predict CNPJ from UF and razaosocial columns.
This script uses a text-to-text approach where we format the input as text and predict CNPJ.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import logging
import os
from typing import Dict, List, Tuple
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNPJDataset(Dataset):
    """Custom dataset for CNPJ prediction task."""
    
    def __init__(self, texts: List[str], targets: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        targets = self.tokenizer(
            target,
            max_length=64,  # CNPJ is shorter
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten()
        }

def load_and_preprocess_data(file_path: str, sample_size: int = None) -> Tuple[List[str], List[str]]:
    """Load and preprocess the CSV data."""
    logger.info(f"Loading data from {file_path}")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Remove rows with missing values in required columns
    df = df.dropna(subset=['uf', 'razaosocial', 'cnpj'])
    
    # Sample data if specified (useful for testing)
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} rows for training")
    
    # Create input text by combining UF and razaosocial
    texts = []
    targets = []
    
    for _, row in df.iterrows():
        # Format input as: "Predict CNPJ for company: [razaosocial] in state: [uf]"
        text = f"Predict CNPJ for company: {row['razaosocial']} in state: {row['uf']}"
        target = str(row['cnpj'])
        
        texts.append(text)
        targets.append(target)
    
    logger.info(f"Prepared {len(texts)} training examples")
    return texts, targets

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Calculate exact match accuracy
    exact_matches = sum(1 for pred, label in zip(decoded_preds, decoded_labels) if pred.strip() == label.strip())
    accuracy = exact_matches / len(decoded_preds)
    
    return {"accuracy": accuracy}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune model for CNPJ prediction")
    parser.add_argument("--data_file", default="data.csv", help="Path to CSV data file")
    parser.add_argument("--model_name", default="google/flan-t5-small", help="Hugging Face model name")
    parser.add_argument("--output_dir", default="./cnpj_model", help="Output directory for fine-tuned model")
    parser.add_argument("--sample_size", type=int, default=10000, help="Sample size for training (None for full dataset)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum input sequence length")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load and preprocess data
    texts, targets = load_and_preprocess_data(args.data_file, args.sample_size)
    
    # Split data
    train_texts, val_texts, train_targets, val_targets = train_test_split(
        texts, targets, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set size: {len(train_texts)}")
    logger.info(f"Validation set size: {len(val_texts)}")
    
    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name}")
    global tokenizer  # Make tokenizer global for compute_metrics
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = CNPJDataset(train_texts, train_targets, tokenizer, args.max_length)
    val_dataset = CNPJDataset(val_texts, val_targets, tokenizer, args.max_length)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Evaluate on validation set
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Test predictions on a few examples
    logger.info("Testing predictions on sample data...")
    model.eval()
    
    # Take first 5 validation examples for testing
    test_examples = val_texts[:5]
    test_targets = val_targets[:5]
    
    for i, (text, target) in enumerate(zip(test_examples, test_targets)):
        inputs = tokenizer(text, return_tensors="pt", max_length=args.max_length, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Example {i+1}:")
        logger.info(f"  Input: {text}")
        logger.info(f"  Target: {target}")
        logger.info(f"  Prediction: {prediction}")
        logger.info(f"  Match: {prediction.strip() == target.strip()}")
        logger.info("-" * 50)

if __name__ == "__main__":
    main()

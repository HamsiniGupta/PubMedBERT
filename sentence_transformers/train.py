#!/usr/bin/env python3
"""
PubMedQA SUPERVISED SimCSE Training Script
Modified for supervised contrastive learning with validation monitoring
"""

from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
import logging
from datetime import datetime
import os
import csv
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def load_supervised_data(csv_file, validation_split=0.1):
    """
    Load SUPERVISED PubMedQA data with labels and create train/validation splits
    
    Args:
        csv_file: Path to supervised CSV file (sent0, sent1, label)
        validation_split: Fraction of data to use for validation
    
    Returns:
        train_samples, dev_samples: Training and validation samples
    """
    logging.info(f"Loading SUPERVISED data from {csv_file}")
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    required_cols = ['sent0', 'sent1', 'label']
    if not all(col in df.columns for col in required_cols):
        df.columns = ['sent0', 'sent1', 'label']
    
    # Clean the data
    df = df.dropna()
    df = df[df['sent0'].str.len() > 10]  
    df = df[df['sent1'].str.len() > 10]
    
    logging.info(f"Loaded {len(df)} samples")
    logging.info(f"Positive samples: {sum(df['label'] == 1)}")
    logging.info(f"Negative samples: {sum(df['label'] == 0)}")
    
    # Split into train and validation
    train_df, val_df = train_test_split(
        df, 
        test_size=validation_split, 
        random_state=42,
        stratify=df['label']  # Maintain label balance
    )
    
    logging.info(f"Train samples: {len(train_df)}")
    logging.info(f"Validation samples: {len(val_df)}")
    
    # Create training samples for contrastive learning
    train_samples = []
    for _, row in train_df.iterrows():
        if row['label'] == 1:  # Only positive pairs for contrastive training
            train_samples.append(InputExample(texts=[row['sent0'], row['sent1']]))
    
    # Create validation samples for evaluation
    val_samples = []
    val_labels = []
    for _, row in val_df.iterrows():
        val_samples.append(InputExample(texts=[row['sent0'], row['sent1']]))
        val_labels.append(int(row['label']))
    
    logging.info(f"Created {len(train_samples)} training pairs (positive only)")
    logging.info(f"Created {len(val_samples)} validation pairs (positive + negative)")
    
    return train_samples, val_samples, val_labels

def create_model(model_name, max_seq_length=256):
    """
    Create the sentence transformer model
    
    Args:
        model_name: HuggingFace model name
        max_seq_length: Maximum sequence length
    
    Returns:
        SentenceTransformer model
    """
    logging.info(f"Creating model with {model_name}")
    
    # Use Huggingface/transformers model for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    
    # Create the sentence transformer model
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    return model

def create_evaluator(val_samples, val_labels, name="validation"):
    """
    Create evaluator for validation monitoring
    
    Args:
        val_samples: Validation InputExample samples
        val_labels: Validation labels (0/1)
        name: Name for the evaluator
    
    Returns:
        BinaryClassificationEvaluator
    """
    logging.info(f"Creating {name} evaluator with {len(val_samples)} samples")
    
    # Extract sentences and labels for evaluator
    sentences1 = [sample.texts[0] for sample in val_samples]
    sentences2 = [sample.texts[1] for sample in val_samples]
    
    evaluator = BinaryClassificationEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        labels=val_labels,
        name=name,
        batch_size=16,
        show_progress_bar=True,
        write_csv=True
    )
    
    return evaluator

class ValidationLossTracker:
    """Custom callback to track validation loss during training"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.validation_scores = []
        self.best_score = -1
        self.best_epoch = 0
        
    def __call__(self, score, epoch, steps):
        """Called after each evaluation"""
        self.validation_scores.append({
            'epoch': epoch,
            'steps': steps,
            'score': score
        })
        
        logging.info(f"Validation score at epoch {epoch}: {score:.4f}")
        
        # Track best score
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            logging.info(f"New best validation score: {score:.4f} at epoch {epoch}")
        
        # Save validation history
        val_df = pd.DataFrame(self.validation_scores)
        val_df.to_csv(os.path.join(self.output_dir, 'validation_history.csv'), index=False)
        
        # Don't return anything - this fixes the callback issue

def train_model(model, train_samples, val_samples, val_labels, args):
    """
    Train the SUPERVISED SimCSE model with validation monitoring
    
    Args:
        model: SentenceTransformer model
        train_samples: Training data (positive pairs only)
        val_samples: Validation data (positive + negative)
        val_labels: Validation labels
        args: Training arguments
    """
    # Create validation evaluator
    dev_evaluator = create_evaluator(val_samples, val_labels, "validation")
    
    # Create validation loss tracker
    loss_tracker = ValidationLossTracker(args.output_dir)
    
    # Configure training
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)
    
    # Use MultipleNegativesRankingLoss for contrastive learning
    # This automatically creates negatives from other samples in the batch
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Calculate warmup steps
    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * 0.1)  # 10% of train data
    logging.info(f"Warmup steps: {warmup_steps}")
    
    logging.info("Starting SUPERVISED training with validation monitoring")
    logging.info(f"Training on {len(train_samples)} positive pairs")
    logging.info(f"Validating on {len(val_samples)} pairs ({sum(val_labels)} positive)")
    
    # Train the model with validation
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=args.epochs,
        evaluation_steps=args.eval_steps,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        save_best_model=True,  # Save best model based on validation
        show_progress_bar=True
        # Remove callback for now to avoid the error
    )
    
    logging.info(f"Training completed!")
    logging.info(f"Best validation score: {loss_tracker.best_score:.4f} at epoch {loss_tracker.best_epoch}")
    logging.info(f"Model saved to {args.output_dir}")
    
    return loss_tracker

def evaluate_final_model(model_path, val_samples, val_labels):
    """
    Evaluate the final trained model
    
    Args:
        model_path: Path to trained model
        val_samples: Validation samples
        val_labels: Validation labels
    """
    logging.info("Evaluating final model...")
    
    model = SentenceTransformer(model_path)
    
    # Create final evaluator
    final_evaluator = create_evaluator(val_samples, val_labels, "final_evaluation")
    
    # Run evaluation
    final_score = final_evaluator(model, output_path=model_path)
    
    logging.info(f"Final model performance: {final_score:.4f}")
    
    # Test with example pairs
    if len(val_samples) > 0:
        # Positive example
        pos_idx = next(i for i, label in enumerate(val_labels) if label == 1)
        pos_texts = val_samples[pos_idx].texts
        
        # Negative example  
        neg_idx = next(i for i, label in enumerate(val_labels) if label == 0)
        neg_texts = val_samples[neg_idx].texts
        
        # Calculate similarities
        pos_embeddings = model.encode(pos_texts)
        neg_embeddings = model.encode(neg_texts)
        
        pos_similarity = util.pytorch_cos_sim(pos_embeddings[0], pos_embeddings[1]).item()
        neg_similarity = util.pytorch_cos_sim(neg_embeddings[0], neg_embeddings[1]).item()
        
        logging.info(f"\nExample results:")
        logging.info(f"Positive pair similarity: {pos_similarity:.4f}")
        logging.info(f"Negative pair similarity: {neg_similarity:.4f}")
        logging.info(f"Difference: {pos_similarity - neg_similarity:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train SUPERVISED SimCSE on PubMedQA data')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, 
                       default='google-bert/bert-base-uncased',
                       help='Pre-trained model name')
    parser.add_argument('--max_seq_length', type=int, default=256,
                       help='Maximum sequence length')
    
    # Data arguments
    parser.add_argument('--train_file', type=str, 
                       default='data/pubmedqa_train_supervised.csv',
                       help='Path to SUPERVISED PubMedQA CSV file')
    parser.add_argument('--validation_split', type=float, default=0.1,
                       help='Fraction of data to use for validation')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluation steps during training')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, 
                       default=f'output/pubmedqa-supervised-simcse',
                       help='Output directory for saved model')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info("Starting PubMedQA SUPERVISED SimCSE Training")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Data: {args.train_file}")
    logging.info(f"Output: {args.output_dir}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Validation split: {args.validation_split}")
    
    # Load supervised data
    train_samples, val_samples, val_labels = load_supervised_data(
        args.train_file, 
        args.validation_split
    )
    
    if not train_samples:
        logging.error("No training samples found! Check your CSV file format.")
        return
    
    # Create model
    model = create_model(args.model_name, args.max_seq_length)
    
    # Train model with validation
    loss_tracker = train_model(model, train_samples, val_samples, val_labels, args)
    
    # Evaluate final model
    evaluate_final_model(args.output_dir, val_samples, val_labels)
    
    logging.info("\nTraining completed successfully!")
    logging.info(f"Best validation score: {loss_tracker.best_score:.4f}")
    logging.info(f"Validation history saved to: {args.output_dir}/validation_history.csv")

if __name__ == "__main__":
    main()
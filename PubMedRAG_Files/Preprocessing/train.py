#!/usr/bin/env python3
"""
PubMedQA SimCSE Training Script
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

def load_supervised_data(train_file, val_file):
    """Load the files created from preprocess.py"""

    logging.info(f"Loading supervised data...")
    logging.info(f"Train file: {train_file}")
    logging.info(f"Validation file: {val_file}")
    
    train_df = pd.read_csv(train_file)
    if 'sent0' not in train_df.columns:
        train_df.columns = ['sent0', 'sent1', 'label']
    
    val_df = pd.read_csv(val_file)
    if 'sent1' not in val_df.columns:
        val_df.columns = ['sent0', 'sent1', 'label']
    else:
        # preprocess.py uses sent1/sent2 for val and test files
        val_df = val_df.rename(columns={'sent1': 'sent0', 'sent2': 'sent1'})

    # Clean data
    train_df = train_df.dropna()
    val_df = val_df.dropna()

    logging.info(f"Loaded {len(train_df)} training samples")
    logging.info(f"Loaded {len(val_df)} validation samples")
    logging.info(f"Training - Positive: {sum(train_df['label'] == 1)}, Negative: {sum(train_df['label'] == 0)}")
    logging.info(f"Validation - Positive: {sum(val_df['label'] == 1)}, Negative: {sum(val_df['label'] == 0)}")

   # Create training samples with labels for ContrastiveLoss
    train_samples = []
    for _, row in train_df.iterrows():
        train_samples.append(InputExample(
            texts=[row['sent0'], row['sent1']], 
            label=float(row['label'])
        ))
    
    # Create validation samples
    val_samples = []
    val_labels = []
    for _, row in val_df.iterrows():
        val_samples.append(InputExample(texts=[row['sent0'], row['sent1']]))
        val_labels.append(int(row['label']))
    
    logging.info(f"Created {len(train_samples)} training pairs with labels")
    logging.info(f"Created {len(val_samples)} validation pairs")
    
    return train_samples, val_samples, val_labels


def create_model(model_name, max_seq_length=256):
    """
    Create the sentence transformer mode
    Args:
        model_name: BERT
        max_seq_length: Maximum sequence length
    """
    logging.info(f"Creating model with {model_name}")
    
    # Use model for mapping tokens to embeddings
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
    Train the supervised SimCSE model with validation monitoring
    
    Args:
        model: SentenceTransformer model
        train_samples: Training data 
        val_samples: Validation data 
        val_labels: Validation labels
        args: Training arguments
    """
    # Create validation evaluator
    dev_evaluator = create_evaluator(val_samples, val_labels, "validation")
    
    # Create validation loss tracker
    loss_tracker = ValidationLossTracker(args.output_dir)
    
    # Configure training
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)
    
    # Use ContrastiveLoss for contrastive learning
    train_loss = losses.ContrastiveLoss(model)
    
    # Calculate warmup steps
    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * 0.1)  # 10% of train data
    logging.info(f"Warmup steps: {warmup_steps}")
    
    logging.info("Starting supervised contrastive training with validation monitoring")
    logging.info(f"Training on {len(train_samples)} pairs")
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
        show_progress_bar=True,
        callback=loss_tracker
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
    
    # Run evaluation - returns a dict of metrics
    final_results = final_evaluator(model, output_path=model_path)
    
    # Log the results properly
    logging.info(f"Final model results: {final_results}")
    
    # Extract specific metrics if available
    if isinstance(final_results, dict):
        if 'accuracy' in final_results:
            logging.info(f"Final accuracy: {final_results['accuracy']:.4f}")
        if 'f1' in final_results:
            logging.info(f"Final F1: {final_results['f1']:.4f}")
    else:
        logging.info(f"Final model performance: {final_results:.4f}")
    
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
    parser = argparse.ArgumentParser(description='Train supervised contrastive model on PubMedQA data')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, 
                       default='google-bert/bert-base-uncased',
                       help='Pre-trained model name')
    parser.add_argument('--max_seq_length', type=int, default=256,
                       help='Maximum sequence length')
    
    # Data arguments
    parser.add_argument('--train_file', type=str, 
                       default='../data/pubmedqa_train_supervised.csv',
                       help='Path to pre-split training CSV file')
    parser.add_argument('--val_file', type=str, default='../data/pubmedqa_val_clean.csv',
                       help='Path to pre-split validation CSV file')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluation steps during training')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, 
                       default=f'../output/pubmedqa-supervised-simcse',
                       help='Output directory for saved model')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info("Starting PubMedQA supervised SimCSE Training")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Train file: {args.train_file}")
    logging.info(f"Validation file: {args.val_file}")
    logging.info(f"Output: {args.output_dir}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Epochs: {args.epochs}")
    
    # Load supervised data
    train_samples, val_samples, val_labels = load_supervised_data(
        args.train_file,
        args.val_file, 
    )
    
    if not train_samples:
        logging.error("No training samples found! Check CSV file.")
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
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
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
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def extract_embeddings(model, tokenizer, texts, device):
    """Extract [CLS] embeddings for both BERT and SentenceTransformer"""
    if isinstance(model, SentenceTransformer):
        # Use SentenceTransformer's built-in encode method
        return model.encode(texts, convert_to_numpy=True, device=device)
    
    # Otherwise, assume it's a raw HuggingFace model like AutoModel
    model.to(device)
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings.append(cls_embedding.cpu().numpy())
    
    return np.vstack(embeddings)

def test_dataset_questions(dataset, baseline_model, trained_model, tokenizer, device):
    """Test each question against its third context (if available)"""
    
    results = []
    
    for pub_id, data in dataset.items():
        question = data["QUESTION"]
        contexts = data["CONTEXTS"]
        
        # Check if third context exists (index 2)
        if len(contexts) >= 3:
            third_context = contexts[1]
            
            # Create test pair
            test_texts = [question, third_context]
            
            # Extract embeddings
            baseline_embs = extract_embeddings(baseline_model, tokenizer, test_texts, device)
            trained_embs = extract_embeddings(trained_model, tokenizer, test_texts, device)
            
            # Calculate similarities
            baseline_sim = cosine_similarity(baseline_embs[0:1], baseline_embs[1:2])[0, 0]
            trained_sim = cosine_similarity(trained_embs[0:1], trained_embs[1:2])[0, 0]
            improvement = trained_sim - baseline_sim
            
            # Store results
            result = {
                'pub_id': pub_id,
                'question': question,
                'third_context': third_context,
                'baseline_sim': baseline_sim,
                'trained_sim': trained_sim,
                'improvement': improvement
            }
            results.append(result)
            
            # Print results for this question
            print(f"\nPub ID: {pub_id}")
            print(f"Question: {question}")
            print(f"Third Context: {third_context}")
            print(f"Baseline similarity: {baseline_sim:.3f}")
            print(f"Trained similarity: {trained_sim:.3f}")
            print(f"Improvement: {improvement:.3f}")
            print("-" * 80)
        else:
            print(f"\nPub ID: {pub_id} - Skipped (less than 3 contexts available)")
    
    return results

# Load your dataset (assuming it's in JSON format)
# Replace 'your_dataset.json' with the actual path to your dataset file
with open('ori_pqal.json', 'r') as f:
    dataset = json.load(f)

# Load baseline BERT
baseline_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

# Load your trained model  
trained_model = SentenceTransformer("output/pubmedqa-supervised-simcse")

# Test all questions in the dataset
results = test_dataset_questions(dataset, baseline_model, trained_model, tokenizer, device)

# Print summary statistics
if results:
    improvements = [r['improvement'] for r in results]
    baseline_sims = [r['baseline_sim'] for r in results]
    trained_sims = [r['trained_sim'] for r in results]
    
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total questions tested: {len(results)}")
    print(f"Average baseline similarity: {np.mean(baseline_sims):.3f}")
    print(f"Average trained similarity: {np.mean(trained_sims):.3f}")
    print(f"Average improvement: {np.mean(improvements):.3f}")
    print(f"Std improvement: {np.std(improvements):.3f}")
    print(f"Best improvement: {np.max(improvements):.3f}")
    print(f"Worst improvement: {np.min(improvements):.3f}")
    
    # Show best and worst performing questions
    best_idx = np.argmax(improvements)
    worst_idx = np.argmin(improvements)
    
    print(f"\nBest performing question (Pub ID: {results[best_idx]['pub_id']}):")
    print(f"Improvement: {results[best_idx]['improvement']:.3f}")
    
    print(f"\nWorst performing question (Pub ID: {results[worst_idx]['pub_id']}):")
    print(f"Improvement: {results[worst_idx]['improvement']:.3f}")

# Optional: Save results to CSV for further analysis
df_results = pd.DataFrame(results)
df_results.to_csv('similarity_test_results.csv', index=False)
print(f"\nResults saved to 'similarity_test_results.csv'")
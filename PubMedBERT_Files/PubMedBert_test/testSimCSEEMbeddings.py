#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from SimCSEEmbeddings import SimCSEEmbeddings

def evaluate_simcse_embeddings(model_path, eval_file="data/pubmedqa_test_clean.csv"):

    print("Loading PubMedBERT model...")
    embeddings_model = SimCSEEmbeddings(model_path)
    
    print("Loading evaluation data...")
    df = pd.read_csv(eval_file)
    print(f"Loaded {len(df)} evaluation pairs")
    
    similarities = []
    labels = []
    
    print("Computing embeddings and similarities...")
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"Processing pair {idx+1}/{len(df)}")
            
        # Get embeddings for both sentences
        emb1 = embeddings_model.embed_query(row['sent1'])
        emb2 = embeddings_model.embed_query(row['sent2'])
        
        # Convert to numpy arrays
        emb1 = np.array(emb1).reshape(1, -1)
        emb2 = np.array(emb2).reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(emb1, emb2)[0][0]
        
        similarities.append(similarity)
        labels.append(row['label'])
    
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    # Calculate metrics
    print("\n" + "="*50)
    print("PubMedBERT Embedding Evaluation Results")
    print("="*50)
    
    # Basic statistics
    pos_similarities = similarities[labels == 1]
    neg_similarities = similarities[labels == 0]
    
    print(f"\nSimilarity Statistics:")
    print(f"Relevant pairs (label=1):   mean={pos_similarities.mean():.4f}, std={pos_similarities.std():.4f}")
    print(f"Irrelevant pairs (label=0): mean={neg_similarities.mean():.4f}, std={neg_similarities.std():.4f}")
    print(f"Difference in means: {pos_similarities.mean() - neg_similarities.mean():.4f}")
    
    # ROC AUC 
    auc_score = roc_auc_score(labels, similarities)
    print(f"\nROC AUC Score: {auc_score:.4f}")
    print("(1.0 = perfect, 0.5 = random)")
    
    # Try different thresholds for binary classification
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8]
    best_f1 = 0
    best_threshold = 0
    
    print(f"\nThreshold Analysis:")
    print("Threshold | Accuracy | Precision | Recall | F1-Score")
    print("-" * 55)
    
    for threshold in thresholds:
        predictions = (similarities > threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        
        print(f"{threshold:8.1f} | {accuracy:8.4f} | {precision:9.4f} | {recall:6.4f} | {f1:8.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold} (F1={best_f1:.4f})")
    
    # Correlation analysis
    correlation = np.corrcoef(similarities, labels)[0, 1]
    print(f"Pearson correlation: {correlation:.4f}")
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Similarity distributions
    plt.subplot(1, 2, 1)
    plt.hist(neg_similarities, bins=30, alpha=0.7, label='Irrelevant (0)', color='red')
    plt.hist(pos_similarities, bins=30, alpha=0.7, label='Relevant (1)', color='green')
    plt.xlabel('Cosine Similarity', fontsize = 18)
    plt.ylabel('Frequency', fontsize = 18)
    plt.title('PubMedBERT Evaluation', fontsize = 18)
    plt.legend(fontsize=13)  
    
    # Plot 2: Scatter plot
    plt.subplot(1, 2, 2)
    jittered_labels = labels + np.random.normal(0, 0.05, len(labels))
    plt.scatter(similarities, jittered_labels, alpha=0.6)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Label')
    plt.title('Similarity vs Label')
    plt.ylim(-0.5, 1.5)
    
    plt.tight_layout()
    plt.savefig('../plots/PubMedBERT_Embeddings_Visuals.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary assessment
    print(f"\n" + "="*50)
    print("Summary Assessment")
    print("="*50)
    
    if auc_score > 0.8:
        print("EXCELLENT: shows strong semantic understanding")
    elif auc_score > 0.7:
        print("GOOD: captures semantic similarity well")
    elif auc_score > 0.6:
        print("FAIR: some semantic understanding")
    else:
        print("POOR: may need more training or data")
    
    print(f"Key metrics:")
    print(f"ROC AUC: {auc_score:.4f}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Relevant vs Irrelevant gap: {pos_similarities.mean() - neg_similarities.mean():.4f}")
    
    return {
        'similarities': similarities,
        'labels': labels,
        'auc_score': auc_score,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'pos_mean': pos_similarities.mean(),
        'neg_mean': neg_similarities.mean(),
        'correlation': correlation
    }
import json
def save_results(results, filename):
    """Save evaluation results to JSON file"""
    results_copy = {}
    
    for key, value in results.items():
        if hasattr(value, 'tolist'):  # numpy arrays
            results_copy[key] = value.tolist()
        elif hasattr(value, 'item'):  # numpy scalars
            results_copy[key] = value.item()
        else:  # regular Python types
            results_copy[key] = value
    
    with open(filename, 'w') as f:
        json.dump(results_copy, f, indent=2)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    
    import os

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "output", "pubmedqa-supervised-simcse"))

    EVAL_DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data", "pubmedqa_test_clean.csv"))

    results = evaluate_simcse_embeddings(MODEL_PATH, EVAL_DATA_PATH)
    save_results(results, '../data/pubmedbert_embedding_results_file.json')

    print("\nCheck PubMedBERT_Embeddings_Visuals.png for visualizations.")

#!/usr/bin/env python3
"""
Test trained SimCSE model on PubMedQA test dataset
"""

import csv
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def load_test_data(test_file):
    print(f"Loading test data from {test_file}")
    
    sentences1 = []
    sentences2 = []
    labels = []
    
    with open(test_file, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentences1.append(row['sent1'].strip())
            sentences2.append(row['sent2'].strip())
            labels.append(int(row['label']))
    
    print(f"Loaded {len(sentences1)} test pairs")
    print(f"Positive pairs (label=1): {sum(labels)}")
    print(f"Negative pairs (label=0): {len(labels) - sum(labels)}")
    
    return sentences1, sentences2, labels

def compute_similarity_scores(model, sentences1, sentences2):
    print("Computing embeddings...")
    
    # Encode all sentences
    embeddings1 = model.encode(sentences1, show_progress_bar=True)
    embeddings2 = model.encode(sentences2, show_progress_bar=True)
    
    print("Computing similarity scores...")
    
    # Compute cosine similarities
    similarities = []
    for emb1, emb2 in zip(embeddings1, embeddings2):
        sim = util.pytorch_cos_sim(emb1, emb2).item()
        similarities.append(sim)
    
    return similarities

def evaluate_model(similarities, labels, threshold=0.5):
    print(f"\nModel Evaluation Results")
    
    # Convert similarities to binary predictions using threshold
    predictions = [1 if sim >= threshold else 0 for sim in similarities]
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    try:
        auc = roc_auc_score(labels, similarities)
    except ValueError:
        auc = "N/A (need both classes)"
    
    print(f"Threshold: {threshold:.3f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc}")
    
    # Correlation between similarities and labels
    correlation, p_value = pearsonr(similarities, labels)
    print(f"Pearson Correlation: {correlation:.4f} (p={p_value:.4f})")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'correlation': correlation
    }

def analyze_results(sentences1, sentences2, similarities, labels):
    print(f"\nExample Analysis")
    
    # Sort by similarity scores
    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    
    print("\nHighest Similarity Pairs:")
    for i in range(min(3, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"Similarity: {similarities[idx]:.4f} | Label: {labels[idx]}")
        print(f"Q: {sentences1[idx][:100]}...")
        print(f"A: {sentences2[idx][:100]}...")
        print("-" * 50)
    
    print("\nLowest Similarity Pairs:")
    for i in range(max(0, len(sorted_indices)-3), len(sorted_indices)):
        idx = sorted_indices[i]
        print(f"Similarity: {similarities[idx]:.4f} | Label: {labels[idx]}")
        print(f"Q: {sentences1[idx][:100]}...")
        print(f"A: {sentences2[idx][:100]}...")
        print("-" * 50)
    
    print(f"\nPotential Issues:")
    
    # High similarity but wrong label
    wrong_high = [(i, similarities[i]) for i, (sim, label) in enumerate(zip(similarities, labels)) 
                  if sim > 0.7 and label == 0]
    if wrong_high:
        print(f"High similarity but label=0: {len(wrong_high)} cases")
        for i, sim in wrong_high[:2]:
            print(f"  Sim: {sim:.4f} - Q: {sentences1[i][:80]}...")
    
    # Low similarity but correct label  
    wrong_low = [(i, similarities[i]) for i, (sim, label) in enumerate(zip(similarities, labels)) 
                 if sim < 0.3 and label == 1]
    if wrong_low:
        print(f"Low similarity but label=1: {len(wrong_low)} cases")
        for i, sim in wrong_low[:2]:
            print(f"  Sim: {sim:.4f} - Q: {sentences1[i][:80]}...")

def plot_results(similarities, labels, save_path=None):
    plt.figure(figsize=(12, 4))
    
    # Subplot 1: Histogram
    plt.subplot(1, 2, 1)
    pos_sims = [sim for sim, label in zip(similarities, labels) if label == 1]
    neg_sims = [sim for sim, label in zip(similarities, labels) if label == 0]
    
    plt.hist(pos_sims, alpha=0.7, label='Related (label=1)', bins=20, color='green')
    plt.hist(neg_sims, alpha=0.7, label='Unrelated (label=0)', bins=20, color='red')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Similarity Score Distribution')
    plt.legend()
    
    # Subplot 2: Box plot
    plt.subplot(1, 2, 2)
    data_to_plot = [pos_sims, neg_sims]
    plt.boxplot(data_to_plot, labels=['Related\n(label=1)', 'Unrelated\n(label=0)'])
    plt.ylabel('Cosine Similarity')
    plt.title('Similarity by Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def find_best_threshold(similarities, labels):
    print(f"\n=== Finding Best Threshold ===")
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    results = []
    for threshold in thresholds:
        predictions = [1 if sim >= threshold else 0 for sim in similarities]
        _, _, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        results.append((threshold, f1))
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
    
    # Show top 5 thresholds
    results.sort(key=lambda x: x[1], reverse=True)
    print("Top 5 thresholds:")
    for i, (thresh, f1) in enumerate(results[:5]):
        print(f"  {i+1}. Threshold: {thresh:.3f}, F1: {f1:.4f}")
    
    return best_threshold

def main():
    MODEL_PATH = "output/pubmedqa-supervised-simcse"  
    TEST_FILE = "data/pubmedqa_val_clean.csv"
    THRESHOLD = 0.5
    SAVE_PLOT = "plots/evaluation_results.png"
    
    print("Testing SimCSE Model on PubMedQA")
    print(f"Model: {MODEL_PATH}")
    print(f"Test file: {TEST_FILE}")
    
    # Load model
    print("Loading trained model...")
    model = SentenceTransformer(MODEL_PATH)
    
    # Load test data
    sentences1, sentences2, labels = load_test_data(TEST_FILE)
    
    # Compute similarities
    similarities = compute_similarity_scores(model, sentences1, sentences2)
    
    # Find best threshold
    best_threshold = find_best_threshold(similarities, labels)
    
    # Evaluate with best threshold
    metrics = evaluate_model(similarities, labels, best_threshold)
    
    # Evaluate with provided threshold too
    if THRESHOLD != best_threshold:
        print(f"\nResults with provided threshold ({THRESHOLD})")
        evaluate_model(similarities, labels, THRESHOLD)
    
    # Analyze results
    analyze_results(sentences1, sentences2, similarities, labels)
    
    # Plot results
    plot_results(similarities, labels, SAVE_PLOT)
    
    print(f"\nSummary:")
    print(f"Model achieved {metrics['accuracy']:.1%} accuracy")
    print(f"Correlation with labels: {metrics['correlation']:.4f}")
    
    if metrics['correlation'] > 0.5:
        print("Model learned meaningful representations!")
    elif metrics['correlation'] > 0.3:
        print("Model is working but could improve")
    else:
        print("Model may need more training")

if __name__ == "__main__":
    main()
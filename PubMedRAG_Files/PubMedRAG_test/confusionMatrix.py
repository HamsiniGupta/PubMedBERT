#!/usr/bin/env python3
"""
Evaluate PubMedRAG model on pubmedqa_test_clean.csv
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from SimCSEEmbeddings import SimCSEEmbeddings

def evaluate_simcse_embeddings(model_path, eval_file="../data/pubmedqa_test_clean.csv"):

    print("Loading PubMedRAG...")
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
    
    print("\n" + "="*50)
    print("PubMedRAG Evaluation Results")
    print("="*50)
    
    pos_similarities = similarities[labels == 1]
    neg_similarities = similarities[labels == 0]
    
    print(f"\nSimilarity Statistics:")
    print(f"Relevant pairs (label=1):   mean={pos_similarities.mean():.4f}, std={pos_similarities.std():.4f}")
    print(f"Irrelevant pairs (label=0): mean={neg_similarities.mean():.4f}, std={neg_similarities.std():.4f}")
    print(f"Difference in means: {pos_similarities.mean() - neg_similarities.mean():.4f}")
    
    # ROC-AUC 
    auc_score = roc_auc_score(labels, similarities)
    print(f"\nROC-AUC Score: {auc_score:.4f}")
    print("(1.0 = perfect, 0.5 = random)")
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
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
        
    print(f"\n" + "="*50)
    print("Summary Assessment")
    print("="*50)
    
    if auc_score > 0.8:
        print("Excellent: strong semantic understanding")
    elif auc_score > 0.7:
        print("Good: captures semantic similarity well")
    elif auc_score > 0.6:
        print("Fair: some semantic understanding")
    else:
        print("Poor: may need more training or data")
    
    print(f"Key metrics:")
    print(f"ROC-AUC: {auc_score:.4f}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Relevant vs Irrelevant gap: {pos_similarities.mean() - neg_similarities.mean():.4f}")
    return {
        'similarities': similarities,
        'labels': labels,
        'auc_score': auc_score,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'pos_mean': pos_similarities.mean(),
        'neg_mean': neg_similarities.mean()
    }

def create_horizontal_confusion_matrix(similarities, labels, threshold, save_path="../plots/confusion_matrix_horizontal.png"):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    predictions = (similarities > threshold).astype(int)
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 6))  
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',  
                xticklabels=['Irrelevant', 'Relevant'],
                yticklabels=['Irrelevant', 'Relevant'],
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 22})
    plt.xticks(fontsize=16)  
    plt.yticks(fontsize=16)  
    
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.title(f'Confusion Matrix (Threshold = {threshold})', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"confusion matrix saved to {save_path}")
    
    return cm

if __name__ == "__main__":
    
    import os

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "output", "pubmedqa-supervised-simcse"))

    EVAL_DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data", "pubmedqa_test_clean.csv"))

    results = evaluate_simcse_embeddings(MODEL_PATH, EVAL_DATA_PATH)

    cm = create_horizontal_confusion_matrix(
        results['similarities'],
        results['labels'],
        results['best_threshold'],
        save_path=os.path.join(CURRENT_DIR, "..", "plots", "confusion_matrix_horizontal.png")
    )
    

    print("\nEvaluation complete!")
    print("Check ../plots/confusion_matrix_horizontal.png for confusion matrix.")
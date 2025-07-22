#!/usr/bin/env python3
"""
Evaluate PubMedBert model on pubmedqa_eval.csv with ROC curve analysis
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from SimCSEEmbeddings import SimCSEEmbeddings

def evaluate_simcse_embeddings(model_path, eval_file="data/pubmedqa_val_clean.csv"):

    print("Loading SimCSE model...")
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
    print("SIMCSE EMBEDDING EVALUATION RESULTS")
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
    
    # Generate ROC and PR curves
    generate_roc_analysis(similarities, labels, auc_score, best_threshold)
    
    create_similarity_plots(pos_similarities, neg_similarities, similarities, labels)
    
    print(f"\n" + "="*50)
    print("SUMMARY ASSESSMENT")
    print("="*50)
    
    if auc_score > 0.8:
        print("EXCELLENT: strong semantic understanding")
    elif auc_score > 0.7:
        print("GOOD: captures semantic similarity well")
    elif auc_score > 0.6:
        print("FAIR: some semantic understanding")
    else:
        print("POOR: may need more training or data")
    
    print(f"Key metrics:")
    print(f"- ROC-AUC: {auc_score:.4f}")
    print(f"- Best F1: {best_f1:.4f}")
    print(f"- Relevant vs Irrelevant gap: {pos_similarities.mean() - neg_similarities.mean():.4f}")
    cm_horizontal = create_horizontal_confusion_matrix(similarities, labels, best_threshold)
    return {
        'similarities': similarities,
        'labels': labels,
        'auc_score': auc_score,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'pos_mean': pos_similarities.mean(),
        'neg_mean': neg_similarities.mean()
    }

def generate_roc_analysis(similarities, labels, auc_score, best_threshold):

    print("\nGenerating ROC curve analysis...")
    
    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(labels, similarities)
    
    # Calculate Precision Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(labels, similarities)
    pr_auc = auc(recall, precision)
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Medical Embedding Model - ROC Analysis', fontsize=16, fontweight='bold')
    
    # 1. ROC Curve
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=3, 
                    label=f'ROC Curve (AUC = {auc_score:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8, 
                    label='Random Classifier')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)
        
    # 2. Precision Recall Curve
    axes[0, 1].plot(recall, precision, color='purple', lw=3,
                    label=f'PR Curve (AUC = {pr_auc:.3f})')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('Recall', fontsize=12)
    axes[0, 1].set_ylabel('Precision', fontsize=12)
    axes[0, 1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc="lower left")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Threshold vs Metrics
    threshold_range = np.linspace(0.1, 0.9, 50)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in threshold_range:
        preds = (similarities > thresh).astype(int)
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
    
    axes[0, 2].plot(threshold_range, accuracies, 'b-', label='Accuracy', lw=2)
    axes[0, 2].plot(threshold_range, precisions, 'r-', label='Precision', lw=2)
    axes[0, 2].plot(threshold_range, recalls, 'g-', label='Recall', lw=2)
    axes[0, 2].plot(threshold_range, f1_scores, 'm-', label='F1-Score', lw=2)
    axes[0, 2].axvline(x=best_threshold, color='black', linestyle='--', alpha=0.8, 
                       label=f'Best Threshold ({best_threshold})')
    axes[0, 2].set_xlabel('Threshold', fontsize=12)
    axes[0, 2].set_ylabel('Score', fontsize=12)
    axes[0, 2].set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Similarity Distribution with ROC regions
    axes[1, 0].hist(similarities[labels == 0], bins=30, alpha=0.7, 
                    label='Irrelevant (0)', color='red', density=True)
    axes[1, 0].hist(similarities[labels == 1], bins=30, alpha=0.7, 
                    label='Relevant (1)', color='green', density=True)
    axes[1, 0].axvline(x=best_threshold, color='black', linestyle='--', lw=2,
                       label=f'Best Threshold ({best_threshold})')
    axes[1, 0].set_xlabel('Cosine Similarity', fontsize=12)
    axes[1, 0].set_ylabel('Density', fontsize=12)
    axes[1, 0].set_title('Similarity Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Confusion Matrix at Best Threshold
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    best_preds = (similarities > best_threshold).astype(int)
    cm = confusion_matrix(labels, best_preds)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=['Irrelevant', 'Relevant'],
                yticklabels=['Irrelevant', 'Relevant'])
    axes[1, 1].set_xlabel('Predicted', fontsize=12)
    axes[1, 1].set_ylabel('Actual', fontsize=12)
    axes[1, 1].set_title(f'Confusion Matrix (Threshold = {best_threshold})', 
                         fontsize=14, fontweight='bold')
    
    # 6. Model Performance Summary
    axes[1, 2].axis('off')
    
    # Calculate final metrics at best threshold
    best_preds = (similarities > best_threshold).astype(int)
    final_acc = accuracy_score(labels, best_preds)
    final_prec, final_rec, final_f1, _ = precision_recall_fscore_support(labels, best_preds, average='binary')
    # Check similarity distribution
    print(f"Similarities below 0.3: {sum(similarities < 0.3)}")
    print(f"Similarities 0.3-0.5: {sum((similarities >= 0.3) & (similarities < 0.5))}")
    print(f"Similarities 0.5-0.7: {sum((similarities >= 0.5) & (similarities < 0.7))}")
    print(f"Similarities above 0.7: {sum(similarities >= 0.7)}")

    # Check if model is just defaulting
    print(f"\nActual range: {similarities.min():.3f} to {similarities.max():.3f}")
    print(f"Median similarity: {np.median(similarities):.3f}")
    summary_text = f"""
    MODEL PERFORMANCE SUMMARY
    
    ROC-AUC Score: {auc_score:.4f}
    PR-AUC Score: {pr_auc:.4f}
    
    Best Threshold: {best_threshold:.3f}
    
    At Best Threshold:
    Accuracy: {final_acc:.4f}
    Precision: {final_prec:.4f}
    Recall: {final_rec:.4f}
    F1-Score: {final_f1:.4f}
    
    Interpretation:
    ROC-AUC > 0.8: {auc_score > 0.8}
    Good Precision: {final_prec > 0.7}
    High Recall: {final_rec > 0.8}
    

    """
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('plots/comprehensive_roc_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"ROC analysis saved to plots/comprehensive_roc_analysis.png")
    
    # Print detailed ROC interpretation
    print(f"\nROC Curve Interpretation:")
    print(f"ROC-AUC = {auc_score:.4f}")
    if auc_score > 0.9:
        print("EXCELLENT: outstanding discriminative ability")
    elif auc_score > 0.8:
        print("VERY GOOD: strong discriminative ability")
    elif auc_score > 0.7:
        print("GOOD: decent discriminative ability")
    elif auc_score > 0.6:
        print("FAIR: some discriminative ability")
    else:
        print("POOR: limited discriminative ability")
    
    print(f"\nPrecision Recall AUC = {pr_auc:.4f}")
def create_horizontal_confusion_matrix(similarities, labels, threshold, save_path="plots/confusion_matrix_horizontal.png"):
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
    
    print(f"Horizontal confusion matrix saved to {save_path}")
    
    return cm
def create_similarity_plots(pos_similarities, neg_similarities, similarities, labels):
   
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Similarity distributions
    plt.subplot(1, 2, 1)
    plt.hist(neg_similarities, bins=30, alpha=0.7, label='Irrelevant (0)', color='red')
    plt.hist(pos_similarities, bins=30, alpha=0.7, label='Relevant (1)', color='green')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution by Label')
    plt.legend()
    
    # Plot 2: Scatter plot
    plt.subplot(1, 2, 2)
    jittered_labels = labels + np.random.normal(0, 0.05, len(labels))
    plt.scatter(similarities, jittered_labels, alpha=0.6)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Label (jittered)')
    plt.title('Similarity vs Label')
    plt.ylim(-0.5, 1.5)
    
    plt.tight_layout()
    plt.savefig('plots/simcse_embeddings_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    
    import os

    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "output", "pubmedqa-supervised-simcse"))

    EVAL_DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data", "pubmedqa_val_clean.csv"))

    results = evaluate_simcse_embeddings(MODEL_PATH, EVAL_DATA_PATH)
    

    print("\nEvaluation complete!")
    print("Check plots/comprehensive_roc_analysis.png for detailed ROC analysis.")
    print("Check plots/simcse_embeddings_results.png for similarity distributions.")
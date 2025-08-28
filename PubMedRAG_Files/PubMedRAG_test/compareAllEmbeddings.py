from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from SimCSEEmbeddings import SimCSEEmbeddings
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_bert_embeddings(model, tokenizer, texts, device):
    """Extract [CLS] embeddings from raw BERT model"""
    model.to(device)
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', padding=True, 
                             truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings.append(cls_embedding.cpu().numpy())
    
    return np.vstack(embeddings)

def extract_simcse_embeddings(model, texts):
    """Extract embeddings from PubMedRAG model"""
    embeddings = []
    for text in texts:
        emb = model.embed_query(text)
        embeddings.append(np.array(emb))
    return np.vstack(embeddings)

def test_file_set(test_csv, baseline_model, baseline_tokenizer, trained_model):
    """Test on test set"""
    
    print(f"Loading test data from {test_csv}...")
    df = pd.read_csv(test_csv)
    
    print(f"Loaded {len(df)} test pairs")
    print(f"Positive pairs (label=1): {sum(df['label'] == 1)}")
    print(f"Negative pairs (label=0): {sum(df['label'] == 0)}")
    
    results = []
    
    for idx, row in df.iterrows():
        try:
            if idx % 50 == 0:
                print(f"Processing pair {idx+1}/{len(df)}")
            
            question = row['sent1']
            context = row['sent2']
            true_label = row['label']
            
            # Skip if texts are too short
            if len(question.strip()) < 10 or len(context.strip()) < 10:
                continue
            
            # Extract embeddings
            test_texts = [question, context]
            
            # Baseline BERT embeddings
            baseline_embs = extract_bert_embeddings(baseline_model, baseline_tokenizer, 
                                                  test_texts, device)
            
            # PubMedRAG Embeddings
            trained_embs = extract_simcse_embeddings(trained_model, test_texts)
            
            # Calculate similarities
            baseline_sim = cosine_similarity([baseline_embs[0]], [baseline_embs[1]])[0, 0]
            trained_sim = cosine_similarity([trained_embs[0]], [trained_embs[1]])[0, 0]
            improvement = trained_sim - baseline_sim
            
            # Store results
            result = {
                'idx': idx,
                'question': question,
                'context': context,
                'true_label': true_label,
                'baseline_sim': float(baseline_sim),
                'trained_sim': float(trained_sim),
                'improvement': float(improvement)
            }
            results.append(result)
                
        except Exception as e:
            print(f"Error processing pair {idx}: {str(e)}")
            continue
    
    return results

def analyze_performance_with_labels(results):
    """Analyze performance"""
    if not results:
        print("No results to analyze!")
        return
    
    # Basic statistics
    improvements = [r['improvement'] for r in results]
    baseline_sims = [r['baseline_sim'] for r in results]
    trained_sims = [r['trained_sim'] for r in results]
    true_labels = [r['true_label'] for r in results]
    
    print(f"\nAverage Scores")
    print(f"Total pairs tested: {len(results)}")
    print(f"Average baseline similarity: {np.mean(baseline_sims):.3f} ± {np.std(baseline_sims):.3f}")
    print(f"Average trained similarity: {np.mean(trained_sims):.3f} ± {np.std(trained_sims):.3f}")
    print(f"Average improvement: {np.mean(improvements):.3f} ± {np.std(improvements):.3f}")
    
    # Performance by label
    positive_pairs = [r for r in results if r['true_label'] == 1]
    negative_pairs = [r for r in results if r['true_label'] == 0]
    
    print(f"\nPerformance By Label")
    print(f"Positive pairs (relevant): {len(positive_pairs)}")
    print(f"Baseline similarity: {np.mean([r['baseline_sim'] for r in positive_pairs]):.3f}")
    print(f"Trained similarity: {np.mean([r['trained_sim'] for r in positive_pairs]):.3f}")
    print(f"Average improvement: {np.mean([r['improvement'] for r in positive_pairs]):.3f}")
    
    print(f"\nNegative pairs (irrelevant): {len(negative_pairs)}")
    print(f"Baseline similarity: {np.mean([r['baseline_sim'] for r in negative_pairs]):.3f}")
    print(f"Trained similarity: {np.mean([r['trained_sim'] for r in negative_pairs]):.3f}")
    print(f"Average improvement: {np.mean([r['improvement'] for r in negative_pairs]):.3f}")
    
    # ROC-AUC Analysis
    baseline_auc = roc_auc_score(true_labels, baseline_sims)
    trained_auc = roc_auc_score(true_labels, trained_sims)
    
    print(f"\n ROC-AUC Comparison")
    print(f"Baseline BERT AUC: {baseline_auc:.4f}")
    print(f"PubMedRAG AUC: {trained_auc:.4f}")
    print(f"AUC improvement: {trained_auc - baseline_auc:+.4f}")
    
    # Threshold analysis for both models
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8]
    
    print(f"\nThreshold Analysis")
    print("Threshold | BERT F1 | PubMedRAG F1 | Improvement")
    print("-" * 50)
    
    for threshold in thresholds:
        # Baseline predictions
        baseline_preds = (np.array(baseline_sims) > threshold).astype(int)
        baseline_f1 = precision_recall_fscore_support(true_labels, baseline_preds, average='binary')[2]
        
        # Trained predictions  
        trained_preds = (np.array(trained_sims) > threshold).astype(int)
        trained_f1 = precision_recall_fscore_support(true_labels, trained_preds, average='binary')[2]
        
        f1_improvement = trained_f1 - baseline_f1
        
        print(f"{threshold:8.1f} | {baseline_f1:7.4f} | {trained_f1:11.4f} | {f1_improvement:+.4f}")
    
    # Show best and worst examples
    best_idx = np.argmax([r['improvement'] for r in results])
    worst_idx = np.argmin([r['improvement'] for r in results])
    
    print(f"\nBest Improvement Example")
    best = results[best_idx]
    print(f"Improvement: {best['improvement']:+.3f}")
    print(f"True label: {best['true_label']}")
    print(f"Baseline sim: {best['baseline_sim']:.3f} -> Trained sim: {best['trained_sim']:.3f}")
    print(f"Question: {best['question'][:150]}...")
    print(f"Context: {best['context'][:150]}...")
    
    print(f"\nWorst Improvement Example")
    worst = results[worst_idx]
    print(f"Improvement: {worst['improvement']:+.3f}")
    print(f"True label: {worst['true_label']}")
    print(f"Baseline sim: {worst['baseline_sim']:.3f} -> Trained sim: {worst['trained_sim']:.3f}")
    print(f"Question: {worst['question'][:150]}...")
    print(f"Context: {worst['context'][:150]}...")
    
    return {
        'baseline_auc': baseline_auc,
        'trained_auc': trained_auc,
        'auc_improvement': trained_auc - baseline_auc,
        'avg_improvement': np.mean(improvements),
        'positive_improvement': np.mean([r['improvement'] for r in positive_pairs]),
        'negative_improvement': np.mean([r['improvement'] for r in negative_pairs])
    }

if __name__ == "__main__":
    try:
        # Load BERT
        print("Loading BERT...")
        baseline_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        baseline_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
        
        # Load PubMedRAG
        print("Loading PubMedRAG...")
        trained_model = SimCSEEmbeddings("../output/pubmedqa-supervised-simcse")
        
        # Test on test set
        test_file = '../data/pubmedqa_test_clean.csv'  
        print("Testing on test set...")
        results = test_file_set(test_file, baseline_model, baseline_tokenizer, trained_model)
        
        # Analyze performance
        performance_metrics = analyze_performance_with_labels(results)
        
        # Save results
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv('../data/PubMedRAG_vs_BERT_Embeddings.csv', index=False)
            print(f"\nResults saved to 'PubMedRAG_vs_BERT_Embeddings.csv'")
            
            # Save summary metrics
            summary = {
                'total_pairs': len(results),
                'baseline_auc': performance_metrics['baseline_auc'],
                'trained_auc': performance_metrics['trained_auc'],
                'auc_improvement': performance_metrics['auc_improvement'],
                'avg_improvement': performance_metrics['avg_improvement']
            }
            
            summary_df = pd.DataFrame([summary])
            summary_df.to_csv('../data/summary_embeddings.csv', index=False)
            print(f"Summary metrics saved to 'summary_embeddings.csv'")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find test file: {e}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
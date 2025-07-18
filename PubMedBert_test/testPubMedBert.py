#!/usr/bin/env python3

import pandas as pd
import time
from datasets import load_dataset
from typing import List, Dict
import json
from collections import defaultdict

def generate_test_results(pipeline, weaviate_manager, embeddings, max_samples=50, output_file="pubmedbert_test_results.csv"):
    
    print(f"Loading PubMedQA test dataset...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    

    # Check available splits first
    print(f"Available splits: {list(dataset.keys())}")

    # Use validation split if available, otherwise use train
    if "validation" in dataset:
        test_data = dataset["validation"]
        print("Using validation split for testing")
    elif "test" in dataset:
        test_data = dataset["test"] 
        print("Using test split for testing")
    else:
        # Use a subset of train data
        test_data = dataset["train"]
        print("Using subset of train split for testing")
        # Take the last 100 items to avoid overlap with training
        total_items = len(test_data)
        start_idx = max(0, total_items - 100)
        test_data = test_data.select(range(start_idx, total_items))

    print(f"Test data contains {len(test_data)} items")
    
    results = []
    
    print(f"Generating results for {max_samples} questions...")
    
    for i, item in enumerate(test_data):
        if i >= max_samples:
            break
            
        question = item['question']
        true_answer = item['final_decision']
        long_answer = item['long_answer']
        pubid = item.get('pubid', f'test_{i}')
        
        print(f"Processing question {i+1}/{max_samples}: {question[:50]}...")
        
        try:
            start_time = time.time()
            
            # Get retrieved documents (contexts)
            print(f"About to search with question type: {type(question)}")
            print(f"Question: {question[:50]}...")
            retrieved_docs = weaviate_manager.search_documents_vector_only(query=question, limit=3)
            context1 = retrieved_docs[0].page_content if len(retrieved_docs) > 0 else "No context retrieved"
            context2 = retrieved_docs[1].page_content if len(retrieved_docs) > 1 else "No second context"
            
            # Get RAG pipeline answer
            rag_response = pipeline.invoke(question)
            print(f"model response: {rag_response[:100]}")
            
            elapsed_time = time.time() - start_time
            
            # Extract scores if available
            doc1_score = retrieved_docs[0].metadata.get('score', 0) if len(retrieved_docs) > 0 else 0
            doc2_score = retrieved_docs[1].metadata.get('score', 0) if len(retrieved_docs) > 1 else 0
            
            result = {
                'question': question,
                'Context1': context1,
                'Context2': context2,
                'Answer': rag_response,
                'Ground_truth': true_answer,
                'Long_answer': long_answer,
                'pubid': pubid,
                'response_time': elapsed_time,
                'retrieval_score_1': doc1_score,
                'retrieval_score_2': doc2_score,
                'question_id': i
            }
            
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
            result = {
                'question': question,
                'Context1': "Error retrieving context",
                'Context2': "Error retrieving context", 
                'Answer': f"Error: {str(e)}",
                'Ground_truth': true_answer,
                'Long_answer': long_answer,
                'pubid': pubid,
                'response_time': 0,
                'retrieval_score_1': 0,
                'retrieval_score_2': 0,
                'question_id': i
            }
        
        results.append(result)
        
        if i % 10 == 0 and i > 0:
            print(f"Completed {i} questions...")
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Generated {len(results)} test results")
    print(f"Columns: {list(df.columns)}")
   
    
    return df

def quick_accuracy_check(df):

    def extract_decision(text):
        if not text or pd.isna(text):
            return 'unknown'
        text = str(text).lower().strip()
        print(f"model response: {text}")
        if 'yes' in text:
            return 'yes'
        elif 'no' in text:
            return 'no' 
        elif 'maybe' in text:
            return 'maybe'
        return 'unknown'
    
    df['predicted_label'] = df['Answer'].apply(extract_decision)
    df['true_label'] = df['Ground_truth'].apply(extract_decision)
    
    correct = (df['predicted_label'] == df['true_label']).sum()
    total = len(df)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nQuick Accuracy Check:")
    print(f"   Correct: {correct}/{total}")
    print(f"   Accuracy: {accuracy:.3f}")
    
    return accuracy

def calculate_precision_recall_f1(df):
    """
    Calculate precision, recall, and F1 score for each class and display in table format
    """
    
    def extract_decision(text):
        if not text or pd.isna(text):
            return 'unknown'
        text = str(text).lower().strip()
        if 'yes' in text:
            return 'yes'
        elif 'no' in text:
            return 'no' 
        elif 'maybe' in text:
            return 'maybe'
        return 'unknown'
    
    # Apply extraction if not already done
    if 'predicted_label' not in df.columns:
        df['predicted_label'] = df['Answer'].apply(extract_decision)
    if 'true_label' not in df.columns:
        df['true_label'] = df['Ground_truth'].apply(extract_decision)
    
    # Get unique labels
    all_labels = sorted(set(df['true_label'].tolist() + df['predicted_label'].tolist()))
    
    # Calculate metrics for each class
    metrics = {}
    
    for label in all_labels:
        # True Positives: predicted as label AND actually label
        tp = len(df[(df['predicted_label'] == label) & (df['true_label'] == label)])
        
        # False Positives: predicted as label BUT actually not label
        fp = len(df[(df['predicted_label'] == label) & (df['true_label'] != label)])
        
        # False Negatives: predicted as not label BUT actually label
        fn = len(df[(df['predicted_label'] != label) & (df['true_label'] == label)])
        
        # True Negatives: predicted as not label AND actually not label
        tn = len(df[(df['predicted_label'] != label) & (df['true_label'] != label)])
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    # Calculate overall accuracy
    correct = (df['predicted_label'] == df['true_label']).sum()
    total = len(df)
    accuracy = correct / total if total > 0 else 0
    
    # Display table in your requested format
    print("\n" + "="*60)
    print("PRECISION, RECALL, AND F1 SCORE TABLE")
    print("="*60)
    
    # Header
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 48)
    
    # Sort classes to match your example (MAYBE, NO, YES)
    class_order = ['maybe', 'no', 'yes']
    ordered_labels = [label for label in class_order if label in all_labels]
    # Add any remaining labels not in the predefined order
    ordered_labels.extend([label for label in all_labels if label not in class_order])
    
    for label in ordered_labels:
        if label in metrics:
            precision = metrics[label]['precision']
            recall = metrics[label]['recall']
            f1 = metrics[label]['f1']
            
            print(f"{label.upper():<10} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f}")
    
    print("-" * 48)
    print(f"{'Accuracy':<10} {'':<12} {'':<12} {accuracy:<12.3f}")
    
    # Also create a styled version similar to your image
    print("\n" + "="*60)
    print("STYLED METRICS TABLE")
    print("="*60)
    
    for label in ordered_labels:
        if label in metrics:
            precision = metrics[label]['precision']
            recall = metrics[label]['recall']
            f1 = metrics[label]['f1']
            
            print(f"{label.upper():<10} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")
    
    print(f"{'Accuracy':<10} {'':<10} {'':<10} {accuracy:>10.3f}")
    
    return metrics, accuracy

def detailed_confusion_analysis(df):
    """
    Detailed analysis of model predictions vs ground truth
    Shows confusion matrix and specific error patterns
    """
    
    def extract_decision(text):
        if not text or pd.isna(text):
            return 'unknown'
        text = str(text).lower().strip()
        if 'yes' in text:
            return 'yes'
        elif 'no' in text:
            return 'no' 
        elif 'maybe' in text:
            return 'maybe'
        return 'unknown'
    
    # Apply extraction if not already done
    if 'predicted_label' not in df.columns:
        df['predicted_label'] = df['Answer'].apply(extract_decision)
    if 'true_label' not in df.columns:
        df['true_label'] = df['Ground_truth'].apply(extract_decision)
    
    print("="*60)
    print("DETAILED CONFUSION MATRIX ANALYSIS")
    print("="*60)
    
    # Get unique labels
    all_labels = sorted(set(df['true_label'].tolist() + df['predicted_label'].tolist()))
    
    # Create confusion matrix manually
    confusion_counts = {}
    for true_label in all_labels:
        for pred_label in all_labels:
            count = len(df[(df['true_label'] == true_label) & (df['predicted_label'] == pred_label)])
            confusion_counts[(true_label, pred_label)] = count
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("                    PREDICTED")
    print("                ", end="")
    for pred_label in all_labels:
        print(f"{pred_label:>8}", end="")
    print()
    
    for true_label in all_labels:
        print(f"TRUE {true_label:>6}", end="")
        for pred_label in all_labels:
            count = confusion_counts[(true_label, pred_label)]
            print(f"{count:>8}", end="")
        print()
    
    # Detailed breakdown
    print(f"\nDetailed Error Analysis:")
    total_questions = len(df)
    
    for true_label in all_labels:
        for pred_label in all_labels:
            count = confusion_counts[(true_label, pred_label)]
            if count > 0:
                percentage = (count / total_questions) * 100
                if true_label == pred_label:
                    print(f"? TRUE '{true_label}' ? PREDICTED '{pred_label}': {count} cases ({percentage:.1f}%) - CORRECT")
                else:
                    print(f"? TRUE '{true_label}' ? PREDICTED '{pred_label}': {count} cases ({percentage:.1f}%) - ERROR")
    
    # Calculate per-class accuracy
    print(f"\nPer-Class Performance:")
    for true_label in all_labels:
        total_true = sum(confusion_counts[(true_label, pred)] for pred in all_labels)
        correct_true = confusion_counts[(true_label, true_label)]
        if total_true > 0:
            accuracy = (correct_true / total_true) * 100
            print(f"  {true_label.upper()}: {correct_true}/{total_true} correct ({accuracy:.1f}%)")
    
    # Most common errors
    print(f"\nMost Common Errors:")
    errors = [(true_label, pred_label, count) for (true_label, pred_label), count in confusion_counts.items() 
              if true_label != pred_label and count > 0]
    errors.sort(key=lambda x: x[2], reverse=True)
    
    for i, (true_label, pred_label, count) in enumerate(errors[:5]):
        percentage = (count / total_questions) * 100
        print(f"  {i+1}. Should be '{true_label}' but predicted '{pred_label}': {count} times ({percentage:.1f}%)")
    
    # Show example errors for the most common mistake
    if errors:
        most_common_error = errors[0]
        true_label, pred_label, count = most_common_error
        
        print(f"\nExample cases where TRUE='{true_label}' but PREDICTED='{pred_label}':")
        error_cases = df[(df['true_label'] == true_label) & (df['predicted_label'] == pred_label)]
        
        for i, (_, row) in enumerate(error_cases.head(3).iterrows()):
            print(f"\nExample {i+1}:")
            print(f"  Question: {row['question'][:80]}...")
            print(f"  Ground Truth: {row['Ground_truth']}")
            print(f"  Model Answer: {row['Answer'][:100]}...")
            print(f"  Context Quality Score: {row.get('retrieval_score_1', 'N/A')}")
    
    return confusion_counts

# Example usage function for your main script
def run_test_generation(pipeline, weaviate_manager, embeddings):
    
    # Generate test results
    output_file = "pubmedbert_simcse_evaluation_data.csv"
    
    print("Starting test result generation for evaluation...")
    
    df = generate_test_results(
        pipeline=pipeline,
        weaviate_manager=weaviate_manager, 
        embeddings=embeddings,
        max_samples=100,  # testing with a max of 100 samples
        output_file=output_file
    )
    
    # Quick accuracy check
    accuracy = quick_accuracy_check(df)
    
    # NEW: Calculate precision, recall, and F1 scores
    metrics, overall_accuracy = calculate_precision_recall_f1(df)
    
    # Detailed confusion analysis
    confusion_matrix = detailed_confusion_analysis(df)
    print(f"{confusion_matrix}")
    
    print(f"\nGenerated evaluation dataset:")
    print(f"   File: {output_file}")
    print(f"   Questions: {len(df)}")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Use this file in your:")
    print(f"   AnswerRelevancy.ipynb")
    print(f"   ContextAdherence.ipynb") 
    print(f"   ContextRelevancy.ipynb")
    
    return df, output_file, metrics

if __name__ == "__main__":
    print("Import this script and call run_test_generation() with pipeline objects")
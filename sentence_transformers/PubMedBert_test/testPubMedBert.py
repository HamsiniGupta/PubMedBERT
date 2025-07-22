#!/usr/bin/env python3

import pandas as pd
import time
from datasets import load_dataset
from typing import List, Dict
import json
from collections import defaultdict
start = time.time()

import json
import pandas as pd
import time

def generate_test_results_from_json(pipeline, weaviate_manager, json_file, max_samples=50, output_file="pubmedbert_test_results.csv"):
    print(f"Loading filtered PubMedQA test dataset from {json_file}...")

    # Load your filtered dataset (from a JSON file created earlier)
    with open(json_file, 'r', encoding='utf-8') as f:
        filtered_data = json.load(f)

    # Convert to list of tuples for iteration
    items = list(filtered_data.items())
    print(f"Loaded {len(items)} entries from filtered dataset")

    results = []

    for i, (pubid, entry) in enumerate(items):
        if i >= max_samples:
            break

        question = entry.get("QUESTION", "")
        true_answer = entry.get("final_decision", "")
        long_answer = entry.get("LONG_ANSWER", "")

        print(f"Processing question {i+1}/{max_samples}: {question[:50]}...")

        try:
            start_time = time.time()

            retrieved_docs = weaviate_manager.search_documents_vector_only(query=question, limit=3)
            context1 = retrieved_docs[0].page_content if len(retrieved_docs) > 0 else "No context retrieved"
            context2 = retrieved_docs[1].page_content if len(retrieved_docs) > 1 else "No second context"

            rag_response = pipeline.invoke(question)

            elapsed_time = time.time() - start_time

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
    
    if 'predicted_label' not in df.columns:
        df['predicted_label'] = df['Answer'].apply(extract_decision)
    if 'true_label' not in df.columns:
        df['true_label'] = df['Ground_truth'].apply(extract_decision)
    
    all_labels = sorted(set(df['true_label'].tolist() + df['predicted_label'].tolist()))
    
    # Calculate metrics for each class
    metrics = {}
    
    for label in all_labels:
        # True Positives
        tp = len(df[(df['predicted_label'] == label) & (df['true_label'] == label)])
        
        # False Positives
        fp = len(df[(df['predicted_label'] == label) & (df['true_label'] != label)])
        
        # False Negatives
        fn = len(df[(df['predicted_label'] != label) & (df['true_label'] == label)])
        
        # True Negatives
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
    
    print("\n" + "="*60)
    print("PRECISION, RECALL, AND F1 SCORE TABLE")
    print("="*60)
    
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 48)
    
    class_order = ['maybe', 'no', 'yes']
    ordered_labels = [label for label in class_order if label in all_labels]
    ordered_labels.extend([label for label in all_labels if label not in class_order])
    
    for label in ordered_labels:
        if label in metrics:
            precision = metrics[label]['precision']
            recall = metrics[label]['recall']
            f1 = metrics[label]['f1']
            
            print(f"{label.upper():<10} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f}")
    
    print("-" * 48)
    print(f"{'Accuracy':<10} {'':<12} {'':<12} {accuracy:<12.3f}")
    
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

    if 'predicted_label' not in df.columns:
        df['predicted_label'] = df['Answer'].apply(extract_decision)
    if 'true_label' not in df.columns:
        df['true_label'] = df['Ground_truth'].apply(extract_decision)
    
    print("="*60)
    print("DETAILED CONFUSION MATRIX ANALYSIS")
    print("="*60)
    
    all_labels = sorted(set(df['true_label'].tolist() + df['predicted_label'].tolist()))
    
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
                    print(f"TRUE '{true_label}'PREDICTED '{pred_label}': {count} cases ({percentage:.1f}%) - CORRECT")
                else:
                    print(f"TRUE '{true_label}'PREDICTED '{pred_label}': {count} cases ({percentage:.1f}%) - ERROR")
    
    print(f"\nPer Class Performance:")
    for true_label in all_labels:
        total_true = sum(confusion_counts[(true_label, pred)] for pred in all_labels)
        correct_true = confusion_counts[(true_label, true_label)]
        if total_true > 0:
            accuracy = (correct_true / total_true) * 100
            print(f"  {true_label.upper()}: {correct_true}/{total_true} correct ({accuracy:.1f}%)")
    
    print(f"\nMost Common Errors:")
    errors = [(true_label, pred_label, count) for (true_label, pred_label), count in confusion_counts.items() 
              if true_label != pred_label and count > 0]
    errors.sort(key=lambda x: x[2], reverse=True)
    
    for i, (true_label, pred_label, count) in enumerate(errors[:5]):
        percentage = (count / total_questions) * 100
        print(f"  {i+1}. Should be '{true_label}' but predicted '{pred_label}': {count} times ({percentage:.1f}%)")
    
    if errors:
        most_common_error = errors[0]
        true_label, pred_label, count = most_common_error
        
        print(f"\nExample cases where TRUE='{true_label}' but PREDICTED='{pred_label}':")
        error_cases = df[(df['true_label'] == true_label) & (df['predicted_label'] == pred_label)]
        
        for i, (_, row) in enumerate(error_cases.head(3).iterrows()):
            print(f"\nExample {i+1}:")
            print(f"Question: {row['question'][:80]}...")
            print(f"Ground Truth: {row['Ground_truth']}")
            print(f"Model Answer: {row['Answer'][:100]}...")
            print(f"Context Quality Score: {row.get('retrieval_score_1', 'N/A')}")
    
    return confusion_counts

def run_test_generation(pipeline, weaviate_manager, embeddings):
    
    # Generate test results
    output_file = "pubmedbert_simcse_evaluation_data.csv"
    
    print("Starting test result generation for evaluation...")
    
    df = generate_test_results_from_json(
        pipeline=pipeline,
        weaviate_manager=weaviate_manager,
        json_file="actual_testing_dataset.json", 
        max_samples=150,
        output_file=output_file
    )
    
    
    # Quick accuracy check
    accuracy = quick_accuracy_check(df)
    
    metrics, overall_accuracy = calculate_precision_recall_f1(df)
    
    # Detailed confusion analysis
    confusion_matrix = detailed_confusion_analysis(df)
    print(f"{confusion_matrix}")
    
    print(f"\nGenerated evaluation dataset:")
    print(f"File: {output_file}")
    print(f"Questions: {len(df)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Use this file in your:")
    print(f"AnswerRelevancy.ipynb")
    print(f"ContextAdherence.ipynb") 
    print(f"ContextRelevancy.ipynb")
    end = time.time()
    total = end - start
    print(f"{total} seconds taken for testing PubMedBERT")

    return df, output_file, metrics

if __name__ == "__main__":
    print("Import this script and call run_test_generation() with pipeline objects")
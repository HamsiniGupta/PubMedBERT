import json
import csv
import pandas as pd
from datasets import load_dataset
import os
import logging
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_dataset_by_questions(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    logger.info("Splitting dataset to ensure no question overlap...")
    
    # Convert to list and shuffle
    all_data = list(dataset)
    random.shuffle(all_data)
    
    total = len(all_data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]
    
    logger.info(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Verify no overlap
    train_questions = set(item['question'] for item in train_data)
    val_questions = set(item['question'] for item in val_data)
    test_questions = set(item['question'] for item in test_data)
    
    overlap_train_val = train_questions.intersection(val_questions)
    overlap_train_test = train_questions.intersection(test_questions)
    overlap_val_test = val_questions.intersection(test_questions)
    
    logger.info(f"Question overlap check:")
    logger.info(f"  Train-Val overlap: {len(overlap_train_val)} (should be 0)")
    logger.info(f"  Train-Test overlap: {len(overlap_train_test)} (should be 0)")
    logger.info(f"  Val-Test overlap: {len(overlap_val_test)} (should be 0)")
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        logger.error("OVERLAP DETECTED!")
    else:
        logger.info("No overlap detected!")
    
    return train_data, val_data, test_data

def create_supervised_simcse_pairs(data, output_file, max_pairs=50000):

    logger.info(f"Creating SUPERVISED SimCSE pairs from {len(data)} samples...")
    
    pairs = []
    
    # collect all contexts for negative sampling
    all_contexts = []
    all_answers = []
    question_to_contexts = {}
    
    for item in data:
        question = item['question'].strip()
        contexts = item['context']['contexts']
        long_answer = item['long_answer'].strip()
        
        # Store mapping for positive pairs
        if question not in question_to_contexts:
            question_to_contexts[question] = []
        
        for context in contexts:
            context = context.strip()
            if len(context) > 50:
                question_to_contexts[question].append(context)
                all_contexts.append(context)
        
        if long_answer and len(long_answer) > 50:
            all_answers.append(long_answer)
    
    # Remove duplicates but keep track of which belong to which question
    all_contexts = list(set(all_contexts))
    all_answers = list(set(all_answers))
    
    logger.info(f"Collected {len(all_contexts)} unique contexts and {len(all_answers)} unique answers")
    
    # Generate positive pairs
    positive_pairs = []
    for item in data:
        question = item['question'].strip()
        contexts = item['context']['contexts']
        long_answer = item['long_answer'].strip()
        
        # Strategy 1: Question-Answer pairs (POSITIVE)
        if question and long_answer and len(long_answer) > 50:
            positive_pairs.append([question, long_answer, 1])
        
        # Strategy 2: Question-Context pairs (POSITIVE)
        if contexts and question:
            for context in contexts:
                context = context.strip()
                if len(context) > 50:
                    positive_pairs.append([question, context, 1])
        
        # Strategy 3: Context-Answer pairs (POSITIVE)
        if contexts and long_answer and len(long_answer) > 50:
            for context in contexts[:2]:  # Limit to first 2 contexts
                context = context.strip()
                if len(context) > 50:
                    positive_pairs.append([context, long_answer, 1])
    
    logger.info(f"Generated {len(positive_pairs)} positive pairs")
    
    # Generate HARD negative pairs
    negative_pairs = []
    
    # Method 1: Random negatives (easy negatives)
    for item in data:
        question = item['question'].strip()
        
        # Random wrong contexts
        current_contexts = question_to_contexts.get(question, [])
        wrong_contexts = [ctx for ctx in all_contexts if ctx not in current_contexts]
        
        if wrong_contexts:
            # Sample 2-3 random wrong contexts per question
            sampled_wrong = random.sample(wrong_contexts, min(3, len(wrong_contexts)))
            for wrong_ctx in sampled_wrong:
                negative_pairs.append([question, wrong_ctx, 0])
        
        # Random wrong answers
        if len(all_answers) > 1:
            # Get wrong answers
            current_answer = item['long_answer'].strip()
            wrong_answers = [ans for ans in all_answers if ans != current_answer]
            
            if wrong_answers:
                sampled_wrong_ans = random.sample(wrong_answers, min(2, len(wrong_answers)))
                for wrong_ans in sampled_wrong_ans:
                    negative_pairs.append([question, wrong_ans, 0])
    
    # Method 2: Hard negatives using TF-IDF similarity
    logger.info("Generating hard negatives using TF-IDF similarity...")
    
    def generate_tfidf_hard_negatives(positive_pairs, all_contexts, similarity_threshold=0.3):
        """Generate hard negatives that are similar but incorrect"""
        hard_negatives = []
        
        if len(all_contexts) < 100:  # Skip if too few contexts
            return hard_negatives
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
            tfidf_matrix = vectorizer.fit_transform(all_contexts)
            
            # For each positive question-context pair, find similar but wrong contexts
            question_context_pairs = [(p[0], p[1]) for p in positive_pairs if len(p) >= 2]
            
            for question, correct_context in question_context_pairs[:500]:  # Limit to avoid too many
                try:
                    correct_idx = all_contexts.index(correct_context)
                    
                    # Calculate similarity with all other contexts
                    similarities = cosine_similarity(tfidf_matrix[correct_idx:correct_idx+1], tfidf_matrix).flatten()
                    
                    # Find contexts that are similar but not the correct one
                    similar_indices = np.where(
                        (similarities > similarity_threshold) & 
                        (similarities < 0.8) &  # Not too similar
                        (np.arange(len(similarities)) != correct_idx)
                    )[0]
                    
                    # Add hard negatives (limit to 1 per positive pair)
                    for idx in similar_indices[:1]:
                        hard_negatives.append([question, all_contexts[idx], 0])
                        
                except (ValueError, IndexError):
                    continue  # Skip if context not found
                    
        except Exception as e:
            logger.warning(f"TF-IDF hard negative generation failed: {e}")
            
        return hard_negatives
    
    hard_negatives = generate_tfidf_hard_negatives(positive_pairs, all_contexts)
    negative_pairs.extend(hard_negatives)
    
    logger.info(f"Generated {len(negative_pairs)} total negative pairs ({len(hard_negatives)} hard negatives)")
    
    # Method 3: Medical domain-specific negatives
    def generate_medical_negatives(positive_pairs):
        """Generate medical domain-specific negatives"""
        medical_negatives = []
        
        # Separate by answer types
        yes_pairs = [p for p in positive_pairs if any(word in p[1].lower() for word in ['yes', 'positive', 'increased', 'higher', 'effective'])]
        no_pairs = [p for p in positive_pairs if any(word in p[1].lower() for word in ['no', 'negative', 'decreased', 'lower', 'ineffective'])]
        
        # Cross-pair yes questions with no answers
        for yes_pair in yes_pairs[:50]:  # Limit to avoid explosion
            for no_pair in no_pairs[:2]:  # Just 2 per yes question
                if yes_pair[0] != no_pair[0]:  # Different questions
                    medical_negatives.append([yes_pair[0], no_pair[1], 0])
        
        return medical_negatives
    
    medical_negatives = generate_medical_negatives(positive_pairs)
    negative_pairs.extend(medical_negatives)
    
    logger.info(f"Added {len(medical_negatives)} medical domain negatives")
    
    # Combine and balance
    all_pairs = positive_pairs + negative_pairs
    
    # Balance positive and negative pairs
    pos_count = len(positive_pairs)
    neg_count = len(negative_pairs)
    
    if neg_count > pos_count * 1.5:  # Too many negatives
        negative_pairs = random.sample(negative_pairs, int(pos_count * 1.2))
    elif neg_count < pos_count * 0.8:  # Too few negatives
        # Add more random negatives
        additional_needed = int(pos_count * 0.8) - neg_count
        logger.info(f"Adding {additional_needed} additional random negatives...")
        
        for _ in range(additional_needed):
            random_question = random.choice([p[0] for p in positive_pairs])
            random_wrong_context = random.choice(all_contexts)
            negative_pairs.append([random_question, random_wrong_context, 0])
    
    # Final dataset
    final_pairs = positive_pairs + negative_pairs
    random.shuffle(final_pairs)
    
    # Limit total pairs
    if len(final_pairs) > max_pairs:
        final_pairs = final_pairs[:max_pairs]
    
    pos_final = sum(1 for p in final_pairs if p[2] == 1)
    neg_final = sum(1 for p in final_pairs if p[2] == 0)
    
    logger.info(f"Final training set: {len(final_pairs)} pairs ({pos_final} positive, {neg_final} negative)")
    logger.info(f"Positive/Negative ratio: {pos_final/neg_final:.2f}")
    
    # Save with labels (SUPERVISED SimCSE format)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sent0', 'sent1', 'label'])  # Include label column
        for pair in final_pairs:
            writer.writerow(pair)
    
    logger.info(f"Saved {len(final_pairs)} SUPERVISED training pairs to {output_file}")
    return output_file

def create_labeled_pairs_from_data(data, output_file, pair_type="test"):
    """
    Create labeled pairs (positive/negative) from specific data split
    """
    logger.info(f"Creating labeled {pair_type} pairs from {len(data)} samples...")
    
    pairs = []
    
    # Collect all contexts for negative sampling
    all_contexts = []
    for item in data:
        contexts = item['context']['contexts']
        all_contexts.extend([c.strip() for c in contexts if len(c.strip()) > 50])
    
    # Remove duplicates from contexts
    all_contexts = list(set(all_contexts))
    
    for i, item in enumerate(data):
        question = item['question'].strip()
        contexts = item['context']['contexts']
        long_answer = item['long_answer'].strip()
        
        # Positive pairs
        if question and long_answer and len(long_answer) > 30:
            pairs.append([question, long_answer, 1])
        
        if contexts and question:
            context = contexts[0].strip()
            if len(context) > 30:
                pairs.append([question, context, 1])
        
        # Negative pairs (use contexts from other questions)
        if question and len(all_contexts) > i + 10:
            # Sample a few negative contexts
            negative_indices = [(i + 10 + j) % len(all_contexts) for j in range(3)]
            for neg_idx in negative_indices:
                negative_context = all_contexts[neg_idx]
                if negative_context and len(negative_context) > 30:
                    pairs.append([question, negative_context, 0])
                if len(pairs) % 100 == 0:  
                    break
    
    # Shuffle pairs
    random.shuffle(pairs)
    
    # Balance positive and negative pairs
    positive_pairs = [p for p in pairs if p[2] == 1]
    negative_pairs = [p for p in pairs if p[2] == 0]
    
    # Take equal numbers of positive and negative
    min_count = min(len(positive_pairs), len(negative_pairs))
    balanced_pairs = positive_pairs[:min_count] + negative_pairs[:min_count]
    random.shuffle(balanced_pairs)
    
    logger.info(f"Created {len(balanced_pairs)} balanced pairs ({min_count} positive, {min_count} negative)")
    
    # Save labeled pairs
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sent1', 'sent2', 'label'])
        for pair in balanced_pairs:
            writer.writerow(pair)
    
    logger.info(f"Saved {len(balanced_pairs)} labeled pairs to {output_file}")
    return output_file

def verify_no_overlap(train_file, val_file, test_file):
    
    logger.info("Verifying no overlap between train/val/test...")
    
    # Load the files
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    # Extract questions from each file
    train_questions = set()
    for _, row in train_df.iterrows():
        train_questions.add(row['sent0'].strip().lower())
        train_questions.add(row['sent1'].strip().lower())
    
    val_questions = set(val_df['sent1'].str.strip().str.lower())
    test_questions = set(test_df['sent1'].str.strip().str.lower())
    
    # Check overlaps
    train_val_overlap = train_questions.intersection(val_questions)
    train_test_overlap = train_questions.intersection(test_questions)
    val_test_overlap = val_questions.intersection(test_questions)
    
    logger.info(f"Overlap analysis:")
    logger.info(f"  Train questions: {len(train_questions)}")
    logger.info(f"  Val questions: {len(val_questions)}")
    logger.info(f"  Test questions: {len(test_questions)}")
    logger.info(f"  Train Val overlap: {len(train_val_overlap)}")
    logger.info(f"  Train Test overlap: {len(train_test_overlap)}")
    logger.info(f"  Val Test overlap: {len(val_test_overlap)}")
    
    if train_test_overlap:
        logger.error("Train Test overlap detected!")
        logger.error("Sample overlapping questions:")
        for q in list(train_test_overlap)[:3]:
            logger.error(f"  '{q[:100]}...'")
        return False
    else:
        logger.info("No train test overlap!")
        return True

def main():
    logger.info("Starting PubMedQA preprocessing with SUPERVISED SimCSE (labeled training pairs)...")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    try:
        # Load dataset
        logger.info("Loading PubMedQA dataset...")
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
        train_data = dataset['train']
        
        # Split dataset by questions 
        train_split, val_split, test_split = split_dataset_by_questions(
            train_data, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15
        )
        
        # Create SUPERVISED training pairs 
        train_file = create_supervised_simcse_pairs(
            train_split,
            output_file="data/pubmedqa_train_supervised.csv",
            max_pairs=30000
        )
        
        # Create validation pairs (labeled)
        val_file = create_labeled_pairs_from_data(
            val_split,
            output_file="data/pubmedqa_val_clean.csv",
            pair_type="validation"
        )
        
        # Create test pairs (labeled)
        test_file = create_labeled_pairs_from_data(
            test_split,
            output_file="data/pubmedqa_test_clean.csv",
            pair_type="test"
        )
        
        logger.info("Files created:")
        logger.info(f"   Training (SUPERVISED): {train_file}")
        logger.info(f"   Validation: {val_file}")
        logger.info(f"   Test: {test_file}")
        
        # Show sample of training data
        logger.info("\nSample training pairs:")
        train_df = pd.read_csv(train_file)
        for i, row in train_df.head(3).iterrows():
            label_text = "POSITIVE" if row['label'] == 1 else "NEGATIVE"
            logger.info(f"  {label_text}: '{row['sent0'][:50]}...' -> '{row['sent1'][:50]}...'")
        
        pos_count = sum(train_df['label'] == 1)
        neg_count = sum(train_df['label'] == 0)
        logger.info(f"\nTraining set balance: {pos_count} positive, {neg_count} negative")
        
        is_clean = verify_no_overlap(train_file, val_file, test_file)
        
        if is_clean:
            logger.info("Clean SimCSE dataset created!")
        else:
            logger.error("FAILED: Data leakage detected.")
            
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()
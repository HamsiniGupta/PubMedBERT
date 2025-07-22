from transformers import AutoTokenizer, AutoModel  # Required for baseline BERT
import torch  # Needed for tensors and no_grad
from sklearn.metrics.pairwise import cosine_similarity  # Used but not imported
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

def compare_embeddings(baseline_model, trained_model, test_texts):
    """Compare embeddings before and after training"""
    
    # Extract embeddings from both models
    baseline_embs = extract_embeddings(baseline_model, tokenizer, test_texts, device)
    trained_embs = extract_embeddings(trained_model, tokenizer, test_texts, device)
    
    # Compare similarities
    for i, text in enumerate(test_texts):
        print(f"Text: {text}")
        print(f"Baseline embedding norm: {np.linalg.norm(baseline_embs[i])}")
        print(f"Trained embedding norm: {np.linalg.norm(trained_embs[i])}")
        print(f"Cosine similarity: {cosine_similarity(baseline_embs[i], trained_embs[i])}")

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



def analyze_medical_concepts(model, tokenizer, medical_pairs):
    """Analyze how well model captures medical relationships"""
    
    similarities = []
    for pair in medical_pairs:
        emb1 = extract_embeddings(model, tokenizer, [pair[0]], device)[0]
        emb2 = extract_embeddings(model, tokenizer, [pair[1]], device)[0]
        
        sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0, 0]
        similarities.append((pair, sim))
        
    return similarities
# Load baseline BERT
baseline_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

# Load your trained model  
from sentence_transformers import SentenceTransformer
trained_model = SentenceTransformer("output/pubmedqa-supervised-simcse")


# Compare on your exercise example
test_texts = [
    "A total of 123 dysphonic individuals with benign vocal pathologies were recruited. They were given either genuine acupuncture (n = 40), sham acupuncture (n = 44), or no treatment (n = 39) for 6 weeks (two 30-minute sessions/wk). The genuine acupuncture group received needles puncturing nine voice-related acupoints for 30 minutes, two times a week for 6 weeks, whereas the sham acupuncture group received blunted needles stimulating the skin surface of the nine acupoints for the same frequency and duration. The no-treatment group did not receive any intervention but attended just the assessment sessions. One-hundred seventeen subjects completed the study (genuine acupuncture = 40; sham acupuncture = 43; and no treatment = 34), but only 84 of them had a complete set of vocal functions and quality of life measures (genuine acupuncture = 29; sham acupuncture = 33; and no-treatment = 22) and 42 of them with a complete set of endoscopic data (genuine acupuncture = 16; sham acupuncture = 15; and no treatment = 11).",
    "Is Acupuncture Efficacious for Treating Phonotraumatic Vocal Pathologies?"
]

# Extract and compare embeddings
baseline_embs = extract_embeddings(baseline_model, tokenizer, test_texts, device)
trained_embs = extract_embeddings(trained_model, tokenizer, test_texts, device)

# Calculate similarities
baseline_sim = cosine_similarity(baseline_embs[0:1], baseline_embs[1:2])[0, 0]
trained_sim = cosine_similarity(trained_embs[0:1], trained_embs[1:2])[0, 0]

print(f"Baseline similarity: {baseline_sim:.3f}")
print(f"Trained similarity: {trained_sim:.3f}")
print(f"Improvement: {trained_sim - baseline_sim:.3f}")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def detailed_single_question_analysis(baseline_model, trained_model, tokenizer, device):
    """Detailed analysis of your single acupuncture question-context pair"""
    
    question = "Is Acupuncture Efficacious for Treating Phonotraumatic Vocal Pathologies?"
    context = "A total of 123 dysphonic individuals with benign vocal pathologies were recruited. They were given either genuine acupuncture (n = 40), sham acupuncture (n = 44), or no treatment (n = 39) for 6 weeks (two 30-minute sessions/wk). The genuine acupuncture group received needles puncturing nine voice-related acupoints for 30 minutes, two times a week for 6 weeks, whereas the sham acupuncture group received blunted needles stimulating the skin surface of the nine acupoints for the same frequency and duration. The no-treatment group did not receive any intervention but attended just the assessment sessions. One-hundred seventeen subjects completed the study (genuine acupuncture = 40; sham acupuncture = 43; and no treatment = 34), but only 84 of them had a complete set of vocal functions and quality of life measures (genuine acupuncture = 29; sham acupuncture = 33; and no-treatment = 22) and 42 of them with a complete set of endoscopic data (genuine acupuncture = 16; sham acupuncture = 15; and no treatment = 11)."

    print("=" * 80)
    print("DETAILED ANALYSIS: ACUPUNCTURE QUESTION-CONTEXT PAIR")
    print("=" * 80)
    
    # Extract embeddings
    print("Extracting embeddings...")
    baseline_q_emb = extract_embeddings(baseline_model, tokenizer, [question], device)[0]
    baseline_c_emb = extract_embeddings(baseline_model, tokenizer, [context], device)[0]
    
    trained_q_emb = extract_embeddings(trained_model, tokenizer, [question], device)[0]
    trained_c_emb = extract_embeddings(trained_model, tokenizer, [context], device)[0]
    
    # Calculate similarities
    baseline_sim = cosine_similarity(baseline_q_emb.reshape(1, -1), baseline_c_emb.reshape(1, -1))[0, 0]
    trained_sim = cosine_similarity(trained_q_emb.reshape(1, -1), trained_c_emb.reshape(1, -1))[0, 0]
    improvement = trained_sim - baseline_sim
    
    print(f"\n1. SIMILARITY SCORES:")
    print(f"Baseline BERT:     {baseline_sim:.3f}")
    print(f"SimCSE-trained:    {trained_sim:.3f}")
    print(f"Improvement:       {improvement:+.3f} ({improvement/baseline_sim*100:+.1f}%)")
    
    # Embedding norms and properties
    print(f"\n2. EMBEDDING PROPERTIES:")
    print(f"Question embeddings:")
    print(f"Baseline norm:   {np.linalg.norm(baseline_q_emb):.3f}")
    print(f"Trained norm:    {np.linalg.norm(trained_q_emb):.3f}")
    print(f"Change:          {np.linalg.norm(trained_q_emb) - np.linalg.norm(baseline_q_emb):+.3f}")
    
    print(f"Context embeddings:")
    print(f"Baseline norm:   {np.linalg.norm(baseline_c_emb):.3f}")
    print(f"Trained norm:    {np.linalg.norm(trained_c_emb):.3f}")
    print(f"Change:          {np.linalg.norm(trained_c_emb) - np.linalg.norm(baseline_c_emb):+.3f}")
    
    # Dimension analysis - top changed dimensions
    print(f"\n3. TOP CHANGED DIMENSIONS:")
    
    # Question embedding changes
    q_diff = np.abs(trained_q_emb - baseline_q_emb)
    q_top_dims = np.argsort(q_diff)[-10:]
    
    print(f"Question embedding - Top 10 most changed dimensions:")
    for i, dim in enumerate(reversed(q_top_dims)):
        change = trained_q_emb[dim] - baseline_q_emb[dim]
        print(f"Dim {dim:3d}: {baseline_q_emb[dim]:+.3f} ? {trained_q_emb[dim]:+.3f} (?{change:+.3f})")
    
    # Context embedding changes  
    c_diff = np.abs(trained_c_emb - baseline_c_emb)
    c_top_dims = np.argsort(c_diff)[-10:]
    
    print(f"\nContext embedding - Top 10 most changed dimensions:")
    for i, dim in enumerate(reversed(c_top_dims)):
        change = trained_c_emb[dim] - baseline_c_emb[dim]
        print(f"Dim {dim:3d}: {baseline_c_emb[dim]:+.3f} ? {trained_c_emb[dim]:+.3f} (?{change:+.3f})")
    
    # Cross-embedding analysis
    print(f"\n4. CROSS-EMBEDDING SIMILARITY:")
    
    # How similar are question embeddings between models
    q_self_sim = cosine_similarity(baseline_q_emb.reshape(1, -1), trained_q_emb.reshape(1, -1))[0, 0]
    c_self_sim = cosine_similarity(baseline_c_emb.reshape(1, -1), trained_c_emb.reshape(1, -1))[0, 0]
    
    print(f"Question: baseline trained = {q_self_sim:.3f}")
    print(f"Context:  baseline trained = {c_self_sim:.3f}")
    print(f"Interpretation: {q_self_sim:.3f} means embeddings are {'very similar' if q_self_sim > 0.8 else 'moderately similar' if q_self_sim > 0.6 else 'quite different'}")
    
    # Statistical summary
    print(f"\n5. STATISTICAL SUMMARY:")
    print(f"Embedding dimension: {len(baseline_q_emb)}")
    print(f"Mean absolute change (question): {np.mean(q_diff):.4f}")
    print(f"Std absolute change (question):  {np.std(q_diff):.4f}")
    print(f"Max absolute change (question):  {np.max(q_diff):.4f}")
    
    print(f"Mean absolute change (context):  {np.mean(c_diff):.4f}")
    print(f"Std absolute change (context):   {np.std(c_diff):.4f}")
    print(f"Max absolute change (context):   {np.max(c_diff):.4f}")
    

    return {
        'baseline_similarity': baseline_sim,
        'trained_similarity': trained_sim, 
        'improvement': improvement,
        'question_self_similarity': q_self_sim,
        'context_self_similarity': c_self_sim,
        'top_question_dims': q_top_dims,
        'top_context_dims': c_top_dims
    }

# Run the analysis
results = detailed_single_question_analysis(baseline_model, trained_model, tokenizer, device)
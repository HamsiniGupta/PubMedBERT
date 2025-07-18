import pandas as pd
import os
import re
from typing import List, Dict
import torch
import time
import numpy as np

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline as hf_pipeline
from langchain_huggingface.llms import HuggingFacePipeline

# =====================

from huggingface_hub import HfApi
from PubMedQAProcessor import PubMedQAProcessor
from WeaviateManager import WeaviateManager
from RAGPipeline import RAGPipeline
from SimCSEEmbeddings import SimCSEEmbeddings

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Debug: verify they're loaded (remove this after testing)
print(f"HF Token loaded: {bool(os.getenv('HUGGINGFACE_TOKEN'))}")
print(f"Weaviate key loaded: {bool(os.getenv('WEAVIATE_API_KEY'))}")
print(f"Weaviate key starts with: {os.getenv('WEAVIATE_API_KEY', '')[:10]}...")

# Setting HF token
# Access the token from the environment variable

import os
hf_token = os.getenv('HUGGINGFACE_TOKEN')
if not hf_token:
    raise ValueError("Please set HUGGINGFACE_TOKEN environment variable")

# Use the token to authenticate
api = HfApi()

# =====================
from transformers import AutoModelForCausalLM, AutoTokenizer

# Defining LLM 
def llm_model(model_name="meta-llama/Llama-3.1-8B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.1,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.2,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    llm.model_rebuild()  # Fixes the PydanticUserError
    return llm

# =====================

import os

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SIMCSE_MODEL_PATH = os.path.join(CURRENT_FILE_DIR, "..", "output", "pubmedqa-supervised-simcse")
SIMCSE_MODEL_PATH = os.path.abspath(SIMCSE_MODEL_PATH)

# Verify it exists
if os.path.exists(SIMCSE_MODEL_PATH):
    print(f"Found SimCSE model at: {SIMCSE_MODEL_PATH}")
else:
    print(f"SimCSE model not found at: {SIMCSE_MODEL_PATH}")

# Configuration for weaviate and huggingface
WEAVIATE_URL = os.getenv('WEAVIATE_URL', 'https://wrqxngldroq4aybpj2a8vw.c0.us-west3.gcp.weaviate.cloud')

# Load Weaviate API key from environment variable
import os
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')
if not WEAVIATE_API_KEY:
    raise ValueError("Weaviate API key not found in environment variables")

def initialize_rag_with_simcse():
    """Initialize the RAG system with your trained SimCSE model"""
    print("Initializing LLM...")
    llm = llm_model("meta-llama/Llama-3.1-8B")

    print("Connecting to Weaviate...")
    weaviate_manager = WeaviateManager(WEAVIATE_URL, 
        WEAVIATE_API_KEY, hf_token, simcse_model_path=SIMCSE_MODEL_PATH)

    print("Loading and processing PubMedQA dataset...")
    dataset = PubMedQAProcessor.load_pubmedqa_dataset()
    documents = PubMedQAProcessor.process_contexts_to_documents(dataset)

    print("Reindexing documents with SimCSE...")
    collection_name = weaviate_manager.reindex_with_simcse(documents, "output/pubmedqa-supervised-simcse")
    
    weaviate_manager.collection_name = collection_name

    print("Creating RAG pipeline...")
    # Initialize SimCSE embeddings for query encoding
    embeddings = SimCSEEmbeddings(SIMCSE_MODEL_PATH)
    
    rag = RAGPipeline(weaviate_manager, embeddings, llm)
    pipeline = rag.create_pipeline()

    return pipeline, weaviate_manager, embeddings

# =====================

# Load dataset directly if embeddings already stored
def load_initialized_rag():
    """Load an already initialized RAG system without reloading the dataset."""
    print("Initializing LLM...")
    llm = llm_model("meta-llama/Llama-3.1-8B")

    print("Connecting to Weaviate...")
    weaviate_manager = WeaviateManager(WEAVIATE_URL, 
            WEAVIATE_API_KEY, hf_token, simcse_model_path=SIMCSE_MODEL_PATH)
    
    print("Loading SimCSE embeddings...")
    embeddings = SimCSEEmbeddings(SIMCSE_MODEL_PATH)
    
    print("Creating RAG pipeline...")
    rag = RAGPipeline(weaviate_manager, embeddings, llm)
    pipeline = rag.create_pipeline()

    return pipeline, weaviate_manager, embeddings  

collection_name = "PMQA_PubMedBert"
FORCE_REINDEX = False

# Temporary client to check collection existence
temp_manager = WeaviateManager(WEAVIATE_URL, WEAVIATE_API_KEY, 
            hf_token, simcse_model_path=SIMCSE_MODEL_PATH)
client = temp_manager.client

if FORCE_REINDEX:
    if client.collections.exists(collection_name):
        print("[FORCE] Deleting existing collection...")
        client.collections.delete(collection_name)
    pipeline, weaviate_manager, embeddings = initialize_rag_with_simcse()

elif client.collections.exists(collection_name):
    print("[INFO] Collection exists. Loading RAG...")
    pipeline, weaviate_manager, embeddings = load_initialized_rag()

else:
    print("[INFO] Collection does not exist. Initializing RAG...")
    pipeline, weaviate_manager, embeddings = initialize_rag_with_simcse()

import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams

# List all collections
collections = client.collections.list_all()
print("Collections in Weaviate:")
for collection in collections:
    print("-", collection) 

# Import the test generation script
from testPubMedBert import run_test_generation

print("\nGenerating Test Results for Evaluation...")

# Generate test results using your loaded pipeline
df, results_file = run_test_generation(
    pipeline=pipeline,
    weaviate_manager=weaviate_manager,
    embeddings=embeddings
)

print(f"\nTest generation completed!")
print(f"Results saved to: {results_file}")
print(f"\nNext steps:")
print(f"1. Use {results_file} in AnswerRelevancy.ipynb")
print(f"2. Use {results_file} in ContextAdherence.ipynb") 
print(f"3. Use {results_file} in ContextRelevancy.ipynb")

weaviate_manager.close()
client.close()

print("Weaviate client closed")

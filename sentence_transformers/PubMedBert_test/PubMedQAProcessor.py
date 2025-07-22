from typing import List, Dict

from datasets import load_dataset

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import time

start = time.time()

# PubMedQA dataset processor
class PubMedQAProcessor:
    """Handles loading and processing of PubMedQA dataset."""
    
    @staticmethod
    def load_pubmedqa_dataset():
        """Load the PubMedQA dataset from Hugging Face."""
        print("Loading PubMedQA dataset from Hugging Face...")
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
        return dataset
    
    @staticmethod
    def process_contexts_to_documents(dataset) -> List[Document]:
        """Process contexts from the dataset into Document objects."""
        documents = []
        
        # Access the training split of the dataset
        train_data = dataset["train"]
        
        for i, item in enumerate(train_data):
            # Extract PubMed ID if available
            pub_id = item.get("pubid", f"pubid_{i}")
            
            # Extract and process context based on dataset structure
            context = item.get("context", {})
            question = item.get("question", "")
            
            # Check if context is a dictionary with 'contexts' key or just a list
            if isinstance(context, dict) and "contexts" in context:
                # Join all contexts into a single string
                context_text = " ".join(context["contexts"])
            elif isinstance(context, list):
                # If it's already a list, join them directly
                context_text = " ".join(context)
            else:
                # If it's a string or some other format, convert to string
                context_text = str(context) if context else ""
            
            # Skip empty contexts
            if not context_text.strip():
                continue
                
            # Create metadata with question and pubid
            metadata = {
                "source": f"PubMedQA_{pub_id}",
                "question": question
            }
            
            # Create a Document with the context
            doc = Document(
                page_content=context_text,
                metadata=metadata
            )
            documents.append(doc)
            
        print(f"Processed {len(documents)} documents from PubMedQA dataset")
        return documents
    
    @staticmethod
    def split_documents(documents: List[Document], chunk_size=2000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "- ", "-\n", ". ", " ", ""]
        )
    
        # Use the built-in split_documents method
        split_docs = text_splitter.split_documents(documents)            
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        return split_docs
end = time.time()
total = end - start
print(f"{total} seconds taken for Processor")
# =====================
from typing import List, Dict
import torch
import numpy as np
import time

# =====================

from weaviate.classes.config import Configure, Property, DataType, VectorDistances
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery

# =====================

from langchain_core.documents import Document
from SimCSEEmbeddings import SimCSEEmbeddings

# =====================

start = time.time()
class WeaviateManager:
    
    def __init__(self, url: str, api_key: str, hf_token: str, simcse_model_path: str = "pubmedqa-simcse-model"):
        """Initialize Weaviate client with connection details."""
        self.url = url
        self.api_key = api_key
        self.hf_token = hf_token
        self.simcse_model_path = simcse_model_path
        
        # Initialize Weaviate client
        self._connect_to_weaviate()

        # Device config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading PubMedRAG for query encoding...")
        self.simcse_embeddings = SimCSEEmbeddings(simcse_model_path)
        print("PubMedRAG loaded successfully!")

    def _connect_to_weaviate(self):
        
        print(f"Connecting to Weaviate...")
        
        try:
            print("Trying API Key authentication...")
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.url,
                auth_credentials=Auth.api_key(self.api_key),
                headers={"X-HuggingFace-Api-Key": self.hf_token} if self.hf_token else None,
            )
            
            # Test connection
            if self.client.is_ready():
                print("Connected successfully with API Key authentication")
                return
            else:
                print("Client not ready")
                self.client.close()
                
        except Exception as e:
            print(f"API Key auth failed: {e}")
            try:
                self.client.close()
            except:
                pass
        

    def reindex_with_simcse(self, documents, simcse_model_path=None):
        """Reindex documents using PubMedRAG"""
        if simcse_model_path is None:
            simcse_model_path = self.simcse_model_path
            
        print("Initializing PubMedRAG embeddings...")
        
        # Use the existing PubMedRAG embeddings 
        simcse_embeddings = self.simcse_embeddings
        
        collection_name = "PMQA_PubMedRAG" 
        
        # Delete existing collection if it exists
        if self.client.collections.exists(collection_name):
            print(f"Deleting existing collection: {collection_name}")
            self.client.collections.delete(collection_name)
        
        # Create new collection for SimCSE
        print(f"Creating collection: {collection_name}")
        self.client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.none(),  # provide PubMedRAG vectors
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.TEXT),
                Property(name="context_id", data_type=DataType.TEXT),
            ]
        )
        
        collection = self.client.collections.get(collection_name)
        
        print(f"Encoding {len(documents)} documents with PubMedRAG...")
        
        # Process documents in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            # Extract text content
            texts = [doc.page_content for doc in batch_docs]
            
            # Get embeddings from model
            embeddings = simcse_embeddings.encode(texts)
            
            with collection.batch.dynamic() as batch:
                for j, doc in enumerate(batch_docs):
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    
                    batch.add_object(
                        properties={
                            "content": doc.page_content,
                            "source": str(metadata.get("source", "")),
                            "document_id": str(metadata.get("document_id", "")),
                            "context_id": str(metadata.get("context_id", "")),
                        },
                        vector=embeddings[j].tolist()
                    )
            
            print(f"Processed {min(i + batch_size, len(documents))}/{len(documents)} documents")
        
        print(f"Successfully indexed {len(documents)} documents with PubMedRAG!")
        return collection_name
            
    def create_schema(self):
        schema = "PMQA_PubMedRAG"
        if self.client.collections.exists(schema):
            print("Schema already exists")
            return

        self.client.collections.create(
            schema,
            vectorizer_config=Configure.Vectorizer.none(),  
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.TEXT),
                Property(name="context_id", data_type=DataType.TEXT),
            ]
        )
        print("Schema created successfully")

    def add_documents(self, documents: List[Document], embeddings_model=None) -> None:
        collection = self.client.collections.get("PMQA_PubMedRAG")
       
        with collection.batch.fixed_size(batch_size=100) as batch:
            for doc in documents:
                # Generate embedding using PubMedRAG
                vector = self.simcse_embeddings.embed_query(doc.page_content)
                
                if hasattr(vector, 'tolist'):
                    vector = vector.tolist()
                elif isinstance(vector, np.ndarray):
                    vector = vector.tolist()
                if not isinstance(vector, list):
                    vector = list(vector)
                    
                batch.add_object(
                    properties={
                        "content": doc.page_content,
                        "source": str(doc.metadata.get("source", "")),
                        "document_id": str(doc.metadata.get("document_id", "")),
                        "context_id": str(doc.metadata.get("context_id", "")),
                    },
                    vector=vector
                )
        
        if collection.batch.failed_objects:
            print(f"Failed to import {len(collection.batch.failed_objects)} documents")
            print("First failure:", collection.batch.failed_objects[0])                    

    def embed_doc(self, text: str):
        return self.simcse_embeddings.embed_query(text)

    def search_documents(self, query: str, limit: int = 2) -> List[Document]:
        
        # Get the query 
        query_embedding = self.simcse_embeddings.embed_query(query)
        
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        elif isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        if not isinstance(query_embedding, list):
            query_embedding = list(query_embedding)
        
        print(f"Query embedding type: {type(query_embedding)}, length: {len(query_embedding)}")

        collection = self.client.collections.get("PMQA_PubMedRAG")
    
        # Use hybrid search with PubMedRAG
        results = collection.query.hybrid(
            query=query,
            vector=query_embedding,
            limit=limit,
            alpha=0.65,  # Balance between vector search (0.0) and keyword search (1.0)
            return_metadata=MetadataQuery(score=True, explain_score=True, distance=True),
        )

        candidates = []
        for i, result in enumerate(results.objects):
            properties = result.properties
            metadata = result.metadata

            print(f"\nDocument {i+1} Scores:")
            print(f"Score: {metadata.score}")
            print(f"Explain Score: {metadata.explain_score}")
            print(f"Distance: {metadata.distance}")

            doc = Document(
                page_content=properties.get("content", ""),
                metadata={
                    "source": properties.get("source", ""),
                    "document_id": properties.get("document_id", ""),
                    "context_id": properties.get("context_id", ""),
                }
            )
            
            doc.metadata["score"] = metadata.score
            doc.metadata["distance"] = metadata.distance
            
            candidates.append(doc)
        
        return candidates[:limit]
    
    def get_embedding_stats(self):
        """Get statistics about the collection"""
        collection = self.client.collections.get("PMQA_PubMedRAG")
        
        # Get collection info
        config = collection.config.get()
        print(f"Collection name: {config.name}")
        print(f"Vector index config: {config.vector_index_config}")
        print(f"Properties: {[prop.name for prop in config.properties]}")
        
        # Get some sample embeddings to check dimensions
        results = collection.query.fetch_objects(limit=1)
        if results.objects:
            sample_vector = results.objects[0].vector
            if sample_vector:
                print(f"Embedding dimension: {len(sample_vector)}")
            else:
                print("No vector found in sample object")
        
        return config
    
    def close(self):
        self.client.close()
end = time.time()
total = end - start
print(f"{total} seconds taken for Weaviate")
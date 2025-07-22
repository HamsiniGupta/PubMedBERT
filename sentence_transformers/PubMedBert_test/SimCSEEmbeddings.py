import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import numpy as np
import os
import json

class SimCSEEmbeddings:
    
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_path = model_path

        print(f"SimCSE received model_path: {model_path}")
        print(f"Path exists: {os.path.exists(model_path)}")
        print(f"Path is absolute: {os.path.isabs(model_path)}")
        
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
            self.model_path = model_path
            print(f"Converted to absolute path: {model_path}")
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Load model with config fix
        self._load_model_with_config_fix()
        
        # Load SimCSE components with error handling
        self._load_simcse_components()
        
        print(f"SimCSE model loaded successfully!")
        if hasattr(self, 'config'):
            print(f"Pooler type: {self.config.get('pooler_type', 'cls')}")
            print(f"Temperature: {self.config.get('temp', 0.05)}")
    
    def _load_model_with_config_fix(self):
        config_path = os.path.join(self.model_path, "config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            original_config = json.load(f)
        
        needs_fix = original_config.get("model_type") == "simcse"
        
        if needs_fix:
            print("Fixing config.json temporarily...")
            modified_config = original_config.copy()
            modified_config["model_type"] = "bert"
            
            with open(config_path, 'w') as f:
                json.dump(modified_config, f, indent=2)
        
        try:
            print(f"Loading SimCSE model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
            self.model.eval()
            
        finally:
            if needs_fix:
                print("Restoring original config.json...")
                with open(config_path, 'w') as f:
                    json.dump(original_config, f, indent=2)
    
    def _load_simcse_components(self):
        simcse_components_path = os.path.join(self.model_path, 'simcse_components.pt')
        
        if os.path.exists(simcse_components_path):
            try:
                simcse_components = torch.load(simcse_components_path, map_location=self.device)
                print(f"Loaded simcse_components, keys: {list(simcse_components.keys()) if isinstance(simcse_components, dict) else 'Not a dict'}")
                
                if isinstance(simcse_components, dict):
                    self.config = {
                        'pooler_type': 'cls',  # Default pooling
                        'temp': simcse_components.get('temp', 0.05),
                        'model_type': simcse_components.get('model_type', 'simcse'),
                        'base_model': simcse_components.get('base_model', 'bert-base-uncased')
                    }
                    
                    print(f"Using config from simcse_components:")
                    print(f"   Temperature: {self.config['temp']}")
                    print(f"   Model type: {self.config['model_type']}")
                    print(f"   Base model: {self.config['base_model']}")
                    
                    self.mlp = torch.nn.Identity().to(self.device)
                    print("Using identity mapping (no MLP projection)")
                    
                else:
                    print("simcse_components.pt is not a dictionary, using defaults")
                    self._use_defaults()
                    
            except Exception as e:
                print(f"Error loading simcse_components.pt: {e}")
                self._use_defaults()
        else:
            print("simcse_components.pt not found, using defaults")
            self._use_defaults()
    
    def _use_defaults(self):
        self.config = {
            'pooler_type': 'cls',
            'temp': 0.05,
            'model_type': 'simcse'
        }
        self.mlp = torch.nn.Identity().to(self.device)
    
    def pooling(self, last_hidden_state, attention_mask):
        pooler_type = self.config.get('pooler_type', 'cls')
        
        if pooler_type == "cls":
            return last_hidden_state[:, 0]  # CLS token
        elif pooler_type == "avg":
            # Average pooling with attention mask
            sum_embeddings = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1)
            sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)
            return sum_embeddings / sum_mask
        else:
            # Default to CLS
            return last_hidden_state[:, 0]
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, max_length: int = 128) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get BERT embeddings
                outputs = self.model(**inputs)
                
                # Apply pooling
                embeddings = self.pooling(outputs.last_hidden_state, inputs['attention_mask'])
                
                if not isinstance(self.mlp, torch.nn.Identity):
                    embeddings = self.mlp(embeddings)
                
                # Normalize 
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        result = np.vstack(all_embeddings)
        
        # Return single embedding if single input
        if len(texts) == 1:
            return result[0]
        return result
    
    def embed_query(self, text: str) -> List[float]:
    
        embedding = self.encode(text)
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.encode(texts)
        if embeddings.ndim == 1:
            return [embeddings.tolist()]
        return embeddings.tolist()
    
    @property
    def embedding_dimension(self) -> int:
        return self.model.config.hidden_size

if __name__ == "__main__":
    model_path = "output/pubmedqa-supervised-simcse" 
    try:
        embedder = SimCSEEmbeddings(model_path)
        
        # Test single query
        test_query = "Do mitochondria play a role in programmed cell death?"
        embedding = embedder.embed_query(test_query)
        print(f"Query embedding type: {type(embedding)}, length: {len(embedding)}")
        
        # Test batch documents
        test_docs = [
            "Programmed cell death occurs in many organisms.",
            "Mitochondria are involved in cellular processes."
        ]
        embeddings = embedder.embed_documents(test_docs)
        print(f"Document embeddings type: {type(embeddings)}, shape: {len(embeddings)} x {len(embeddings[0])}")
        
        # Test medical similarity
        med_q = "Do mitochondria play a role in programmed cell death?"
        med_a = "Mitochondria are involved in cellular apoptosis."
        unrelated = "What is the weather today?"
        
        emb_q = embedder.encode(med_q)
        emb_a = embedder.encode(med_a)
        emb_u = embedder.encode(unrelated)
        
        sim_relevant = np.dot(emb_q, emb_a) / (np.linalg.norm(emb_q) * np.linalg.norm(emb_a))
        sim_irrelevant = np.dot(emb_q, emb_u) / (np.linalg.norm(emb_q) * np.linalg.norm(emb_u))
        
        print(f"Medical similarity test:")
        print(f"   Relevant: {sim_relevant:.3f}")
        print(f"   Irrelevant: {sim_irrelevant:.3f}")
        print(f"   Gap: {sim_relevant - sim_irrelevant:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
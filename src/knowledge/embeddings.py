"""
Embeddings for medical knowledge base
"""

import os
import warnings
from typing import List
from langchain_core.embeddings import Embeddings

# Suppress warnings
warnings.filterwarnings('ignore')

class SentenceTransformerEmbeddings(Embeddings):
    """SentenceTransformer embeddings wrapper with error handling"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            print(f"✅ SentenceTransformer model '{model_name}' loaded")
        except ImportError:
            raise ImportError("sentence-transformers package is required")
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer model: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            print(f"⚠️  SentenceTransformer embeddings error: {e}")
            # Return dummy embeddings as fallback
            return []
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            embedding = self.model.encode([text], convert_to_tensor=False)
            return embedding[0].tolist()
        except Exception as e:
            print(f"⚠️  SentenceTransformer query embedding error: {e}")
            # Return dummy embedding as fallback
            return []

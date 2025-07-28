"""
Custom embeddings for Together AI (simplified implementation)
"""

import os
from typing import List
from dotenv import load_dotenv

class TogetherEmbeddings:
    """Custom embedding class for Together AI"""
    
    def __init__(self):
        load_dotenv()
        # Check if Together API key is available
        self.api_key = os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for documents
        
        Note: This is a simplified implementation using hash-based embeddings.
        In production, you would want to use a proper embedding model like:
        - sentence-transformers
        - OpenAI embeddings
        - Hugging Face embeddings
        """
        embeddings = []
        for text in texts:
            # Create simple hash-based embedding (this is very basic)
            embedding = [
                float(hash(text[i:i+10]) % 1000) / 1000.0 
                for i in range(0, min(len(text), 100), 10)
            ]
            # Pad to consistent size
            while len(embedding) < 100:
                embedding.append(0.0)
            embeddings.append(embedding[:100])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]


class SentenceTransformerEmbeddings:
    """Alternative implementation using sentence-transformers (requires installation)"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using sentence-transformers"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]

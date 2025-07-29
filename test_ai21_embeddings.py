#!/usr/bin/env python3
"""
Test AI21 embeddings wrapper
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Set environment variables
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGCHAIN_ENDPOINT'] = ''
os.environ['LANGCHAIN_API_KEY'] = ''

def test_ai21_embeddings():
    """Test AI21 embeddings wrapper"""
    try:
        from knowledge.embeddings import AI21EmbeddingsWrapper
        
        print("🧪 Testing AI21EmbeddingsWrapper...")
        embedder = AI21EmbeddingsWrapper()
        print(f"✅ AI21EmbeddingsWrapper created: {type(embedder)}")
        
        # Test single query embedding
        query = "What are the symptoms of diabetes?"
        query_embedding = embedder.embed_query(query)
        print(f"✅ Query embedding shape: {len(query_embedding)}")
        
        # Test document embeddings
        docs = [
            "Diabetes is a metabolic disorder characterized by high blood sugar.",
            "Symptoms include frequent urination, increased thirst, and fatigue."
        ]
        doc_embeddings = embedder.embed_documents(docs)
        print(f"✅ Document embeddings shape: {len(doc_embeddings)} x {len(doc_embeddings[0])}")
        
        # Test that embeddings are callable (this was the issue)
        print(f"✅ Embedder is callable: {callable(embedder)}")
        print(f"✅ embed_query method exists: {hasattr(embedder, 'embed_query')}")
        print(f"✅ embed_documents method exists: {hasattr(embedder, 'embed_documents')}")
        
        # Test inheritance
        from langchain_core.embeddings import Embeddings
        print(f"✅ Inherits from Embeddings: {isinstance(embedder, Embeddings)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing AI21 embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Testing AI21 Embeddings Wrapper...")
    success = test_ai21_embeddings()
    
    if success:
        print("\n✅ All AI21 embeddings tests passed!")
    else:
        print("\n❌ AI21 embeddings tests failed!")
        sys.exit(1)

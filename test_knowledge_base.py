#!/usr/bin/env python3
"""
Test knowledge base with AI21 embeddings
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

def test_knowledge_base():
    """Test knowledge base with new embeddings"""
    try:
        from knowledge.knowledge_base import MedicalKnowledgeBase
        
        print("🧪 Testing MedicalKnowledgeBase...")
        
        # Initialize knowledge base (should use AI21 embeddings first)
        kb = MedicalKnowledgeBase()
        print(f"✅ Knowledge base initialized with embeddings: {type(kb.embeddings)}")
        
        # Test search functionality
        query = "What are the symptoms of diabetes?"
        print(f"🔍 Testing search with query: '{query}'")
        
        # Check if we have any documents
        if hasattr(kb, 'vector_store') and kb.vector_store:
            results = kb.search_knowledge(query, k=2)
            print(f"✅ Search returned {len(results)} results")
            for i, result in enumerate(results):
                print(f"  Result {i+1}: {result.page_content[:100]}...")
        else:
            print("⚠️  No documents in knowledge base, creating test search")
            # Test embeddings directly
            test_docs = ["Diabetes symptoms include thirst and fatigue"]
            embeddings = kb.embeddings.embed_documents(test_docs)
            print(f"✅ Test embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing knowledge base: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Testing Knowledge Base with AI21 Embeddings...")
    success = test_knowledge_base()
    
    if success:
        print("\n✅ Knowledge base test passed!")
    else:
        print("\n❌ Knowledge base test failed!")
        sys.exit(1)

#!/usr/bin/env python3
"""
Test medical diagnosis system with AI21 embeddings
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

def test_medical_diagnosis():
    """Test medical diagnosis system with AI21 embeddings"""
    try:
        from workflow.graph import MedicalDiagnosisWorkflow
        from knowledge.knowledge_base import MedicalKnowledgeBase
        
        print("🧪 Testing MedicalDiagnosisWorkflow...")
        
        # Initialize knowledge base (should use AI21 embeddings)
        knowledge_base = MedicalKnowledgeBase()
        print(f"✅ Knowledge base initialized")
        print(f"✅ Knowledge base embeddings: {type(knowledge_base.embeddings)}")
        
        # Initialize the workflow
        workflow = MedicalDiagnosisWorkflow(knowledge_base)
        print("✅ Medical diagnosis workflow initialized")
        
        # Test a simple diagnosis query
        symptoms = "Patient reports frequent urination, excessive thirst, and unexplained weight loss over the past month."
        
        print(f"🔍 Testing diagnosis with symptoms: '{symptoms[:50]}...'")
        
        # This should not give the callable error anymore
        result = workflow.run_diagnosis(symptoms)
        
        print("✅ Diagnosis completed successfully")
        print(f"📋 Response type: {type(result)}")
        
        if isinstance(result, dict):
            print("✅ Valid response structure")
            print(f"📊 Response keys: {list(result.keys())}")
            
            # Display the available results
            for key, value in result.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"� {key}: {value[:100]}...")
                else:
                    print(f"🔍 {key}: {value}")
        else:
            print(f"📄 Response: {str(result)[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing medical diagnosis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Testing Medical Diagnosis System with AI21 Embeddings...")
    success = test_medical_diagnosis()
    
    if success:
        print("\n✅ Medical diagnosis system test passed!")
    else:
        print("\n❌ Medical diagnosis system test failed!")
        sys.exit(1)

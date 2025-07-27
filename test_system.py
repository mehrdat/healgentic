#!/usr/bin/env python3
"""
Quick test script to check if the system works
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test if imports work"""
    try:
        print("🔍 Testing imports...")
        
        # Add src to Python path
        src_path = str(Path(__file__).parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Test basic imports
        from src.workflow.state import MedicalDiagnosisState
        print("  ✓ State import successful")
        
        from src.llm.together_client import TogetherMedicalLLM
        print("  ✓ Together client import successful")
        
        from src.knowledge.embeddings import TogetherEmbeddings
        print("  ✓ Embeddings import successful")
        
        from src.knowledge.knowledge_base import MedicalKnowledgeBase
        print("  ✓ Knowledge base import successful")
        
        print("✅ All imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        print("🧪 Testing basic functionality...")
        
        # Test state creation
        state = {
            "user_symptoms": "test symptoms",
            "patient_info": {},
            "symptom_analysis": {},
            "differential_diagnosis": {},
            "questions_asked": {},
            "user_answers": [],
            "final_diagnosis": {},
            "medications": [],
            "confidence_score": 0.0,
            "knowledge_sources": [],
            "current_step": "symptom_analysis"
        }
        print("  ✓ State creation successful")
        
        # Test knowledge base creation
        from src.knowledge.knowledge_base import MedicalKnowledgeBase
        kb = MedicalKnowledgeBase()
        print("  ✓ Knowledge base creation successful")
        
        # Test LLM client creation
        from src.llm.together_client import TogetherMedicalLLM
        llm = TogetherMedicalLLM()
        print("  ✓ LLM client creation successful")
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🏥 Medical Diagnosis AI - Quick Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        return False
        
    print()
    
    # Test functionality  
    if not test_basic_functionality():
        return False
    
    print("\n🎉 All tests passed! The system appears to be working.")
    print("\n📋 Next steps:")
    print("  1. Add medical textbooks to data/medical_textbooks/")
    print("  2. Run: python setup.py")
    print("  3. Generate embeddings if you have textbooks")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

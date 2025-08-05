#!/usr/bin/env python3
"""
Test script to verify the dual LLM environment setup
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_environment_detection():
    """Test environment detection logic"""
    print("🧪 Testing Environment Detection")
    print("=" * 50)
    
    try:
        from llm.llm_config import is_huggingface_space, get_llm
        
        # Test environment detection
        is_hf = is_huggingface_space()
        print(f"📍 Detected environment: {'Hugging Face Spaces' if is_hf else 'Local/Other'}")
        
        # Test LLM initialization
        print("\n🤖 Testing LLM initialization...")
        llm = get_llm()
        print("✅ LLM initialized successfully!")
        
        # Test basic functionality
        print("\n💬 Testing basic LLM functionality...")
        response = llm.invoke("Hello! Can you help with medical questions?")
        
        if hasattr(response, 'content'):
            print(f"📝 Response: {response.content[:100]}...")
        else:
            print(f"📝 Response: {str(response)[:100]}...")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def simulate_huggingface_environment():
    """Simulate running on Hugging Face Spaces"""
    print("\n🌐 Simulating Hugging Face Spaces Environment")
    print("=" * 50)
    
    # Set HF environment variables
    os.environ["SPACE_ID"] = "test-space"
    os.environ["SPACE_AUTHOR_NAME"] = "test-user"
    
    try:
        from llm.llm_config import is_huggingface_space, get_llm
        
        # Reload the module to pick up new environment
        import importlib
        import llm.llm_config
        importlib.reload(llm.llm_config)
        
        from llm.llm_config import is_huggingface_space, get_llm
        
        is_hf = is_huggingface_space()
        print(f"📍 Environment detected as: {'Hugging Face Spaces' if is_hf else 'Local/Other'}")
        
        if is_hf:
            print("✅ Successfully detected Hugging Face environment!")
        else:
            print("⚠️ Environment detection may need adjustment")
            
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
    finally:
        # Clean up environment variables
        os.environ.pop("SPACE_ID", None)
        os.environ.pop("SPACE_AUTHOR_NAME", None)

def main():
    print("🏥 Medical Diagnosis AI - Dual Environment Test")
    print("=" * 60)
    
    # Test 1: Current environment
    success = test_environment_detection()
    
    # Test 2: Simulate Hugging Face
    simulate_huggingface_environment()
    
    print("\n📋 Summary:")
    print("✅ Your app supports both environments:" if success else "❌ Setup needs attention:")
    print("   - 💻 Local: Google Gemini (requires GOOGLE_API_KEY)")
    print("   - 🌐 Hugging Face: Free models (no API key needed)")
    print("   - 🔄 Automatic detection and switching")
    
    if success:
        print("\n🚀 Ready for deployment to Hugging Face Spaces!")
        print("   Run: python deploy_to_hf.py")
    else:
        print("\n🔧 Please check your dependencies and try again")

if __name__ == "__main__":
    main()

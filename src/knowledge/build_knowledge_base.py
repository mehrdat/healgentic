
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from knowledge.knowledge_base import MedicalKnowledgeBase

def main():
    """Main function to initialize the knowledge base"""
    
    try:
        print("🏥 Initializing Medical Knowledge Base...")
        
        # Create an instance of the knowledge base
        
        knowledge_base = MedicalKnowledgeBase()
        stats = knowledge_base.get_statistics()

        print(f"📊 Found {stats['available_textbooks']} textbook files")
        
        if stats['available_textbooks'] == 0:
            print("⚠️ No textbooks found. Please ensure the knowledge directory is set correctly.")
            return
        # Process medical textbooks to build the knowledge base
        chunks_processed = knowledge_base.process_medical_textbooks()
        if chunks_processed > 0:
            print(f"✅ Successfully created knowledge base with {chunks_processed} chunks!")
            print("💾 Vector database saved to data/vector_store/")
        else:
            print("❌ Failed to create knowledge base")

        print(f"✅ Knowledge base initialized with {chunks_processed} chunks")
    except Exception as e:
        print(f"❌ Error initializing knowledge base: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()
"""
Main entry point for the Medical Diagnosis AI System
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from workflow.graph import MedicalDiagnosisWorkflow
from knowledge.knowledge_base import MedicalKnowledgeBase


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Medical Diagnosis AI System")
    parser.add_argument("--symptoms", "-s", type=str, help="Symptoms to analyze")
    parser.add_argument("--init-kb", action="store_true", help="Initialize knowledge base")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    parser.add_argument("--status", action="store_true", help="Show system status")
    
    args = parser.parse_args()
    
    print("🏥 Medical Diagnosis AI System")
    print("=" * 50)
    print("⚠️  DISCLAIMER: For educational purposes only!")
    print("   Never use for actual medical diagnosis without proper medical supervision.")
    print()
    
    # Initialize knowledge base
    try:
        print("🔧 Initializing knowledge base...")
        knowledge_base = MedicalKnowledgeBase()
        
        if args.init_kb:
            print("📚 Processing medical textbooks...")
            chunks_processed = knowledge_base.process_medical_textbooks()
            print(f"✅ Knowledge base initialized with {chunks_processed} chunks")
            return
        
        # Initialize workflow
        print("🤖 Initializing diagnosis workflow...")
        workflow = MedicalDiagnosisWorkflow(knowledge_base)
        
        if args.status:
            print("📊 System Status:")
            status = workflow.get_workflow_status()
            for key, value in status.items():
                print(f"  {key}: {value}")
            return
        
        # Run diagnosis
        if args.symptoms:
            print(f"🩺 Analyzing symptoms: {args.symptoms}")
            result = workflow.run_diagnosis(args.symptoms)
            
            print("\n" + "="*50)
            print("📋 DIAGNOSIS RESULTS")
            print("="*50)
            
            final_diagnosis = result.get("final_diagnosis", {})
            print(f"🎯 Diagnosis: {final_diagnosis.get('condition', 'Unknown')}")
            print(f"📊 Confidence: {result.get('confidence_score', 0):.2%}")
            
            # Show knowledge sources
            sources = result.get("knowledge_sources", [])
            if sources:
                print(f"📚 Knowledge Sources: {', '.join(sources[:3])}")
            
            # Show medications if available
            medications = result.get("medications", {})
            if medications and medications.get("structured_medications"):
                print("\n💊 MEDICATION RECOMMENDATIONS:")
                for med in medications["structured_medications"][:3]:
                    print(f"  • {med.get('name', 'Unknown')}")
                    print(f"    Dosage: {med.get('dosage', 'As prescribed')}")
            
            # Show disclaimers
            if medications and medications.get("important_disclaimers"):
                print("\n⚠️  IMPORTANT DISCLAIMERS:")
                for disclaimer in medications["important_disclaimers"][:3]:
                    print(f"  {disclaimer}")
        
        else:
            print("💡 Usage examples:")
            print("  python main.py --symptoms \"headache, fever, nausea\"")
            print("  python main.py --init-kb")
            print("  python main.py --status")
            print("\n🌐 Web Interfaces:")
            print("  streamlit run web_apps/streamlit_app.py")
            print("  python web_apps/gradio_app.py")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

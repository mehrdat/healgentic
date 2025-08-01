"""
Main entry point for the Medical Diagnosis AI System
"""

import sys
import os
from pathlib import Path

# Disable LangSmith warnings and tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

# Add src to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent))

# Import warning suppression first
from utils.disable_warnings import suppress_warnings

from knowledge.knowledge_base import MedicalKnowledgeBase
from workflow.graph import MedicalDiagnosisWorkflow


class MedicalDiagnosisSystem:
    """Main Medical Diagnosis System"""
    
    def __init__(self):
        print("🏥 Initializing Medical Diagnosis System...")
        
        # Initialize knowledge base
        self.knowledge_base = MedicalKnowledgeBase()
        
        # Initialize workflow
        self.workflow = MedicalDiagnosisWorkflow(self.knowledge_base)
        
        print("✅ Medical Diagnosis System initialized")
    
    def initialize_knowledge_base(self):
        """Initialize or rebuild the knowledge base"""
        print("📚 Initializing knowledge base...")
        chunks_processed = self.knowledge_base.process_medical_textbooks()
        print(f"✅ Knowledge base initialized with {chunks_processed} chunks")
        return chunks_processed
    
    def diagnose(self, symptoms: str, patient_info: dict = None) -> dict:
        """Run diagnosis on symptoms"""
        return self.workflow.run_diagnosis(symptoms, patient_info)
    
    def get_system_status(self) -> dict:
        """Get system status"""
        return {
            "knowledge_base": self.knowledge_base.get_statistics(),
        }


def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py --init-kb                 # Initialize knowledge base")
        print("  python main.py --symptoms 'your symptoms'  # Run diagnosis")
        print("  python main.py --status                   # Check system status")
        return
    
    command = sys.argv[1]
    
    if command == "--init-kb":
        system = MedicalDiagnosisSystem()
        system.initialize_knowledge_base()
        
    elif command == "--symptoms":
        if len(sys.argv) < 3:
            print("Please provide symptoms: python main.py --symptoms 'your symptoms'")
            return
        
        symptoms = sys.argv[2]
        system = MedicalDiagnosisSystem()
        
        print(f"🔍 Analyzing symptoms: {symptoms}")
        result = system.diagnose(symptoms)
        
        # Display results
        print("\n" + "="*50)
        print("📋 DIAGNOSIS RESULTS")
        print("="*50)
        
        final_diagnosis = result.get("final_diagnosis", {})
        if final_diagnosis:
            print(f"🎯 Diagnosis: {final_diagnosis.get('condition', 'Unknown')}")
            print(f"🎯 Confidence: {result.get('confidence_score', 0):.1%}")
        
        print(f"📚 Sources: {', '.join(result.get('knowledge_sources', [])[:3])}")
        
    elif command == "--status":
        system = MedicalDiagnosisSystem()
        status = system.get_system_status()
        
        print("🔍 System Status\n")
        print("="*30)
        print(f"Knowledge Base: {status['knowledge_base']}\n")
        print(f"Workflow: {status['workflow']}\n")

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()

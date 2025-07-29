#!/usr/bin/env python3
"""
Quick script to create embeddings directly
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "medical_diagnosis_ai/src"))

from src.knowledge.knowledge_base import MedicalKnowledgeBase

# Create and run
print("🚀 Creating medical knowledge embeddings...")
kb = MedicalKnowledgeBase()

# Show what files will be processed
stats = kb.get_statistics()
print(f"📚 Found {stats['available_textbooks']} textbook files")

# Process and create embeddings
total_chunks = kb.process_medical_textbooks()
print(f"✅ Created embeddings for {total_chunks} chunks!")
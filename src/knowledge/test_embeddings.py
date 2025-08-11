#!/usr/bin/env python3
"""
Test script to demonstrate switching between embedding services
"""

import os
import sys
from pathlib import Path

# Add the src/knowledge directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from medical_diagnosis_ai.src.knowledge.txt_maker import get_embedding_service

def test_embedding_services():
    """Test different embedding services"""
    
    # Test data
    test_texts = [
        "The patient presents with acute chest pain and shortness of breath.",
        "Hypertension is a common cardiovascular condition affecting millions.",
        "Diabetes mellitus requires careful blood glucose monitoring.",
        "Pneumonia symptoms include fever, cough, and difficulty breathing."
    ]
    
    services_to_test = [
        "sentence_transformers",
        "together_ai", 
        "voyage_ai",
        "simple_hash"
    ]
    
    print("🧪 Testing Embedding Services")
    print("=" * 50)
    
    for service_name in services_to_test:
        print(f"\n📡 Testing: {service_name}")
        print("-" * 30)
        
        try:
            # Initialize service
            service = get_embedding_service(service_name)
            print(f"✅ Service initialized: {service.service_name}")
            print(f"📐 Dimension: {service.get_dimension()}")
            
            # Create embeddings
            embeddings = service.encode(test_texts, show_progress=False)
            print(f"📊 Embeddings shape: {embeddings.shape}")
            print(f"📈 Sample embedding (first 5 values): {embeddings[0][:5]}")
            
            # Test single text encoding
            single_embedding = service.encode([test_texts[0]], show_progress=False)
            print(f"🔍 Single embedding shape: {single_embedding.shape}")
            
        except Exception as e:
            print(f"❌ Error with {service_name}: {e}")
    
    print(f"\n🎯 Recommendation:")
    print(f"   - For best quality: Use 'sentence_transformers'")
    print(f"   - For Together AI integration: Use 'together_ai'")
    print(f"   - For testing/fallback: Use 'simple_hash'")

def switch_embedding_service(new_service: str):
    """Switch the embedding service in .env file"""
    
    env_file = Path(__file__).parent.parent / ".env"
    
    if not env_file.exists():
        print(f"❌ .env file not found: {env_file}")
        return
    
    # Read current .env content
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    # Update EMBEDDING_SERVICE line
    updated_lines = []
    service_updated = False
    
    for line in lines:
        if line.startswith('EMBEDDING_SERVICE='):
            updated_lines.append(f'EMBEDDING_SERVICE={new_service}\n')
            service_updated = True
        else:
            updated_lines.append(line)
    
    # If EMBEDDING_SERVICE wasn't found, add it
    if not service_updated:
        updated_lines.append(f'\nEMBEDDING_SERVICE={new_service}\n')
    
    # Write back to .env file
    with open(env_file, 'w') as f:
        f.writelines(updated_lines)
    
    print(f"✅ Updated .env file: EMBEDDING_SERVICE={new_service}")
    print(f"🔄 Restart the embedding script to use the new service")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test and switch embedding services")
    parser.add_argument("--test", action="store_true", help="Test all embedding services")
    parser.add_argument("--switch", type=str, choices=["sentence_transformers", "together_ai", "voyage_ai", "simple_hash"], 
                       help="Switch to a different embedding service")
    
    args = parser.parse_args()
    
    if args.test:
        test_embedding_services()
    elif args.switch:
        switch_embedding_service(args.switch)
    else:
        print("Usage:")
        print("  python test_embeddings.py --test                    # Test all services")
        print("  python test_embeddings.py --switch voyage_ai        # Switch to Voyage AI")
        print("  python test_embeddings.py --switch together_ai      # Switch to Together AI")
        print("  python test_embeddings.py --switch sentence_transformers  # Switch to SentenceTransformers")
        print("  python test_embeddings.py --switch simple_hash      # Switch to simple hash")

#!/usr/bin/env python3
"""
Setup and Installation Script for Medical Diagnosis AI System
"""

import os
import sys
import subprocess
from pathlib import Path


def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, check=True)
        
        print("✅ Requirements installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "data/medical_textbooks",
        "data/vector_store", 
        "logs",
        "web_apps/static",
        "web_apps/templates"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("✅ Directories created")


def check_environment():
    """Check environment setup"""
    print("🔍 Checking environment...")
    
    # Check .env file
    if not os.path.exists(".env"):
        print("❌ .env file not found")
        return False
    
    # Check Together API key
    from dotenv import load_dotenv
    load_dotenv()
    
    together_key = os.getenv("TOGETHER_API_KEY")
    if not together_key:
        print("❌ TOGETHER_API_KEY not found in .env")
        return False
    
    print("✅ Environment configuration looks good")
    return True


def check_medical_textbooks():
    """Check for medical textbooks"""
    print("📚 Checking for medical textbooks...")
    
    textbook_dir = Path("data/medical_textbooks")
    
    if not textbook_dir.exists():
        print("❌ Medical textbooks directory not found")
        return False
    
    pdf_files = list(textbook_dir.glob("*.pdf"))
    epub_files = list(textbook_dir.glob("*.epub"))
    total_files = len(pdf_files) + len(epub_files)
    
    if total_files == 0:
        print("⚠️  No medical textbooks found")
        print("Please add PDF or EPUB medical textbooks to data/medical_textbooks/")
        return False
    
    print(f"✅ Found {total_files} textbook(s)")
    for file in (pdf_files + epub_files)[:5]:  # Show first 5
        print(f"  • {file.name}")
    if total_files > 5:
        print(f"  ... and {total_files - 5} more")
    
    return True


def test_system():
    """Test the system components"""
    print("🧪 Testing system components...")
    
    try:
        # Test imports
        sys.path.insert(0, "src")
        from src.main import MedicalDiagnosisSystem
        
        print("  ✓ Imports successful")
        
        # Test system initialization
        system = MedicalDiagnosisSystem()
        print("  ✓ System initialization successful")
        
        # Test status
        status = system.get_system_status()
        print("  ✓ System status check successful")
        
        print("✅ All tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def run_embeddings():
    """Run the embedding script"""
    print("🔄 Running embedding generation...")
    
    try:
        # Change to knowledge directory
        os.chdir("src/knowledge")
        result = subprocess.run([
            sys.executable, "run_emb.py"
        ], capture_output=True, text=True)
        
        # Change back
        os.chdir("../..")
        
        if result.returncode == 0:
            print("✅ Embeddings generated successfully")
            return True
        else:
            print(f"❌ Embedding generation failed")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running embeddings: {e}")
        return False


def main():
    """Main setup function"""
    print("🏥 Medical Diagnosis AI System Setup")
    print("=" * 50)
    
    steps = [
        ("Install Requirements", install_requirements),
        ("Create Directories", create_directories),
        ("Check Environment", check_environment),
        ("Check Medical Textbooks", check_medical_textbooks),
        ("Test System", test_system)
    ]
    
    all_passed = True
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            all_passed = False
            break
    
    if all_passed:
        print("\n🎉 Setup completed successfully!")
        
        # Ask about embeddings
        if check_medical_textbooks():
            response = input("\n🤖 Generate embeddings now? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                run_embeddings()
        
        print("\n📱 You can now run:")
        print("  • python src/main.py --status")
        print("  • python src/main.py --symptoms 'your symptoms'")
        print("  • streamlit run web_apps/streamlit_app.py")
        print("  • python web_apps/gradio_app.py")
        
    else:
        print("\n❌ Setup incomplete. Please fix the issues above.")


if __name__ == "__main__":
    main()

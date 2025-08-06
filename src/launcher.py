#!/usr/bin/env python3
"""
Launcher script for Medical Diagnosis AI System

Choose between Streamlit or Gradio interface
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("🏥 Medical Diagnosis AI System")
    print("=" * 40)
    print("Choose your interface:")
    print("1. Streamlit (original)")
    print("2. Gradio (new)")
    print("3. Exit")
    print()
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting Streamlit interface...")
            try:
                subprocess.run([
                    sys.executable, "-m", "streamlit", "run", 
                    "app_interactive.py"
                ], check=True)
            except KeyboardInterrupt:
                print("\n👋 Streamlit stopped by user")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Error running Streamlit: {e}")
            break
            
        elif choice == "2":
            print("\n🚀 Starting Gradio interface...")
            try:
                subprocess.run([
                    sys.executable, 
                    "app_gradio.py"
                ], check=True)
            except KeyboardInterrupt:
                print("\n👋 Gradio stopped by user")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Error running Gradio: {e}")
            break
            
        elif choice == "3":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()

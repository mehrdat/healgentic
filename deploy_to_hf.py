#!/usr/bin/env python3
"""
Deployment script for Hugging Face Spaces

This script help    print(f"\n🎯 Ready to deploy to: {hf_repo_url}")
    print("\n📋 Next steps:")
    print("1. Make sure you have a Hugging Face account")
    print("2. Create a new Space on Hugging Face with the same name")
    print("3. Set up your Hugging Face token:")
    print("   - Go to https://huggingface.co/settings/tokens")
    print("   - Create a new token with 'write' permissions")
    print("   - Run: huggingface-cli login")
    print("4. 🎉 NO API KEY NEEDED! The app will automatically use Hugging Face's free models")
    print("5. Push to deploy:")
    print("   git push origin main")
    print("\n💡 Smart LLM Selection:")
    print("   - On Hugging Face Spaces: Uses free Hugging Face models")
    print("   - On your laptop: Uses Google Gemini (requires GOOGLE_API_KEY)")
    print("   - Automatically detects the environment!")your Medical Diagnosis AI to Hugging Face Spaces
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def main():
    print("🚀 Hugging Face Spaces Deployment Script")
    print("=" * 50)
    
    # Get user input
    space_name = input("Enter your Hugging Face space name (username/space-name): ").strip()
    if not space_name:
        print("❌ Space name is required!")
        return
    
    # Check if git is installed
    if not run_command("git --version", "Checking Git installation"):
        print("❌ Git is not installed. Please install Git first.")
        return
    
    # Check if huggingface_hub is installed
    try:
        import huggingface_hub
        print("✅ huggingface_hub is available")
    except ImportError:
        print("📦 Installing huggingface_hub...")
        if not run_command("pip install huggingface_hub", "Installing huggingface_hub"):
            return
    
    # Prepare files for deployment
    print("\n📁 Preparing files for deployment...")
    
    # Copy requirements (already in the right place)
    if Path("requirements.txt").exists():
        print("✅ Requirements file ready for deployment")
    
    # Copy README (already in the right place)
    if Path("README.md").exists():
        print("✅ README file ready for deployment")
    
    # Initialize git repository if not exists
    if not Path(".git").exists():
        run_command("git init", "Initializing Git repository")
    
    # Add files to git
    files_to_add = [
        "app.py",
        "requirements.txt", 
        "README.md",
        "src/",
        "knowledge/",
        "config/"
    ]
    
    for file in files_to_add:
        if Path(file).exists():
            run_command(f"git add {file}", f"Adding {file} to git")
    
    # Commit changes
    run_command('git commit -m "Deploy Medical Diagnosis AI to Hugging Face Spaces"', "Committing changes")
    
    # Add Hugging Face remote
    hf_repo_url = f"https://huggingface.co/spaces/{space_name}"
    run_command(f"git remote remove origin", "Removing existing origin (if any)")  # This might fail, that's OK
    run_command(f"git remote add origin {hf_repo_url}", "Adding Hugging Face remote")
    
    print(f"\n🎯 Ready to deploy to: {hf_repo_url}")
    print("\n📋 Next steps:")
    print("1. Make sure you have a Hugging Face account")
    print("2. Create a new Space on Hugging Face with the same name")
    print("3. Set up your Hugging Face token:")
    print("   - Go to https://huggingface.co/settings/tokens")
    print("   - Create a new token with 'write' permissions")
    print("   - Run: huggingface-cli login")
    print("4. Set up your Google API key as a Space secret:")
    print("   - Go to your Space settings")
    print("   - Add GOOGLE_API_KEY as a secret")
    print("5. Push to deploy:")
    print(f"   git push origin main")
    
    # Ask if user wants to push now
    push_now = input("\n❓ Do you want to push now? (y/N): ").strip().lower()
    if push_now in ['y', 'yes']:
        if run_command("git push origin main", "Pushing to Hugging Face Spaces"):
            print("\n🎉 Deployment successful!")
            print(f"🌐 Your app will be available at: {hf_repo_url}")
            print("\n⏰ Note: It may take a few minutes for your app to build and deploy.")
        else:
            print("\n❌ Deployment failed. Please check the error messages above.")
            print("💡 You might need to:")
            print("   - Login to Hugging Face: huggingface-cli login")
            print("   - Create the Space on Hugging Face website first")
            print("   - Set up your GOOGLE_API_KEY secret in Space settings")
    else:
        print("\n📝 To deploy later, run: git push origin main")

if __name__ == "__main__":
    main()

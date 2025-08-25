#!/usr/bin/env python3
"""
Setup script for the RAG Pipeline
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command("python3 -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command("python3 -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def test_installation():
    """Test if the installation was successful"""
    print("\n🧪 Testing installation...")
    
    try:
        import fitz
        print("✅ PyMuPDF imported successfully")
    except ImportError:
        print("❌ PyMuPDF import failed")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers imported successfully")
    except ImportError:
        print("❌ sentence-transformers import failed")
        return False
    
    try:
        import faiss
        print("✅ FAISS imported successfully")
    except ImportError:
        print("❌ FAISS import failed")
        return False
    
    try:
        from fastapi import FastAPI
        print("✅ FastAPI imported successfully")
    except ImportError:
        print("❌ FastAPI import failed")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up arXiv cs.CL RAG Pipeline")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python test_pipeline.py' to test the pipeline")
    print("2. Run 'python notebook_demo.py' to see the demo")
    print("3. Run 'python main.py' to start the FastAPI service")
    print("4. Run 'python generate_report.py' to generate a comprehensive report")

if __name__ == "__main__":
    main()

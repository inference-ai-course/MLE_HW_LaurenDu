#!/usr/bin/env python3
"""
Complete RAG Pipeline Runner

This script runs the entire RAG pipeline from PDF processing to index creation.
Run this script to set up the complete system.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, description):
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        print(f"✅ Completed in {elapsed:.1f}s")
        
        if result.stdout:
            print("Output:")
            print(result.stdout)
            
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ Failed after {elapsed:.1f}s")
        print(f"Error: {e.stderr}")
        return False

def check_prerequisites():
    print("🔍 Checking prerequisites...")
    
    # Check if PDFs directory exists and has files
    pdf_dir = Path("PDFs")
    if not pdf_dir.exists():
        print("❌ PDFs directory not found!")
        return False
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if len(pdf_files) < 50:
        print(f"⚠️  Warning: Only {len(pdf_files)} PDF files found (expected 50)")
    else:
        print(f"✅ Found {len(pdf_files)} PDF files")
    
    # Check Python modules
    required_modules = [
        "fitz",  # PyMuPDF
        "sentence_transformers",
        "faiss",
        "fastapi",
        "tiktoken",
        "numpy",
        "pandas"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing modules: {', '.join(missing_modules)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All prerequisites satisfied")
    return True

def main():
    print("🚀 RAG Pipeline Runner")
    print("Building complete RAG system for arXiv cs.CL papers")
    
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Pipeline steps
    steps = [
        ("cd src && python pdf_processor.py", "Processing PDF files and extracting text"),
        ("cd src && python chunker.py", "Creating text chunks with token counting"),
        ("cd src && python embedder.py", "Generating embeddings and building FAISS index"),
        ("cd src && python generate_report.py", "Generating retrieval performance report")
    ]
    
    print(f"\n📋 Pipeline has {len(steps)} steps:")
    for i, (_, desc) in enumerate(steps, 1):
        print(f"  {i}. {desc}")
    
    print(f"\n⏱️  Estimated time: 5-15 minutes (depending on hardware)")
    
    input("\nPress Enter to start the pipeline...")
    
    start_time = time.time()
    
    for i, (command, description) in enumerate(steps, 1):
        print(f"\n📍 Step {i}/{len(steps)}")
        
        if not run_command(command, description):
            print(f"\n❌ Pipeline failed at step {i}")
            print("Please check the error messages above and try again.")
            sys.exit(1)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"🎉 RAG Pipeline Completed Successfully!")
    print(f"{'='*60}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    
    # Check generated files
    expected_files = [
        "data/processed_documents.json",
        "data/chunks.json", 
        "data/faiss_index.bin",
        "data/chunk_metadata.pkl",
        "data/embeddings.npy",
        "outputs/retrieval_report.md"
    ]
    
    print(f"\n📁 Generated files:")
    for file_path in expected_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024 / 1024  # MB
            print(f"  ✅ {file_path} ({size:.1f} MB)")
        else:
            print(f"  ❌ {file_path} (missing)")
    
    print(f"\n🎯 Next steps:")
    print(f"  1. Run the API server: python main.py")
    print(f"  2. Try the notebook: jupyter notebook rag_demo.ipynb")
    print(f"  3. Check the report: outputs/retrieval_report.md")
    print(f"  4. Test API: curl 'http://localhost:8000/search?q=transformer&k=3'")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Simple test script for hybrid search functionality
"""

import os
import sys
import logging

# Add current directory to path
sys.path.append('.')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database():
    """Test database functionality"""
    try:
        from database import DocumentDatabase
        
        print("Testing database setup...")
        db = DocumentDatabase("test_db.db")
        
        # Test if we can load metadata
        if os.path.exists("metadata.json"):
            db.load_metadata_from_json()
            print("✅ Metadata loaded successfully")
        
        if os.path.exists("processed_chunks.json"):
            db.load_chunks_from_json()
            print("✅ Chunks loaded successfully")
            
            # Test FTS search
            results = db.search_fts("machine learning", 3)
            print(f"✅ FTS search returned {len(results)} results")
            
        stats = db.get_document_stats()
        print(f"✅ Database stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without full RAG pipeline"""
    
    print("="*50)
    print("HYBRID SEARCH FUNCTIONALITY TEST")
    print("="*50)
    
    # Test database
    if not test_database():
        print("Database test failed, skipping other tests")
        return
    
    print("✅ All basic tests passed!")
    print("\nTo run full evaluation:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run: python run_evaluation.py")
    print("3. Or start the API: python main.py")

def show_usage_examples():
    """Show usage examples"""
    print("\n" + "="*50)
    print("USAGE EXAMPLES")
    print("="*50)
    
    print("\n1. Start the API server:")
    print("   python main.py")
    
    print("\n2. Test hybrid search via API:")
    print("   curl -X POST http://localhost:8000/hybrid_search \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"query\": \"machine learning\", \"k\": 3, \"alpha\": 0.5}'")
    
    print("\n3. Run evaluation:")
    print("   python run_evaluation.py")
    
    print("\n4. Available endpoints:")
    print("   GET  /health - Health check")
    print("   POST /search - Vector search only")  
    print("   POST /hybrid_search - Hybrid vector + keyword search")
    print("   GET  /hybrid_search?query=...&k=3&alpha=0.5 - GET version")

if __name__ == "__main__":
    test_basic_functionality()
    show_usage_examples()
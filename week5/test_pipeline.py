#!/usr/bin/env python3
"""
Test script for the RAG pipeline
"""

import sys
import os
from rag_pipeline import RAGPipeline

def test_pipeline():
    """Test the complete RAG pipeline"""
    
    print("🧪 Testing RAG Pipeline")
    print("=" * 40)
    
    try:
        # Initialize pipeline
        print("1. Initializing pipeline...")
        pipeline = RAGPipeline()
        print("✅ Pipeline initialized")
        
        # Run pipeline
        print("2. Running pipeline...")
        pipeline.run_pipeline()
        print("✅ Pipeline completed")
        
        # Test search functionality
        print("3. Testing search functionality...")
        test_queries = [
            "machine learning",
            "natural language processing",
            "neural networks"
        ]
        
        for query in test_queries:
            results = pipeline.search(query, k=1)
            print(f"   Query: '{query}' -> Top result: {results[0]['metadata']['pdf_file']}")
        
        print("✅ Search functionality working")
        
        # Check generated files
        print("4. Checking generated files...")
        required_files = [
            'processed_chunks.json',
            'embeddings.npy',
            'faiss_index.idx'
        ]
        
        for file in required_files:
            if os.path.exists(file):
                print(f"   ✅ {file} exists")
            else:
                print(f"   ❌ {file} missing")
                return False
        
        print("✅ All files generated successfully")
        
        # Print statistics
        print("5. Pipeline statistics:")
        print(f"   - Total chunks: {len(pipeline.chunks)}")
        print(f"   - Total papers: {len(set(m['pdf_file'] for m in pipeline.chunk_metadata))}")
        print(f"   - Embedding dimension: {pipeline.embedding_dim}")
        
        print("\n🎉 All tests passed! Pipeline is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)


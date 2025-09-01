#!/usr/bin/env python3
"""
RAG Pipeline Demo Script

This script provides the same functionality as the Jupyter notebook
but can be run directly from the command line.
"""

import sys
import json
import matplotlib.pyplot as plt
from rag_pipeline import RAGPipeline

def main():
    """Main demo function"""
    
    print("🔍 RAG Pipeline Demo for arXiv cs.CL Papers")
    print("=" * 60)
    
    # Initialize pipeline
    print("1. Initializing RAG pipeline...")
    pipeline = RAGPipeline()
    pipeline.run_pipeline()
    
    print(f"✅ Pipeline initialized with {len(pipeline.chunks)} chunks from {len(set(m['pdf_file'] for m in pipeline.chunk_metadata))} papers")
    
    # Explore the data
    print("\n2. Data Exploration")
    print("-" * 20)
    
    # Calculate statistics
    paper_stats = {}
    for metadata in pipeline.chunk_metadata:
        paper_id = metadata['pdf_file']
        if paper_id not in paper_stats:
            paper_stats[paper_id] = 0
        paper_stats[paper_id] += 1
    
    chunk_lengths = [len(chunk) for chunk in pipeline.chunks]
    
    print(f"Total papers processed: {len(paper_stats)}")
    print(f"Total chunks created: {len(pipeline.chunks)}")
    print(f"Average chunks per paper: {len(pipeline.chunks) / len(paper_stats):.1f}")
    print(f"Embedding dimension: {pipeline.embedding_dim}")
    print(f"Chunk length - Min: {min(chunk_lengths)}, Max: {max(chunk_lengths)}, Mean: {sum(chunk_lengths) / len(chunk_lengths):.1f}")
    
    # Show papers with most chunks
    sorted_papers = sorted(paper_stats.items(), key=lambda x: x[1], reverse=True)
    print(f"\nPapers with most chunks:")
    for paper, count in sorted_papers[:5]:
        print(f"  {paper}: {count} chunks")
    
    # Interactive search demo
    print("\n3. Interactive Search Demo")
    print("-" * 30)
    
    test_queries = [
        "machine learning",
        "natural language processing", 
        "neural networks",
        "transformer models",
        "deep learning",
        "language models",
        "attention mechanism",
        "semantic analysis",
        "text classification",
        "information retrieval"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 40)
        
        results = pipeline.search(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"Rank {i} (Distance: {result['distance']:.4f})")
            print(f"Paper: {result['metadata']['pdf_file']}")
            print(f"Chunk ID: {result['metadata']['chunk_id']}")
            print(f"Text: {result['chunk'][:200]}...")
            print()
    
    # Performance analysis
    print("\n4. Performance Analysis")
    print("-" * 25)
    
    query_categories = {
        'ML/AI': ['machine learning', 'artificial intelligence', 'neural networks', 'deep learning'],
        'NLP': ['natural language processing', 'language models', 'text analysis', 'semantic analysis'],
        'Technical': ['transformer models', 'attention mechanism', 'embedding', 'vector space'],
        'Applications': ['text classification', 'information retrieval', 'sentiment analysis', 'machine translation']
    }
    
    performance_data = []
    
    for category, queries in query_categories.items():
        print(f"\nCategory: {category}")
        for query in queries:
            results = pipeline.search(query, k=3)
            avg_distance = sum(r['distance'] for r in results) / len(results)
            performance_data.append({
                'category': category,
                'query': query,
                'avg_distance': avg_distance
            })
            print(f"  {query}: avg distance {avg_distance:.4f}")
    
    # Paper retrieval analysis
    print("\n5. Paper Retrieval Analysis")
    print("-" * 30)
    
    paper_retrieval_counts = {}
    
    for category, queries in query_categories.items():
        for query in queries:
            results = pipeline.search(query, k=3)
            for result in results:
                paper = result['metadata']['pdf_file']
                if paper not in paper_retrieval_counts:
                    paper_retrieval_counts[paper] = 0
                paper_retrieval_counts[paper] += 1
    
    sorted_retrievals = sorted(paper_retrieval_counts.items(), key=lambda x: x[1], reverse=True)
    print("Most frequently retrieved papers:")
    for paper, count in sorted_retrievals[:10]:
        print(f"  {paper}: {count} retrievals")
    
    # Generate search report
    print("\n6. Generating Search Report")
    print("-" * 30)
    
    report_data = []
    
    for category, queries in query_categories.items():
        for query in queries:
            results = pipeline.search(query, k=3)
            for i, result in enumerate(results, 1):
                report_data.append({
                    'category': category,
                    'query': query,
                    'rank': i,
                    'paper': result['metadata']['pdf_file'],
                    'chunk_id': result['metadata']['chunk_id'],
                    'distance': result['distance'],
                    'chunk_preview': result['chunk'][:200] + '...'
                })
    
    # Save report
    with open('search_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"✅ Search report saved to 'search_report.json'")
    print(f"Generated {len(report_data)} search results across {len([q for queries in query_categories.values() for q in queries])} queries")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python main.py' to start the FastAPI service")
    print("2. Run 'python generate_report.py' to generate a comprehensive report")
    print("3. Open 'rag_demo.ipynb' in Jupyter for interactive exploration")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Retrieval Report Generator for arXiv cs.CL RAG Pipeline

This script generates a comprehensive report showing example queries and their top-3 retrieved passages
to demonstrate the RAG system performance.
"""

import json
import sys
from rag_pipeline import RAGPipeline
from datetime import datetime

def generate_retrieval_report():
    """Generate a comprehensive retrieval report"""
    
    print("🔍 Generating Retrieval Report for arXiv cs.CL RAG Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    try:
        pipeline = RAGPipeline()
        pipeline.run_pipeline()
        print(f"✅ Pipeline loaded with {len(pipeline.chunks)} chunks from {len(set(m['pdf_file'] for m in pipeline.chunk_metadata))} papers")
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        return
    
    # Define test queries covering different aspects of cs.CL
    test_queries = [
        # Core NLP concepts
        "natural language processing",
        "machine learning",
        "deep learning",
        "neural networks",
        "transformer models",
        
        # Specific techniques
        "attention mechanism",
        "language models",
        "text classification",
        "information retrieval",
        "semantic analysis",
        
        # Applications
        "machine translation",
        "sentiment analysis",
        "named entity recognition",
        "part of speech tagging",
        "dependency parsing",
        
        # Technical concepts
        "embedding vectors",
        "vector space models",
        "probabilistic models",
        "statistical methods",
        "evaluation metrics"
    ]
    
    print(f"\n📊 Testing {len(test_queries)} queries...")
    
    # Generate results for each query
    report_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_queries": len(test_queries),
            "total_papers": len(set(m['pdf_file'] for m in pipeline.chunk_metadata)),
            "total_chunks": len(pipeline.chunks),
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": pipeline.embedding_dim
        },
        "queries": []
    }
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}/{len(test_queries)}: '{query}'")
        
        try:
            # Perform search
            results = pipeline.search(query, k=3)
            
            # Format results
            query_results = {
                "query": query,
                "query_id": i,
                "results": []
            }
            
            for j, result in enumerate(results, 1):
                query_results["results"].append({
                    "rank": j,
                    "paper": result['metadata']['pdf_file'],
                    "chunk_id": result['metadata']['chunk_id'],
                    "distance": result['distance'],
                    "chunk_text": result['chunk'],
                    "chunk_preview": result['chunk'][:200] + "..." if len(result['chunk']) > 200 else result['chunk']
                })
            
            report_data["queries"].append(query_results)
            
            # Print summary
            print(f"   📄 Top result: {results[0]['metadata']['pdf_file']} (distance: {results[0]['distance']:.4f})")
            
        except Exception as e:
            print(f"   ❌ Error processing query: {e}")
            continue
    
    # Calculate statistics
    print(f"\n📈 Calculating statistics...")
    
    # Paper retrieval frequency
    paper_frequency = {}
    for query_data in report_data["queries"]:
        for result in query_data["results"]:
            paper = result["paper"]
            if paper not in paper_frequency:
                paper_frequency[paper] = 0
            paper_frequency[paper] += 1
    
    # Distance statistics
    all_distances = []
    for query_data in report_data["queries"]:
        for result in query_data["results"]:
            all_distances.append(result["distance"])
    
    # Add statistics to report
    report_data["statistics"] = {
        "paper_retrieval_frequency": dict(sorted(paper_frequency.items(), key=lambda x: x[1], reverse=True)[:10]),
        "distance_statistics": {
            "min": min(all_distances),
            "max": max(all_distances),
            "mean": sum(all_distances) / len(all_distances),
            "median": sorted(all_distances)[len(all_distances) // 2]
        },
        "total_retrievals": len(all_distances),
        "unique_papers_retrieved": len(paper_frequency)
    }
    
    # Save comprehensive report
    with open('retrieval_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Generate human-readable report
    generate_human_readable_report(report_data)
    
    print(f"\n✅ Retrieval report generated successfully!")
    print(f"📄 JSON report: retrieval_report.json")
    print(f"📄 Human-readable report: retrieval_report.txt")

def generate_human_readable_report(report_data):
    """Generate a human-readable text report"""
    
    with open('retrieval_report.txt', 'w') as f:
        f.write("ARXIV CS.CL RAG PIPELINE - RETRIEVAL REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Metadata
        f.write("METADATA\n")
        f.write("-" * 20 + "\n")
        f.write(f"Generated: {report_data['metadata']['generated_at']}\n")
        f.write(f"Total Queries: {report_data['metadata']['total_queries']}\n")
        f.write(f"Total Papers: {report_data['metadata']['total_papers']}\n")
        f.write(f"Total Chunks: {report_data['metadata']['total_chunks']}\n")
        f.write(f"Embedding Model: {report_data['metadata']['embedding_model']}\n")
        f.write(f"Embedding Dimension: {report_data['metadata']['embedding_dimension']}\n\n")
        
        # Statistics
        f.write("PERFORMANCE STATISTICS\n")
        f.write("-" * 25 + "\n")
        stats = report_data["statistics"]
        f.write(f"Total Retrievals: {stats['total_retrievals']}\n")
        f.write(f"Unique Papers Retrieved: {stats['unique_papers_retrieved']}\n")
        f.write(f"Distance Statistics:\n")
        f.write(f"  - Min: {stats['distance_statistics']['min']:.4f}\n")
        f.write(f"  - Max: {stats['distance_statistics']['max']:.4f}\n")
        f.write(f"  - Mean: {stats['distance_statistics']['mean']:.4f}\n")
        f.write(f"  - Median: {stats['distance_statistics']['median']:.4f}\n\n")
        
        # Most frequently retrieved papers
        f.write("MOST FREQUENTLY RETRIEVED PAPERS\n")
        f.write("-" * 35 + "\n")
        for paper, count in list(stats['paper_retrieval_frequency'].items())[:10]:
            f.write(f"{paper}: {count} retrievals\n")
        f.write("\n")
        
        # Detailed query results
        f.write("DETAILED QUERY RESULTS\n")
        f.write("-" * 25 + "\n\n")
        
        for query_data in report_data["queries"]:
            f.write(f"Query {query_data['query_id']}: '{query_data['query']}'\n")
            f.write("-" * (len(query_data['query']) + 15) + "\n")
            
            for result in query_data["results"]:
                f.write(f"Rank {result['rank']} (Distance: {result['distance']:.4f})\n")
                f.write(f"Paper: {result['paper']}\n")
                f.write(f"Chunk ID: {result['chunk_id']}\n")
                f.write(f"Text: {result['chunk_preview']}\n")
                f.write("\n")
            
            f.write("\n" + "=" * 60 + "\n\n")

def print_sample_results():
    """Print a few sample results to console"""
    
    print("\n" + "=" * 60)
    print("SAMPLE RETRIEVAL RESULTS")
    print("=" * 60)
    
    # Load the report
    try:
        with open('retrieval_report.json', 'r') as f:
            report_data = json.load(f)
    except FileNotFoundError:
        print("❌ Report not found. Run generate_retrieval_report() first.")
        return
    
    # Show first 5 queries
    for query_data in report_data["queries"][:5]:
        print(f"\n🔍 Query: '{query_data['query']}'")
        print("-" * 40)
        
        for result in query_data["results"]:
            print(f"Rank {result['rank']} (Distance: {result['distance']:.4f})")
            print(f"Paper: {result['paper']}")
            print(f"Text: {result['chunk_preview']}")
            print()

if __name__ == "__main__":
    generate_retrieval_report()
    print_sample_results()


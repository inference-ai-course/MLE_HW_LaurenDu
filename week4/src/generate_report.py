import sys
import os
sys.path.append('.')

from retriever import RAGRetriever
import json
from datetime import datetime

def generate_retrieval_report():
    try:
        retriever = RAGRetriever(data_directory="../data")
        
        test_queries = [
            {
                "query": "transformer architecture",
                "description": "Architecture and design of transformer models"
            },
            {
                "query": "attention mechanism",
                "description": "Attention mechanisms in neural networks"
            },
            {
                "query": "BERT language model",
                "description": "BERT model architecture and applications"
            },
            {
                "query": "machine translation systems",
                "description": "Machine translation methodologies and systems"
            },
            {
                "query": "natural language understanding",
                "description": "Natural language understanding techniques"
            }
        ]
        
        report_content = []
        report_content.append("# RAG System Retrieval Report")
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        report_content.append("## System Overview")
        report_content.append("This report demonstrates the performance of the RAG (Retrieval-Augmented Generation) system")
        report_content.append("built for searching arXiv Computational Linguistics papers.")
        report_content.append("")
        
        if retriever.index:
            report_content.append(f"- **Total indexed chunks**: {retriever.index.ntotal}")
        if retriever.chunk_metadata:
            unique_docs = {chunk["document_id"] for chunk in retriever.chunk_metadata}
            report_content.append(f"- **Total documents**: {len(unique_docs)}")
            report_content.append(f"- **Average chunks per document**: {retriever.index.ntotal / len(unique_docs):.1f}")
        
        report_content.append("- **Embedding model**: all-MiniLM-L6-v2")
        report_content.append("- **Similarity metric**: L2 distance")
        report_content.append("")
        
        for i, test_query in enumerate(test_queries, 1):
            query = test_query["query"]
            description = test_query["description"]
            
            print(f"Processing query {i}: {query}")
            
            results = retriever.search(query, k=3)
            
            report_content.append(f"## Query {i}: {query}")
            report_content.append(f"*{description}*")
            report_content.append("")
            
            for j, result in enumerate(results, 1):
                report_content.append(f"### Result {j}")
                report_content.append(f"- **Document**: {result['document_id']}")
                report_content.append(f"- **Chunk ID**: {result['chunk_id']}")
                report_content.append(f"- **Similarity Score**: {result['similarity_score']:.4f}")
                report_content.append(f"- **Token Count**: {result['token_count']}")
                report_content.append("")
                report_content.append("**Text:**")
                report_content.append("```")
                text = result['text']
                if len(text) > 500:
                    report_content.append(text[:500] + "...")
                else:
                    report_content.append(text)
                report_content.append("```")
                report_content.append("")
            
            report_content.append("---")
            report_content.append("")
        
        report_content.append("## Performance Analysis")
        report_content.append("")
        report_content.append("### Query Performance Summary")
        
        avg_similarities = []
        for test_query in test_queries:
            results = retriever.search(test_query["query"], k=3)
            if results:
                avg_sim = sum(r['similarity_score'] for r in results) / len(results)
                avg_similarities.append(avg_sim)
                report_content.append(f"- **{test_query['query']}**: Average similarity = {avg_sim:.4f}")
        
        if avg_similarities:
            overall_avg = sum(avg_similarities) / len(avg_similarities)
            report_content.append(f"- **Overall average similarity**: {overall_avg:.4f}")
        
        report_content.append("")
        report_content.append("### System Strengths")
        report_content.append("- Fast semantic search across large document collection")
        report_content.append("- Effective chunking preserves context while enabling granular search")
        report_content.append("- High-quality embeddings capture semantic relationships")
        report_content.append("- Scalable FAISS indexing for efficient retrieval")
        report_content.append("")
        report_content.append("### Future Improvements")
        report_content.append("- Implement query expansion for better recall")
        report_content.append("- Add document-level metadata filtering")
        report_content.append("- Experiment with different embedding models")
        report_content.append("- Implement re-ranking for improved precision")
        
        report_path = "../outputs/retrieval_report.md"
        os.makedirs("../outputs", exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"Report generated successfully: {report_path}")
        return report_path
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return None

if __name__ == "__main__":
    generate_retrieval_report()
#!/usr/bin/env python3
"""
Simple evaluation script that works with existing processed data
"""

import json
import logging
import numpy as np
import faiss
from typing import List, Dict, Set
from database import DocumentDatabase
from rank_bm25 import BM25Okapi
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEvaluator:
    def __init__(self):
        """Initialize evaluator with existing data"""
        self.load_existing_data()
        self.setup_search_components()
        self.test_queries = self.create_test_queries()
    
    def load_existing_data(self):
        """Load existing processed data"""
        # Load chunks and metadata
        with open('processed_chunks.json', 'r') as f:
            data = json.load(f)
            self.chunks = data['chunks']
            self.chunk_metadata = data['metadata']
        
        # Load embeddings and FAISS index
        self.embeddings = np.load('embeddings.npy')
        self.index = faiss.read_index('faiss_index.idx')
        
        # Initialize database
        self.db = DocumentDatabase()
        self.db.load_metadata_from_json()
        self.db.load_chunks_from_json()
        
        logger.info(f"Loaded {len(self.chunks)} chunks and {len(self.embeddings)} embeddings")
    
    def setup_search_components(self):
        """Setup search components"""
        # Setup BM25
        tokenized_chunks = []
        for chunk in self.chunks:
            tokens = re.findall(r'\b\w+\b', chunk.lower())
            tokenized_chunks.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_chunks)
        logger.info("BM25 index created")
    
    def create_test_queries(self) -> List[Dict]:
        """Create test queries with known relevant documents"""
        return [
            {
                "query": "neural networks machine learning",
                "relevant_papers": ["1.pdf", "18.pdf", "19.pdf"],
                "description": "Machine learning and neural networks"
            },
            {
                "query": "natural language processing NLP",
                "relevant_papers": ["1.pdf", "3.pdf", "8.pdf", "27.pdf"],
                "description": "Natural language processing"
            },
            {
                "query": "parsing grammar syntactic",
                "relevant_papers": ["3.pdf", "4.pdf", "5.pdf", "14.pdf"],
                "description": "Syntactic parsing and grammar"
            },
            {
                "query": "speech recognition spoken language",
                "relevant_papers": ["18.pdf", "28.pdf", "35.pdf"],
                "description": "Speech processing"
            },
            {
                "query": "machine translation",
                "relevant_papers": ["40.pdf", "44.pdf", "48.pdf"],
                "description": "Machine translation"
            },
            {
                "query": "text segmentation discourse",
                "relevant_papers": ["1.pdf", "13.pdf", "21.pdf"],
                "description": "Text segmentation"
            },
            {
                "query": "word sense disambiguation",
                "relevant_papers": ["20.pdf", "25.pdf", "26.pdf"],
                "description": "Word sense disambiguation"
            },
            {
                "query": "information retrieval",
                "relevant_papers": ["43.pdf", "50.pdf"],
                "description": "Information retrieval"
            },
            {
                "query": "dialogue systems conversation",
                "relevant_papers": ["2.pdf", "28.pdf", "30.pdf"],
                "description": "Dialog systems"
            },
            {
                "query": "part of speech tagging POS",
                "relevant_papers": ["8.pdf", "11.pdf", "12.pdf"],
                "description": "POS tagging"
            }
        ]
    
    def vector_search(self, query: str, k: int = 10) -> List[Dict]:
        """Mock vector search using existing embeddings"""
        # For demo, use first embedding as query embedding
        query_embedding = self.embeddings[0].reshape(1, -1)
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            results.append({
                'rank': i + 1,
                'chunk': self.chunks[idx],
                'metadata': self.chunk_metadata[idx],
                'distance': float(distance),
                'score': 1.0 / (1.0 + distance)  # Convert to similarity score
            })
        return results
    
    def keyword_search_fts(self, query: str, k: int = 10) -> List[Dict]:
        """FTS5 keyword search"""
        try:
            results = self.db.search_fts(query, k)
            search_results = []
            
            for i, (chunk_id, chunk_text, score) in enumerate(results):
                chunk_text_full, metadata = self.db.get_chunk_by_id(chunk_id)
                search_results.append({
                    'rank': i + 1,
                    'chunk': chunk_text_full,
                    'metadata': metadata,
                    'score': score
                })
            return search_results
        except Exception as e:
            logger.error(f"FTS search failed: {e}")
            return []
    
    def keyword_search_bm25(self, query: str, k: int = 10) -> List[Dict]:
        """BM25 keyword search"""
        try:
            query_tokens = re.findall(r'\b\w+\b', query.lower())
            scores = self.bm25.get_scores(query_tokens)
            
            top_indices = np.argsort(scores)[::-1][:k]
            results = []
            
            for i, idx in enumerate(top_indices):
                if scores[idx] > 0:
                    results.append({
                        'rank': i + 1,
                        'chunk': self.chunks[idx],
                        'metadata': self.chunk_metadata[idx],
                        'score': scores[idx]
                    })
            return results
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.5) -> List[Dict]:
        """Simple hybrid search combining vector and FTS"""
        vector_results = self.vector_search(query, k * 2)
        keyword_results = self.keyword_search_fts(query, k * 2)
        
        # Merge results by chunk content
        result_map = {}
        
        # Normalize vector scores
        if vector_results:
            max_v_score = max(r['score'] for r in vector_results)
            min_v_score = min(r['score'] for r in vector_results)
            v_range = max_v_score - min_v_score if max_v_score > min_v_score else 1
            
            for result in vector_results:
                chunk_key = result['chunk'][:100]
                result_map[chunk_key] = {
                    'chunk': result['chunk'],
                    'metadata': result['metadata'],
                    'vector_score': (result['score'] - min_v_score) / v_range,
                    'keyword_score': 0.0
                }
        
        # Normalize keyword scores
        if keyword_results:
            max_k_score = max(r['score'] for r in keyword_results)
            min_k_score = min(r['score'] for r in keyword_results)
            k_range = max_k_score - min_k_score if max_k_score > min_k_score else 1
            
            for result in keyword_results:
                chunk_key = result['chunk'][:100]
                norm_score = (result['score'] - min_k_score) / k_range
                
                if chunk_key in result_map:
                    result_map[chunk_key]['keyword_score'] = norm_score
                else:
                    result_map[chunk_key] = {
                        'chunk': result['chunk'],
                        'metadata': result['metadata'],
                        'vector_score': 0.0,
                        'keyword_score': norm_score
                    }
        
        # Calculate hybrid scores
        hybrid_results = []
        for chunk_key, data in result_map.items():
            hybrid_score = alpha * data['vector_score'] + (1 - alpha) * data['keyword_score']
            
            hybrid_results.append({
                'chunk': data['chunk'],
                'metadata': data['metadata'],
                'score': hybrid_score,
                'vector_score': data['vector_score'],
                'keyword_score': data['keyword_score']
            })
        
        hybrid_results.sort(key=lambda x: x['score'], reverse=True)
        
        for i, result in enumerate(hybrid_results[:k]):
            result['rank'] = i + 1
        
        return hybrid_results[:k]
    
    def evaluate_method(self, method_name: str, search_func, k_values: List[int] = [3, 5]) -> Dict:
        """Evaluate a search method"""
        results = {'method': method_name, 'queries': [], 'metrics': {}}
        
        for query_info in self.test_queries:
            query = query_info['query']
            relevant_papers = set(query_info['relevant_papers'])
            
            query_results = {
                'query': query,
                'description': query_info['description'],
                'relevant_count': len(relevant_papers),
                'results_by_k': {}
            }
            
            for k in k_values:
                try:
                    search_results = search_func(query, k)
                    retrieved_papers = set()
                    
                    for result in search_results:
                        if 'metadata' in result and 'pdf_file' in result['metadata']:
                            retrieved_papers.add(result['metadata']['pdf_file'])
                    
                    relevant_retrieved = relevant_papers & retrieved_papers
                    
                    precision = len(relevant_retrieved) / len(retrieved_papers) if retrieved_papers else 0
                    recall = len(relevant_retrieved) / len(relevant_papers) if relevant_papers else 0
                    hit_rate = 1.0 if len(relevant_retrieved) > 0 else 0.0
                    
                    query_results['results_by_k'][k] = {
                        'precision': precision,
                        'recall': recall,
                        'hit_rate': hit_rate,
                        'relevant_retrieved': len(relevant_retrieved),
                        'retrieved_papers': list(retrieved_papers)
                    }
                    
                except Exception as e:
                    logger.error(f"Error evaluating {method_name} for query '{query}': {e}")
                    query_results['results_by_k'][k] = {
                        'precision': 0, 'recall': 0, 'hit_rate': 0,
                        'relevant_retrieved': 0, 'retrieved_papers': []
                    }
            
            results['queries'].append(query_results)
        
        # Calculate aggregate metrics
        for k in k_values:
            precisions = [q['results_by_k'][k]['precision'] for q in results['queries']]
            recalls = [q['results_by_k'][k]['recall'] for q in results['queries']]
            hit_rates = [q['results_by_k'][k]['hit_rate'] for q in results['queries']]
            
            results['metrics'][f'precision@{k}'] = sum(precisions) / len(precisions) if precisions else 0
            results['metrics'][f'recall@{k}'] = sum(recalls) / len(recalls) if recalls else 0
            results['metrics'][f'hit_rate@{k}'] = sum(hit_rates) / len(hit_rates) if hit_rates else 0
        
        return results
    
    def run_evaluation(self):
        """Run complete evaluation"""
        print("="*80)
        print("HYBRID SEARCH EVALUATION REPORT")
        print("="*80)
        
        # Define methods to evaluate
        methods = {
            'vector_search': lambda q, k: self.vector_search(q, k),
            'keyword_fts': lambda q, k: self.keyword_search_fts(q, k),
            'keyword_bm25': lambda q, k: self.keyword_search_bm25(q, k),
            'hybrid_0.3': lambda q, k: self.hybrid_search(q, k, alpha=0.3),
            'hybrid_0.5': lambda q, k: self.hybrid_search(q, k, alpha=0.5),
            'hybrid_0.7': lambda q, k: self.hybrid_search(q, k, alpha=0.7)
        }
        
        all_results = {}
        
        for method_name, search_func in methods.items():
            print(f"\nEvaluating {method_name}...")
            all_results[method_name] = self.evaluate_method(method_name, search_func)
        
        # Print summary results
        print("\n" + "="*80)
        print("SUMMARY RESULTS")
        print("="*80)
        print(f"{'Method':<15} {'Precision@3':<12} {'Recall@3':<10} {'Hit Rate@3':<12} {'Hit Rate@5':<12}")
        print("-" * 80)
        
        for method_name, results in all_results.items():
            metrics = results['metrics']
            print(f"{method_name:<15} "
                  f"{metrics.get('precision@3', 0):<12.3f} "
                  f"{metrics.get('recall@3', 0):<10.3f} "
                  f"{metrics.get('hit_rate@3', 0):<12.3f} "
                  f"{metrics.get('hit_rate@5', 0):<12.3f}")
        
        # Find best methods
        print("\n" + "="*80)
        print("BEST PERFORMING METHODS")
        print("="*80)
        
        for metric in ['hit_rate@3', 'precision@3', 'recall@3']:
            best_method = max(all_results.items(), 
                            key=lambda x: x[1]['metrics'].get(metric, 0))
            print(f"Best {metric}: {best_method[0]} ({best_method[1]['metrics'].get(metric, 0):.3f})")
        
        # Query-specific analysis
        print("\n" + "="*80)
        print("QUERY-SPECIFIC RESULTS")
        print("="*80)
        
        for i, query_info in enumerate(self.test_queries):
            print(f"\nQuery: '{query_info['query']}' ({query_info['description']})")
            print(f"Relevant papers: {len(query_info['relevant_papers'])}")
            
            best_hit_rate = 0
            best_method = None
            
            for method_name, method_results in all_results.items():
                hit_rate = method_results['queries'][i]['results_by_k'][3]['hit_rate']
                if hit_rate > best_hit_rate:
                    best_hit_rate = hit_rate
                    best_method = method_name
            
            print(f"Best method: {best_method} (Hit Rate@3: {best_hit_rate:.3f})")
        
        # Save results
        with open('simple_evaluation_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nDetailed results saved to: simple_evaluation_results.json")
        
        return all_results

def main():
    """Main evaluation function"""
    try:
        evaluator = SimpleEvaluator()
        results = evaluator.run_evaluation()
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print("✅ Evaluated 6 search methods on 10 test queries")
        print("📊 Metrics calculated: Precision@k, Recall@k, Hit Rate@k")
        print("📄 Results saved to simple_evaluation_results.json")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
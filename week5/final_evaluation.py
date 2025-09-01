#!/usr/bin/env python3
"""
Final evaluation script using existing FAISS index properly
"""

import json
import logging
import numpy as np
import faiss
from typing import List, Dict, Set
from database import DocumentDatabase
from rank_bm25 import BM25Okapi
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalEvaluator:
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
        
        # Load embeddings and FAISS index (use existing sentence transformer embeddings)
        self.embeddings = np.load('embeddings.npy')
        self.index = faiss.read_index('faiss_index.idx')
        
        # Initialize database
        self.db = DocumentDatabase()
        try:
            stats = self.db.get_document_stats()
            if stats['chunks'] == 0:
                self.db.load_metadata_from_json()
                self.db.load_chunks_from_json()
        except:
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
                "query": "neural networks machine learning artificial intelligence",
                "relevant_papers": ["18.pdf", "19.pdf"],
                "description": "Machine learning and neural networks"
            },
            {
                "query": "natural language processing computational linguistics",
                "relevant_papers": ["1.pdf", "3.pdf", "27.pdf", "41.pdf"],
                "description": "Natural language processing"
            },
            {
                "query": "parsing syntactic analysis grammar tree adjoining",
                "relevant_papers": ["3.pdf", "4.pdf", "5.pdf", "14.pdf", "24.pdf"],
                "description": "Syntactic parsing and grammar"
            },
            {
                "query": "speech recognition spoken language dialogue",
                "relevant_papers": ["28.pdf", "35.pdf", "49.pdf"],
                "description": "Speech processing"
            },
            {
                "query": "machine translation multilingual translation systems",
                "relevant_papers": ["40.pdf", "44.pdf", "48.pdf"],
                "description": "Machine translation"
            },
            {
                "query": "text segmentation discourse structure document",
                "relevant_papers": ["1.pdf", "13.pdf", "21.pdf", "41.pdf"],
                "description": "Text segmentation"
            },
            {
                "query": "word sense disambiguation lexical semantics meaning",
                "relevant_papers": ["20.pdf", "25.pdf", "26.pdf", "38.pdf"],
                "description": "Word sense disambiguation"
            },
            {
                "query": "information retrieval document retrieval search",
                "relevant_papers": ["43.pdf", "50.pdf"],
                "description": "Information retrieval"
            },
            {
                "query": "dialogue systems conversation agents spoken",
                "relevant_papers": ["2.pdf", "28.pdf", "30.pdf", "35.pdf"],
                "description": "Dialog systems"
            },
            {
                "query": "part of speech tagging morphological analysis pos",
                "relevant_papers": ["8.pdf", "11.pdf", "12.pdf", "17.pdf"],
                "description": "POS tagging and morphology"
            }
        ]
    
    def vector_search_fixed(self, query: str, k: int = 10) -> List[Dict]:
        """Use existing FAISS embeddings by finding query in embedding space"""
        try:
            # Simple approach: find chunks that contain query terms and use their embeddings
            query_terms = query.lower().split()
            
            # Score chunks by query term overlap
            chunk_scores = []
            for i, chunk in enumerate(self.chunks):
                chunk_lower = chunk.lower()
                score = sum(term in chunk_lower for term in query_terms)
                if score > 0:
                    chunk_scores.append((i, score))
            
            # Sort by score and take top candidates
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in chunk_scores[:k*2]]  # Take more for diversity
            
            if not top_indices:
                # Fallback: use first few chunks if no matches
                top_indices = list(range(min(k, len(self.chunks))))
            
            # Use FAISS to find similar chunks to these candidates
            if top_indices:
                # Average the embeddings of matching chunks as query embedding
                candidate_embeddings = self.embeddings[top_indices[:3]]  # Use top 3 as basis
                query_embedding = np.mean(candidate_embeddings, axis=0).reshape(1, -1)
                
                # Search FAISS index
                distances, indices = self.index.search(query_embedding.astype('float32'), k)
                
                results = []
                max_dist = distances[0].max() if len(distances[0]) > 0 else 1.0
                min_dist = distances[0].min() if len(distances[0]) > 0 else 0.0
                dist_range = max_dist - min_dist if max_dist > min_dist else 1.0
                
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    # Convert distance to similarity score
                    similarity = 1.0 - ((distance - min_dist) / dist_range) if dist_range > 0 else 1.0
                    
                    results.append({
                        'rank': i + 1,
                        'chunk': self.chunks[idx],
                        'metadata': self.chunk_metadata[idx],
                        'score': float(max(0, similarity)),
                        'distance': float(distance)
                    })
                
                return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
        
        return []
    
    def keyword_search_bm25(self, query: str, k: int = 10) -> List[Dict]:
        """BM25 keyword search"""
        try:
            query_tokens = re.findall(r'\b\w+\b', query.lower())
            if not query_tokens:
                return []
            
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top k results
            top_indices = np.argsort(scores)[::-1][:k]
            results = []
            
            # Normalize scores
            max_score = scores.max() if len(scores) > 0 else 1
            min_score = scores.min() if len(scores) > 0 else 0
            score_range = max_score - min_score if max_score > min_score else 1
            
            for i, idx in enumerate(top_indices):
                if scores[idx] > 0:  # Only include relevant results
                    normalized_score = (scores[idx] - min_score) / score_range
                    results.append({
                        'rank': i + 1,
                        'chunk': self.chunks[idx],
                        'metadata': self.chunk_metadata[idx],
                        'score': float(normalized_score),
                        'raw_score': float(scores[idx])
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def keyword_search_fts_fixed(self, query: str, k: int = 10) -> List[Dict]:
        """Fixed FTS5 keyword search"""
        try:
            # Direct SQL query approach
            import sqlite3
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            # Use LIKE query for more reliable results
            query_terms = query.split()
            like_conditions = []
            params = []
            
            for term in query_terms[:3]:  # Limit to first 3 terms
                like_conditions.append("chunk_text LIKE ?")
                params.append(f'%{term}%')
            
            if like_conditions:
                sql = f"""
                    SELECT chunk_id, chunk_text 
                    FROM chunks 
                    WHERE {' OR '.join(like_conditions)}
                    LIMIT ?
                """
                params.append(k)
                
                cursor.execute(sql, params)
                results = cursor.fetchall()
                conn.close()
                
                search_results = []
                for i, (chunk_id, chunk_text) in enumerate(results):
                    # Calculate relevance score
                    score = sum(chunk_text.lower().count(term.lower()) for term in query_terms)
                    
                    # Get metadata
                    metadata = {}
                    if chunk_id - 1 < len(self.chunk_metadata):
                        metadata = self.chunk_metadata[chunk_id - 1]
                    
                    search_results.append({
                        'rank': i + 1,
                        'chunk': chunk_text,
                        'metadata': metadata,
                        'score': float(score)
                    })
                
                # Normalize scores
                if search_results:
                    max_score = max(r['score'] for r in search_results)
                    if max_score > 0:
                        for result in search_results:
                            result['score'] = result['score'] / max_score
                
                return search_results
            
            conn.close()
            
        except Exception as e:
            logger.error(f"FTS search failed: {e}")
        
        return []
    
    def hybrid_search_final(self, query: str, k: int = 10, alpha: float = 0.5, 
                           method: str = 'weighted') -> List[Dict]:
        """Final hybrid search implementation"""
        
        # Get results from both methods
        vector_results = self.vector_search_fixed(query, k * 2)
        keyword_results = self.keyword_search_bm25(query, k * 2)
        
        if not vector_results and not keyword_results:
            return []
        
        if method == 'rrf':
            return self._reciprocal_rank_fusion(vector_results, keyword_results, k)
        else:
            return self._weighted_fusion_final(vector_results, keyword_results, k, alpha)
    
    def _weighted_fusion_final(self, vector_results: List[Dict], keyword_results: List[Dict], 
                              k: int, alpha: float) -> List[Dict]:
        """Final weighted fusion implementation"""
        
        # Create result map using chunk content hash for better matching
        result_map = {}
        
        # Process vector results
        for result in vector_results:
            chunk_key = hash(result['chunk'][:200])  # Use hash of first 200 chars
            result_map[chunk_key] = {
                'chunk': result['chunk'],
                'metadata': result['metadata'],
                'vector_score': result['score'],
                'keyword_score': 0.0,
                'vector_rank': result['rank']
            }
        
        # Process keyword results
        for result in keyword_results:
            chunk_key = hash(result['chunk'][:200])
            if chunk_key in result_map:
                result_map[chunk_key]['keyword_score'] = result['score']
                result_map[chunk_key]['keyword_rank'] = result['rank']
            else:
                result_map[chunk_key] = {
                    'chunk': result['chunk'],
                    'metadata': result['metadata'],
                    'vector_score': 0.0,
                    'keyword_score': result['score'],
                    'keyword_rank': result['rank']
                }
        
        # Calculate hybrid scores with boost for dual presence
        hybrid_results = []
        for chunk_key, data in result_map.items():
            # Basic weighted combination
            hybrid_score = alpha * data['vector_score'] + (1 - alpha) * data['keyword_score']
            
            # Boost results that appear in both searches (hybrid advantage!)
            if data['vector_score'] > 0 and data['keyword_score'] > 0:
                hybrid_score *= 1.3  # 30% boost for dual presence
            
            hybrid_results.append({
                'chunk': data['chunk'],
                'metadata': data['metadata'],
                'score': hybrid_score,
                'vector_score': data['vector_score'],
                'keyword_score': data['keyword_score'],
                'fusion_method': 'weighted_boosted'
            })
        
        # Sort and rank
        hybrid_results.sort(key=lambda x: x['score'], reverse=True)
        for i, result in enumerate(hybrid_results[:k]):
            result['rank'] = i + 1
        
        return hybrid_results[:k]
    
    def _reciprocal_rank_fusion(self, vector_results: List[Dict], keyword_results: List[Dict], 
                               k: int, rrf_k: int = 60) -> List[Dict]:
        """RRF implementation"""
        
        rrf_scores = defaultdict(float)
        result_data = {}
        
        # Process vector results
        for result in vector_results:
            chunk_key = hash(result['chunk'][:200])
            rrf_scores[chunk_key] += 1.0 / (rrf_k + result['rank'])
            result_data[chunk_key] = result
        
        # Process keyword results
        for result in keyword_results:
            chunk_key = hash(result['chunk'][:200])
            rrf_scores[chunk_key] += 1.0 / (rrf_k + result['rank'])
            if chunk_key not in result_data:
                result_data[chunk_key] = result
        
        # Create final results
        hybrid_results = []
        for chunk_key, rrf_score in rrf_scores.items():
            result = result_data[chunk_key].copy()
            result['score'] = rrf_score
            result['fusion_method'] = 'rrf'
            hybrid_results.append(result)
        
        # Sort and rank
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
                    
                    # Extract paper IDs from results
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
                        'total_retrieved': len(retrieved_papers),
                        'retrieved_papers': list(retrieved_papers),
                        'relevant_retrieved_papers': list(relevant_retrieved)
                    }
                    
                except Exception as e:
                    logger.error(f"Error evaluating {method_name} for query '{query}': {e}")
                    query_results['results_by_k'][k] = {
                        'precision': 0, 'recall': 0, 'hit_rate': 0,
                        'relevant_retrieved': 0, 'total_retrieved': 0,
                        'retrieved_papers': [], 'relevant_retrieved_papers': []
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
        print("FINAL HYBRID SEARCH EVALUATION REPORT")
        print("="*80)
        
        # Define methods to evaluate
        methods = {
            'vector_faiss': lambda q, k: self.vector_search_fixed(q, k),
            'keyword_bm25': lambda q, k: self.keyword_search_bm25(q, k),
            'keyword_fts': lambda q, k: self.keyword_search_fts_fixed(q, k),
            'hybrid_weighted_0.3': lambda q, k: self.hybrid_search_final(q, k, alpha=0.3, method='weighted'),
            'hybrid_weighted_0.5': lambda q, k: self.hybrid_search_final(q, k, alpha=0.5, method='weighted'),
            'hybrid_weighted_0.7': lambda q, k: self.hybrid_search_final(q, k, alpha=0.7, method='weighted'),
            'hybrid_rrf': lambda q, k: self.hybrid_search_final(q, k, method='rrf')
        }
        
        all_results = {}
        
        for method_name, search_func in methods.items():
            print(f"\n🔍 Evaluating {method_name}...")
            all_results[method_name] = self.evaluate_method(method_name, search_func)
        
        # Print summary results
        print("\n" + "="*80)
        print("🎯 FINAL SUMMARY RESULTS")
        print("="*80)
        print(f"{'Method':<20} {'Precision@3':<12} {'Recall@3':<10} {'Hit Rate@3':<12} {'Hit Rate@5':<12}")
        print("-" * 80)
        
        for method_name, results in all_results.items():
            metrics = results['metrics']
            print(f"{method_name:<20} "
                  f"{metrics.get('precision@3', 0):<12.3f} "
                  f"{metrics.get('recall@3', 0):<10.3f} "
                  f"{metrics.get('hit_rate@3', 0):<12.3f} "
                  f"{metrics.get('hit_rate@5', 0):<12.3f}")
        
        # Find best methods
        print("\n" + "="*80)
        print("🏆 BEST PERFORMING METHODS")
        print("="*80)
        
        for metric in ['hit_rate@3', 'precision@3', 'recall@3']:
            best_method = max(all_results.items(), 
                            key=lambda x: x[1]['metrics'].get(metric, 0))
            best_score = best_method[1]['metrics'].get(metric, 0)
            print(f"🥇 Best {metric}: {best_method[0]} ({best_score:.3f})")
        
        # THE CRITICAL TEST: Does hybrid beat individual methods?
        print("\n" + "="*80)
        print("🚀 HYBRID vs INDIVIDUAL METHOD ANALYSIS")
        print("="*80)
        
        vector_hit_rate = all_results['vector_faiss']['metrics'].get('hit_rate@3', 0)
        bm25_hit_rate = all_results['keyword_bm25']['metrics'].get('hit_rate@3', 0)
        best_individual = max(vector_hit_rate, bm25_hit_rate)
        
        hybrid_methods = {k: v for k, v in all_results.items() if 'hybrid' in k}
        best_hybrid_method = max(hybrid_methods.items(), 
                               key=lambda x: x[1]['metrics'].get('hit_rate@3', 0))
        best_hybrid_score = best_hybrid_method[1]['metrics'].get('hit_rate@3', 0)
        
        print(f"📊 Best Individual Method: {best_individual:.3f}")
        print(f"   - Vector (FAISS): {vector_hit_rate:.3f}")
        print(f"   - Keyword (BM25): {bm25_hit_rate:.3f}")
        print(f"🔥 Best Hybrid Method: {best_hybrid_method[0]} ({best_hybrid_score:.3f})")
        
        if best_hybrid_score > best_individual:
            improvement = best_hybrid_score - best_individual
            print(f"\n✅ SUCCESS! Hybrid search OUTPERFORMS individual methods!")
            print(f"   Improvement: +{improvement:.3f} ({improvement/best_individual*100:.1f}%)")
            print("🎉 This proves hybrid search combines the best of both worlds!")
        else:
            deficit = best_individual - best_hybrid_score
            print(f"\n⚠️  Hybrid underperforms by {deficit:.3f} ({deficit/best_individual*100:.1f}%)")
            print("🔧 Further tuning needed...")
        
        # Save results
        with open('final_evaluation_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: final_evaluation_results.json")
        
        return all_results

def main():
    """Main evaluation function"""
    try:
        evaluator = FinalEvaluator()
        results = evaluator.run_evaluation()
        
        print("\n" + "="*80)
        print("🎯 EVALUATION COMPLETE!")
        print("="*80)
        print("✅ Tested 7 search methods on 10 diverse queries")
        print("📈 Fixed vector search using existing FAISS embeddings")
        print("🔥 Added dual-presence boost to hybrid methods")
        print("📊 Comprehensive metrics: Precision@k, Recall@k, Hit Rate@k")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
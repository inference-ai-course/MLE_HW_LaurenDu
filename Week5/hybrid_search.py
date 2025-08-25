import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from rank_bm25 import BM25Okapi
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, rag_pipeline):
        """
        Initialize hybrid retriever with RAG pipeline
        
        Args:
            rag_pipeline: RAGPipeline instance with loaded data
        """
        self.rag_pipeline = rag_pipeline
        self.bm25_index = None
        self.bm25_chunks = []
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from chunks"""
        try:
            if not self.rag_pipeline.chunks:
                logger.warning("No chunks available for BM25 indexing")
                return
            
            # Tokenize chunks for BM25
            tokenized_chunks = []
            for chunk in self.rag_pipeline.chunks:
                # Simple tokenization: lowercase and split on non-alphanumeric
                tokens = re.findall(r'\b\w+\b', chunk.lower())
                tokenized_chunks.append(tokens)
            
            self.bm25_index = BM25Okapi(tokenized_chunks)
            self.bm25_chunks = self.rag_pipeline.chunks
            logger.info(f"BM25 index built with {len(tokenized_chunks)} documents")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25_index = None
    
    def vector_search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Perform vector search using FAISS
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with normalized scores
        """
        results = self.rag_pipeline.search(query, k)
        
        # Normalize distances to scores (0-1 range)
        if results:
            max_distance = max(r['distance'] for r in results)
            min_distance = min(r['distance'] for r in results)
            distance_range = max_distance - min_distance if max_distance > min_distance else 1
            
            for result in results:
                # Convert distance to similarity score (higher is better)
                result['score'] = 1 - (result['distance'] - min_distance) / distance_range
                result['search_type'] = 'vector'
        
        return results
    
    def keyword_search_fts(self, query: str, k: int = 10) -> List[Dict]:
        """
        Perform keyword search using FTS5
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with normalized scores
        """
        results = self.rag_pipeline.keyword_search(query, k)
        
        # Normalize scores to 0-1 range
        if results:
            max_score = max(r['score'] for r in results)
            min_score = min(r['score'] for r in results)
            score_range = max_score - min_score if max_score > min_score else 1
            
            for result in results:
                result['score'] = (result['score'] - min_score) / score_range
                result['search_type'] = 'keyword_fts'
        
        return results
    
    def keyword_search_bm25(self, query: str, k: int = 10) -> List[Dict]:
        """
        Perform keyword search using BM25
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with normalized scores
        """
        if not self.bm25_index:
            logger.warning("BM25 index not available")
            return []
        
        try:
            # Tokenize query
            query_tokens = re.findall(r'\b\w+\b', query.lower())
            
            # Get BM25 scores for all documents
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top k results
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            max_score = scores[top_indices[0]] if len(top_indices) > 0 else 1
            min_score = scores[top_indices[-1]] if len(top_indices) > 0 else 0
            score_range = max_score - min_score if max_score > min_score else 1
            
            for i, idx in enumerate(top_indices):
                if scores[idx] > 0:  # Only include relevant results
                    # Normalize score to 0-1 range
                    normalized_score = (scores[idx] - min_score) / score_range
                    
                    results.append({
                        'rank': i + 1,
                        'chunk': self.bm25_chunks[idx],
                        'metadata': self.rag_pipeline.chunk_metadata[idx],
                        'score': normalized_score,
                        'search_type': 'keyword_bm25'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.5, 
                     keyword_method: str = 'fts', merge_method: str = 'weighted') -> List[Dict]:
        """
        Perform hybrid search combining vector and keyword search
        
        Args:
            query: Search query
            k: Number of final results to return
            alpha: Weight for vector search (1-alpha for keyword search)
            keyword_method: 'fts' or 'bm25' for keyword search method
            merge_method: 'weighted' or 'rrf' for score combination method
            
        Returns:
            List of hybrid search results
        """
        # Retrieve more results initially to ensure good coverage
        search_k = min(k * 3, 20)
        
        # Perform both searches
        vector_results = self.vector_search(query, search_k)
        
        if keyword_method == 'bm25':
            keyword_results = self.keyword_search_bm25(query, search_k)
        else:
            keyword_results = self.keyword_search_fts(query, search_k)
        
        if merge_method == 'rrf':
            return self._merge_results_rrf(vector_results, keyword_results, k)
        else:
            return self._merge_results_weighted(vector_results, keyword_results, k, alpha)
    
    def _merge_results_weighted(self, vector_results: List[Dict], keyword_results: List[Dict], 
                               k: int, alpha: float) -> List[Dict]:
        """
        Merge results using weighted score combination
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            k: Number of results to return
            alpha: Weight for vector scores
            
        Returns:
            Merged and ranked results
        """
        # Create a mapping from chunk content to results
        result_map = {}
        
        # Add vector results
        for result in vector_results:
            chunk_key = result['chunk'][:100]  # Use first 100 chars as key
            result_map[chunk_key] = {
                'chunk': result['chunk'],
                'metadata': result['metadata'],
                'vector_score': result['score'],
                'keyword_score': 0.0,
                'vector_rank': result['rank'],
                'keyword_rank': None
            }
        
        # Add keyword results
        for result in keyword_results:
            chunk_key = result['chunk'][:100]
            if chunk_key in result_map:
                result_map[chunk_key]['keyword_score'] = result['score']
                result_map[chunk_key]['keyword_rank'] = result['rank']
            else:
                result_map[chunk_key] = {
                    'chunk': result['chunk'],
                    'metadata': result['metadata'],
                    'vector_score': 0.0,
                    'keyword_score': result['score'],
                    'vector_rank': None,
                    'keyword_rank': result['rank']
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
                'keyword_score': data['keyword_score'],
                'search_type': 'hybrid_weighted'
            })
        
        # Sort by hybrid score and return top k
        hybrid_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Add ranks
        for i, result in enumerate(hybrid_results[:k]):
            result['rank'] = i + 1
        
        return hybrid_results[:k]
    
    def _merge_results_rrf(self, vector_results: List[Dict], keyword_results: List[Dict], 
                          k: int, rrf_k: int = 60) -> List[Dict]:
        """
        Merge results using Reciprocal Rank Fusion (RRF)
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            k: Number of results to return
            rrf_k: RRF parameter (typically 60)
            
        Returns:
            Merged and ranked results using RRF
        """
        # Create a mapping from chunk content to RRF scores
        rrf_scores = defaultdict(float)
        result_data = {}
        
        # Add vector results
        for result in vector_results:
            chunk_key = result['chunk'][:100]
            rrf_scores[chunk_key] += 1.0 / (rrf_k + result['rank'])
            result_data[chunk_key] = result
        
        # Add keyword results
        for result in keyword_results:
            chunk_key = result['chunk'][:100]
            rrf_scores[chunk_key] += 1.0 / (rrf_k + result['rank'])
            if chunk_key not in result_data:
                result_data[chunk_key] = result
        
        # Create final results
        hybrid_results = []
        for chunk_key, rrf_score in rrf_scores.items():
            result = result_data[chunk_key].copy()
            result['score'] = rrf_score
            result['search_type'] = 'hybrid_rrf'
            hybrid_results.append(result)
        
        # Sort by RRF score and return top k
        hybrid_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Add ranks
        for i, result in enumerate(hybrid_results[:k]):
            result['rank'] = i + 1
        
        return hybrid_results[:k]
    
    def compare_search_methods(self, query: str, k: int = 5) -> Dict[str, List[Dict]]:
        """
        Compare different search methods for a query
        
        Args:
            query: Search query
            k: Number of results per method
            
        Returns:
            Dictionary with results from each method
        """
        return {
            'vector': self.vector_search(query, k),
            'keyword_fts': self.keyword_search_fts(query, k),
            'keyword_bm25': self.keyword_search_bm25(query, k),
            'hybrid_weighted': self.hybrid_search(query, k, alpha=0.5, merge_method='weighted'),
            'hybrid_rrf': self.hybrid_search(query, k, alpha=0.5, merge_method='rrf')
        }

def test_hybrid_search():
    """Test function for hybrid search"""
    from rag_pipeline import RAGPipeline
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    pipeline.run_pipeline()
    
    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(pipeline)
    
    # Test query
    query = "machine learning neural networks"
    
    print(f"Testing hybrid search for query: '{query}'")
    print("=" * 50)
    
    # Compare methods
    results = hybrid_retriever.compare_search_methods(query, k=3)
    
    for method, method_results in results.items():
        print(f"\n{method.upper()} RESULTS:")
        print("-" * 30)
        for result in method_results:
            print(f"Rank {result['rank']} (Score: {result['score']:.4f})")
            print(f"Paper: {result['metadata']['pdf_file']}")
            print(f"Chunk: {result['chunk'][:150]}...")
            print()

if __name__ == "__main__":
    test_hybrid_search()
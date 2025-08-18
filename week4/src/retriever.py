import numpy as np
import faiss
import pickle
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", data_directory: str = "../data"):
        self.model_name = model_name
        self.data_directory = data_directory
        self.model = None
        self.index = None
        self.chunk_metadata = None
        self._load_components()
    
    def _load_components(self):
        logger.info("Loading retrieval components...")
        
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Loaded sentence transformer model: {self.model_name}")
        
        index_path = os.path.join(self.data_directory, "faiss_index.bin")
        metadata_path = os.path.join(self.data_directory, "chunk_metadata.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Index or metadata files not found in {self.data_directory}. "
                "Please run the embedding pipeline first."
            )
        
        self.index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
        with open(metadata_path, 'rb') as f:
            self.chunk_metadata = pickle.load(f)
        logger.info(f"Loaded metadata for {len(self.chunk_metadata)} chunks")
    
    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding.astype('float32')
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        query_embedding = self.embed_query(query)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunk_metadata):
                chunk = self.chunk_metadata[idx].copy()
                chunk['similarity_score'] = float(1 / (1 + distance))
                chunk['distance'] = float(distance)
                chunk['rank'] = i + 1
                results.append(chunk)
        
        return results
    
    def search_with_context(self, query: str, k: int = 3, context_window: int = 1) -> List[Dict]:
        initial_results = self.search(query, k * 2)
        
        enhanced_results = []
        for result in initial_results[:k]:
            chunk_idx = result['chunk_index']
            doc_id = result['document_id']
            
            context_chunks = []
            for metadata in self.chunk_metadata:
                if (metadata['document_id'] == doc_id and 
                    abs(metadata['chunk_index'] - chunk_idx) <= context_window):
                    context_chunks.append(metadata)
            
            context_chunks.sort(key=lambda x: x['chunk_index'])
            context_text = " ".join([c['text'] for c in context_chunks])
            
            enhanced_result = result.copy()
            enhanced_result['context_text'] = context_text
            enhanced_result['context_chunks'] = len(context_chunks)
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def format_results(self, results: List[Dict], include_metadata: bool = True) -> str:
        formatted = []
        
        for i, result in enumerate(results, 1):
            formatted.append(f"--- Result {i} ---")
            formatted.append(f"Document: {result['document_id']}")
            formatted.append(f"Chunk: {result['chunk_id']}")
            formatted.append(f"Similarity Score: {result['similarity_score']:.4f}")
            
            if include_metadata:
                formatted.append(f"Token Count: {result['token_count']}")
                formatted.append(f"Chunk Index: {result['chunk_index']}")
            
            formatted.append("Text:")
            formatted.append(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])
            formatted.append("")
        
        return "\n".join(formatted)

if __name__ == "__main__":
    try:
        retriever = RAGRetriever()
        
        test_queries = [
            "machine learning",
            "neural networks",
            "natural language processing",
            "transformer architecture",
            "attention mechanism"
        ]
        
        print("=== RAG Retrieval System Test ===\n")
        
        for query in test_queries:
            print(f"Query: '{query}'")
            print("-" * 50)
            
            results = retriever.search(query, k=3)
            print(retriever.format_results(results, include_metadata=False))
            print("\n" + "="*70 + "\n")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the full pipeline first:")
        print("1. python pdf_processor.py")
        print("2. python chunker.py") 
        print("3. python embedder.py")
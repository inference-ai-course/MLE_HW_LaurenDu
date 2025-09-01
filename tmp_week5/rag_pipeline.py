import os
import json
import pickle
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
import re
from tqdm import tqdm
import logging
from database import DocumentDatabase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, pdf_dir: str = "PDFs", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG pipeline
        
        Args:
            pdf_dir: Directory containing PDF files
            model_name: Name of the sentence transformer model to use
        """
        self.pdf_dir = pdf_dir
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.chunks = []
        self.chunk_metadata = []
        self.index = None
        self.db = DocumentDatabase()
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using PyMuPDF
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and normalizing
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk (in characters)
            overlap: Overlap between chunks (in characters)
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_pdfs(self) -> Tuple[List[str], List[Dict]]:
        """
        Process all PDFs in the directory
        
        Returns:
            Tuple of (chunks, metadata)
        """
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        pdf_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort by number
        
        all_chunks = []
        all_metadata = []
        
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            paper_id = pdf_file.split('.')[0]
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                continue
                
            # Clean text
            text = self.clean_text(text)
            
            # Chunk text
            chunks = self.chunk_text(text)
            
            # Add chunks and metadata
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'paper_id': paper_id,
                    'chunk_id': i,
                    'chunk_text': chunk[:100] + "..." if len(chunk) > 100 else chunk,
                    'pdf_file': pdf_file
                })
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(pdf_files)} papers")
        return all_chunks, all_metadata
    
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for text chunks
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Numpy array of embeddings
        """
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            FAISS index
        """
        logger.info("Building FAISS index...")
        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings.astype('float32'))
        return index
    
    def save_data(self, chunks: List[str], metadata: List[Dict], 
                  embeddings: np.ndarray, index: faiss.Index):
        """
        Save processed data to files
        
        Args:
            chunks: List of text chunks
            metadata: List of chunk metadata
            embeddings: Numpy array of embeddings
            index: FAISS index
        """
        # Save chunks and metadata
        with open('processed_chunks.json', 'w') as f:
            json.dump({
                'chunks': chunks,
                'metadata': metadata
            }, f, indent=2)
        
        # Save embeddings
        np.save('embeddings.npy', embeddings)
        
        # Save FAISS index
        faiss.write_index(index, 'faiss_index.idx')
        
        logger.info("Data saved successfully")
    
    def load_data(self) -> Tuple[List[str], List[Dict], np.ndarray, faiss.Index]:
        """
        Load processed data from files
        
        Returns:
            Tuple of (chunks, metadata, embeddings, index)
        """
        # Load chunks and metadata
        with open('processed_chunks.json', 'r') as f:
            data = json.load(f)
            chunks = data['chunks']
            metadata = data['metadata']
        
        # Load embeddings
        embeddings = np.load('embeddings.npy')
        
        # Load FAISS index
        index = faiss.read_index('faiss_index.idx')
        
        return chunks, metadata, embeddings, index
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search for similar chunks given a query
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        if self.index is None or not self.chunks:
            raise ValueError("Index and chunks not loaded. Run load_data() first.")
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            results.append({
                'rank': i + 1,
                'chunk': self.chunks[idx],
                'metadata': self.chunk_metadata[idx],
                'distance': float(distance)
            })
        
        return results
    
    def run_pipeline(self, force_reprocess: bool = False):
        """
        Run the complete RAG pipeline
        
        Args:
            force_reprocess: If True, reprocess PDFs even if data exists
        """
        # Check if data already exists
        if not force_reprocess and os.path.exists('processed_chunks.json'):
            logger.info("Loading existing processed data...")
            self.chunks, self.chunk_metadata, embeddings, self.index = self.load_data()
            # Setup database with existing data
            self._setup_database()
            return
        
        # Process PDFs
        chunks, metadata = self.process_pdfs()
        
        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # Build FAISS index
        index = self.build_faiss_index(embeddings)
        
        # Save data
        self.save_data(chunks, metadata, embeddings, index)
        
        # Store in memory
        self.chunks = chunks
        self.chunk_metadata = metadata
        self.index = index
        
        # Setup database
        self._setup_database()
        
        logger.info("RAG pipeline completed successfully!")
    
    def _setup_database(self):
        """Setup database with existing data if needed"""
        try:
            stats = self.db.get_document_stats()
            if stats['chunks'] == 0:
                logger.info("Database is empty, loading data...")
                self.db.load_metadata_from_json()
                self.db.load_chunks_from_json()
                logger.info("Database setup complete")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def keyword_search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Perform keyword search using FTS5
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        try:
            results = self.db.search_fts(query, k)
            search_results = []
            
            for i, (chunk_id, chunk_text, score) in enumerate(results):
                chunk_text_full, metadata = self.db.get_chunk_by_id(chunk_id)
                search_results.append({
                    'rank': i + 1,
                    'chunk': chunk_text_full,
                    'metadata': metadata,
                    'score': score,
                    'search_type': 'keyword'
                })
            
            return search_results
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

def main():
    """Main function to run the RAG pipeline"""
    pipeline = RAGPipeline()
    pipeline.run_pipeline()
    
    # Test search functionality
    test_queries = [
        "machine learning",
        "natural language processing",
        "neural networks",
        "transformer models",
        "deep learning"
    ]
    
    print("\n" + "="*50)
    print("TESTING SEARCH FUNCTIONALITY")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 30)
        results = pipeline.search(query, k=3)
        
        for result in results:
            print(f"Rank {result['rank']} (Distance: {result['distance']:.4f})")
            print(f"Paper: {result['metadata']['pdf_file']}")
            print(f"Chunk: {result['chunk'][:200]}...")
            print()

if __name__ == "__main__":
    main()


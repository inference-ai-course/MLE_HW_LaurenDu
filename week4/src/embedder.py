import json
import numpy as np
import faiss
import pickle
import os
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingIndexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", output_directory: str = "../data"):
        self.model_name = model_name
        self.output_directory = output_directory
        self.model = None
        self.index = None
        self.chunk_metadata = None
        
        os.makedirs(output_directory, exist_ok=True)
    
    def load_model(self):
        logger.info(f"Loading sentence transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        if self.model is None:
            self.load_model()
        
        texts = [chunk["text"] for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        dimension = embeddings.shape[1]
        
        logger.info(f"Creating FAISS index with dimension: {dimension}")
        index = faiss.IndexFlatL2(dimension)
        
        logger.info(f"Adding {len(embeddings)} vectors to index...")
        index.add(embeddings.astype('float32'))
        
        logger.info(f"Index created successfully. Total vectors: {index.ntotal}")
        return index
    
    def save_index_and_metadata(self, index: faiss.Index, chunks: List[Dict], embeddings: np.ndarray):
        index_path = os.path.join(self.output_directory, "faiss_index.bin")
        metadata_path = os.path.join(self.output_directory, "chunk_metadata.pkl")
        embeddings_path = os.path.join(self.output_directory, "embeddings.npy")
        
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(index, index_path)
        
        logger.info(f"Saving chunk metadata to {metadata_path}")
        with open(metadata_path, 'wb') as f:
            pickle.dump(chunks, f)
        
        logger.info(f"Saving embeddings to {embeddings_path}")
        np.save(embeddings_path, embeddings)
        
        logger.info("Index, metadata, and embeddings saved successfully!")
    
    def load_index_and_metadata(self) -> Tuple[faiss.Index, List[Dict]]:
        index_path = os.path.join(self.output_directory, "faiss_index.bin")
        metadata_path = os.path.join(self.output_directory, "chunk_metadata.pkl")
        
        logger.info(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        
        logger.info(f"Loading chunk metadata from {metadata_path}")
        with open(metadata_path, 'rb') as f:
            chunk_metadata = pickle.load(f)
        
        self.index = index
        self.chunk_metadata = chunk_metadata
        
        logger.info(f"Loaded index with {index.ntotal} vectors and {len(chunk_metadata)} metadata entries")
        return index, chunk_metadata
    
    def build_index_from_chunks(self, chunks_file: str):
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        embeddings = self.generate_embeddings(chunks)
        index = self.create_faiss_index(embeddings)
        self.save_index_and_metadata(index, chunks, embeddings)
        
        return index, chunks

if __name__ == "__main__":
    indexer = EmbeddingIndexer()
    
    chunks_file = "../data/chunks.json"
    if os.path.exists(chunks_file):
        index, chunks = indexer.build_index_from_chunks(chunks_file)
        print(f"Successfully built index with {index.ntotal} embeddings!")
    else:
        print(f"Chunks file not found: {chunks_file}")
        print("Please run pdf_processor.py and chunker.py first.")
import tiktoken
import json
import logging
from typing import List, Dict, Tuple
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextChunker:
    def __init__(self, chunk_size: int = 512, overlap_size: int = 50, encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, document_id: str) -> List[Dict]:
        tokens = self.encoding.encode(text)
        chunks = []
        chunk_id = 0
        
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk_data = {
                "chunk_id": f"{document_id}_chunk_{chunk_id}",
                "document_id": document_id,
                "chunk_index": chunk_id,
                "text": chunk_text.strip(),
                "token_count": len(chunk_tokens),
                "start_token": start,
                "end_token": end
            }
            
            chunks.append(chunk_data)
            chunk_id += 1
            
            if end >= len(tokens):
                break
            
            start = end - self.overlap_size
        
        return chunks
    
    def process_documents(self, documents: Dict[str, str], output_directory: str) -> List[Dict]:
        all_chunks = []
        
        logger.info(f"Chunking {len(documents)} documents...")
        
        for doc_id, text in documents.items():
            logger.info(f"Chunking {doc_id}...")
            
            if len(text.strip()) == 0:
                logger.warning(f"Skipping empty document: {doc_id}")
                continue
            
            chunks = self.chunk_text(text, doc_id)
            all_chunks.extend(chunks)
            
            logger.info(f"Created {len(chunks)} chunks for {doc_id}")
        
        output_path = os.path.join(output_directory, "chunks.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(all_chunks)} chunks to {output_path}")
        
        self._print_statistics(all_chunks)
        return all_chunks
    
    def _print_statistics(self, chunks: List[Dict]):
        total_chunks = len(chunks)
        token_counts = [chunk["token_count"] for chunk in chunks]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        min_tokens = min(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0
        
        logger.info(f"Chunking Statistics:")
        logger.info(f"  Total chunks: {total_chunks}")
        logger.info(f"  Average tokens per chunk: {avg_tokens:.1f}")
        logger.info(f"  Min tokens: {min_tokens}")
        logger.info(f"  Max tokens: {max_tokens}")

if __name__ == "__main__":
    with open("../data/processed_documents.json", 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    chunker = TextChunker(chunk_size=512, overlap_size=50)
    chunks = chunker.process_documents(documents, "../data")
    print(f"Created {len(chunks)} chunks from {len(documents)} documents!")
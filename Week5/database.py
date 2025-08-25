import sqlite3
import json
import logging
from typing import List, Dict, Tuple
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentDatabase:
    def __init__(self, db_path: str = "document_index.db"):
        """
        Initialize the document database with SQLite and FTS5
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with proper schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create documents table for metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY,
                title TEXT,
                authors TEXT,
                abstract TEXT,
                pdf_url TEXT,
                local_file TEXT,
                year INTEGER,
                keywords TEXT
            )
        """)
        
        # Create doc_chunks table for chunk text and metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                chunk_index INTEGER,
                chunk_text TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            )
        """)
        
        # Create FTS5 virtual table for full-text search on chunks
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts USING fts5(
                chunk_text,
                content='chunks',
                content_rowid='chunk_id'
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def load_metadata_from_json(self, metadata_path: str = "metadata.json"):
        """
        Load document metadata from JSON file into the database
        
        Args:
            metadata_path: Path to the metadata JSON file
        """
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file not found: {metadata_path}")
            return
            
        with open(metadata_path, 'r') as f:
            metadata_list = json.load(f)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM documents")
        cursor.execute("DELETE FROM chunks")
        cursor.execute("DELETE FROM doc_chunks_fts")
        
        for i, doc in enumerate(metadata_list, 1):
            # Extract year from ID if available (format: cs/YYMMDDDvN)
            year = None
            if "abs/cs/" in doc["id"]:
                try:
                    date_part = doc["id"].split("/")[-1][:6]  # YYMMDDv format
                    year = 1900 + int(date_part[:2]) if int(date_part[:2]) > 50 else 2000 + int(date_part[:2])
                except:
                    year = None
            
            # Insert document metadata
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (doc_id, title, authors, abstract, pdf_url, local_file, year, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                i,  # Use 1-based indexing to match PDF files
                doc["title"],
                json.dumps(doc["authors"]),  # Store authors as JSON array
                doc["abstract"],
                doc["pdf_url"],
                doc["local_file"],
                year,
                ""  # We'll populate keywords from the text later
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Loaded {len(metadata_list)} documents into database")
    
    def load_chunks_from_json(self, chunks_path: str = "processed_chunks.json"):
        """
        Load processed chunks into the database
        
        Args:
            chunks_path: Path to the processed chunks JSON file
        """
        if not os.path.exists(chunks_path):
            logger.error(f"Chunks file not found: {chunks_path}")
            return
            
        with open(chunks_path, 'r') as f:
            data = json.load(f)
            chunks = data['chunks']
            chunk_metadata = data['metadata']
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing chunks
        cursor.execute("DELETE FROM chunks")
        cursor.execute("DELETE FROM doc_chunks_fts")
        
        # Insert chunks
        for i, (chunk, metadata) in enumerate(zip(chunks, chunk_metadata)):
            doc_id = int(metadata['paper_id'])
            chunk_index = metadata['chunk_id']
            
            cursor.execute("""
                INSERT INTO chunks (doc_id, chunk_index, chunk_text)
                VALUES (?, ?, ?)
            """, (doc_id, chunk_index, chunk))
            
            chunk_id = cursor.lastrowid
            
            # Insert into FTS table
            cursor.execute("""
                INSERT INTO doc_chunks_fts (rowid, chunk_text)
                VALUES (?, ?)
            """, (chunk_id, chunk))
        
        conn.commit()
        conn.close()
        logger.info(f"Loaded {len(chunks)} chunks into database")
    
    def search_fts(self, query: str, limit: int = 10) -> List[Tuple[int, str, float]]:
        """
        Perform FTS5 search on document chunks
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of tuples (chunk_id, chunk_text, rank_score)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # FTS5 query with BM25 ranking
        cursor.execute("""
            SELECT c.chunk_id, c.chunk_text, fts.rank
            FROM doc_chunks_fts fts
            JOIN chunks c ON c.chunk_id = fts.rowid
            WHERE doc_chunks_fts MATCH ?
            ORDER BY fts.rank
            LIMIT ?
        """, (query, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        # Convert rank to a positive score (FTS5 rank is negative)
        return [(chunk_id, text, -rank) for chunk_id, text, rank in results]
    
    def get_chunk_by_id(self, chunk_id: int) -> Tuple[str, Dict]:
        """
        Get chunk text and metadata by chunk ID
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Tuple of (chunk_text, metadata_dict)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.chunk_text, c.doc_id, c.chunk_index, d.title, d.authors, d.local_file
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.chunk_id = ?
        """, (chunk_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            chunk_text, doc_id, chunk_index, title, authors, local_file = result
            metadata = {
                'paper_id': str(doc_id),
                'chunk_id': chunk_index,
                'pdf_file': local_file,
                'title': title,
                'authors': json.loads(authors) if authors else []
            }
            return chunk_text, metadata
        
        return "", {}
    
    def get_document_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) as chunk_count FROM chunks GROUP BY doc_id")
        counts = cursor.fetchall()
        avg_chunks = sum(c[0] for c in counts) / len(counts) if counts else 0
        
        conn.close()
        
        return {
            'documents': doc_count,
            'chunks': chunk_count,
            'avg_chunks_per_doc': avg_chunks
        }

def setup_database():
    """Setup database with existing data"""
    db = DocumentDatabase()
    
    # Load metadata and chunks
    db.load_metadata_from_json()
    db.load_chunks_from_json()
    
    # Print stats
    stats = db.get_document_stats()
    logger.info(f"Database setup complete: {stats}")
    
    return db

if __name__ == "__main__":
    setup_database()
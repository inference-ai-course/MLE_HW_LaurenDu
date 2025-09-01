# Hybrid RAG Search System

## Overview

This project extends the Week 4 RAG pipeline with hybrid search capabilities, combining vector search (FAISS) with keyword search (SQLite FTS5 and BM25) to improve retrieval performance. The system provides multiple search methods and comprehensive evaluation tools.

## Features

### 🔍 Search Methods
- **Vector Search**: Semantic similarity using FAISS and sentence transformers
- **Keyword Search**: 
  - SQLite FTS5 for full-text search with BM25 ranking
  - Rank-BM25 library for traditional BM25 scoring
- **Hybrid Search**: 
  - Weighted score combination
  - Reciprocal Rank Fusion (RRF)
  - Configurable alpha parameter for vector/keyword balance

### 🗄️ Database Integration
- SQLite database with document metadata
- FTS5 virtual tables for efficient text search
- Automatic data loading from existing processed files

### 📊 Evaluation Framework
- 12 test queries with known relevant documents
- Metrics: Precision@k, Recall@k, Hit Rate@k
- Performance comparison across all methods
- Automated report generation with visualizations

### 🌐 API Endpoints
- `/search` - Original vector search
- `/hybrid_search` - New hybrid search endpoint
- Configurable parameters via GET/POST requests

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Basic Functionality
```bash
python test_hybrid.py
```

### 3. Run Full Evaluation
```bash
python run_evaluation.py
```

### 4. Start API Server
```bash
python main.py
```

## API Usage

### Hybrid Search Endpoint

**POST** `/hybrid_search`

```json
{
  "query": "machine learning neural networks",
  "k": 5,
  "alpha": 0.5,
  "keyword_method": "fts",
  "merge_method": "weighted"
}
```

**GET** `/hybrid_search?query=machine learning&k=3&alpha=0.6&keyword_method=bm25&merge_method=rrf`

#### Parameters
- `query` (string): Search query
- `k` (int, default=3): Number of results to return
- `alpha` (float, default=0.5): Weight for vector search (0.0 = keyword only, 1.0 = vector only)
- `keyword_method` (string, default="fts"): "fts" or "bm25"
- `merge_method` (string, default="weighted"): "weighted" or "rrf"

### Example Response

```json
{
  "query": "machine learning neural networks",
  "results": [
    {
      "rank": 1,
      "paper": "1.pdf",
      "chunk_id": 0,
      "score": 0.823,
      "chunk_text": "Machine learning approaches...",
      "search_type": "hybrid_weighted",
      "vector_score": 0.756,
      "keyword_score": 0.934
    }
  ],
  "total_results": 5,
  "method": "weighted_fts",
  "alpha": 0.5
}
```

## Architecture

### Core Components

1. **database.py**: SQLite database management with FTS5 integration
2. **rag_pipeline.py**: Extended RAG pipeline with keyword search
3. **hybrid_search.py**: HybridRetriever class implementing all search methods
4. **evaluation.py**: Comprehensive evaluation framework
5. **main.py**: FastAPI application with hybrid search endpoints

### Database Schema

```sql
-- Document metadata
CREATE TABLE documents (
    doc_id INTEGER PRIMARY KEY,
    title TEXT,
    authors TEXT,
    abstract TEXT,
    pdf_url TEXT,
    local_file TEXT,
    year INTEGER,
    keywords TEXT
);

-- Text chunks
CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER,
    chunk_index INTEGER,
    chunk_text TEXT,
    FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE doc_chunks_fts USING fts5(
    chunk_text,
    content='chunks',
    content_rowid='chunk_id'
);
```

## Search Method Details

### Vector Search
- Uses existing FAISS index with L2 distance
- Scores normalized to 0-1 range
- Based on sentence-transformers embeddings

### Keyword Search (FTS5)
- SQLite FTS5 with built-in BM25 ranking
- Automatic tokenization and stemming
- Supports complex query syntax

### Keyword Search (BM25)
- Rank-BM25 library implementation
- Customizable parameters (k1=1.2, b=0.75)
- Simple space-based tokenization

### Hybrid Methods

#### Weighted Combination
```python
hybrid_score = alpha * vector_score + (1 - alpha) * keyword_score
```

#### Reciprocal Rank Fusion (RRF)
```python
rrf_score = sum(1 / (k + rank_i) for rank_i in method_ranks)
```

## Evaluation Results

The evaluation framework tests 7 different search methods:
- vector
- keyword_fts
- keyword_bm25
- hybrid_weighted_0.3
- hybrid_weighted_0.5
- hybrid_weighted_0.7
- hybrid_rrf

### Test Queries

1. "neural networks machine learning"
2. "natural language processing NLP"
3. "parsing grammar syntactic analysis"
4. "speech recognition spoken language"
5. "machine translation translation systems"
6. "text segmentation discourse structure"
7. "word sense disambiguation lexical semantics"
8. "information retrieval IR text retrieval"
9. "dialogue systems conversational agents"
10. "part of speech tagging POS morphological analysis"
11. "probabilistic models statistical methods"
12. "corpus annotation linguistic annotation"

### Metrics

- **Precision@k**: Fraction of retrieved documents that are relevant
- **Recall@k**: Fraction of relevant documents that are retrieved
- **Hit Rate@k**: Binary metric (1 if any relevant document in top-k, 0 otherwise)

## File Structure

```
Week4/
├── database.py              # Database management
├── rag_pipeline.py          # Extended RAG pipeline
├── hybrid_search.py         # Hybrid search implementation
├── evaluation.py            # Evaluation framework
├── main.py                  # FastAPI application
├── run_evaluation.py        # Evaluation runner script
├── test_hybrid.py           # Basic functionality tests
├── requirements.txt         # Dependencies
├── HYBRID_SEARCH_README.md  # This documentation
├── document_index.db        # SQLite database (created)
├── evaluation_*.json/csv    # Evaluation results (created)
└── evaluation_*.png/md      # Evaluation reports (created)
```

## Usage Examples

### 1. Command Line Testing
```bash
# Test database functionality
python test_hybrid.py

# Run comprehensive evaluation
python run_evaluation.py
```

### 2. API Testing
```bash
# Start server
python main.py

# Test vector search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "k": 3}'

# Test hybrid search (weighted)
curl -X POST http://localhost:8000/hybrid_search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "k": 3, "alpha": 0.5, "merge_method": "weighted"}'

# Test hybrid search (RRF)
curl -X GET "http://localhost:8000/hybrid_search?query=neural networks&k=5&merge_method=rrf"
```

### 3. Python API Usage
```python
from rag_pipeline import RAGPipeline
from hybrid_search import HybridRetriever

# Initialize
pipeline = RAGPipeline()
pipeline.run_pipeline()
hybrid = HybridRetriever(pipeline)

# Vector search
vector_results = hybrid.vector_search("machine learning", k=3)

# Keyword search
keyword_results = hybrid.keyword_search_fts("neural networks", k=3)

# Hybrid search
hybrid_results = hybrid.hybrid_search(
    "deep learning", 
    k=5, 
    alpha=0.6, 
    merge_method="weighted"
)

# Compare methods
all_results = hybrid.compare_search_methods("AI research", k=3)
```

## Performance Considerations

### Indexing
- SQLite FTS5 index: ~50MB for 50 papers
- BM25 index: Built in memory, fast startup
- FAISS index: Existing implementation unchanged

### Search Speed
- Vector search: ~10ms per query
- FTS5 search: ~5ms per query
- BM25 search: ~20ms per query
- Hybrid search: Sum of component methods

### Memory Usage
- Base RAG pipeline: ~500MB
- Additional BM25 index: ~100MB
- SQLite database: ~50MB

## Customization

### Adding New Search Methods
1. Extend `HybridRetriever` class
2. Add method to evaluation framework
3. Update API endpoints if needed

### Modifying Scoring
- Adjust alpha parameter for weighted combination
- Customize BM25 parameters (k1, b)
- Implement custom merge functions

### Evaluation Queries
- Modify `_create_test_queries()` in `SearchEvaluator`
- Add domain-specific test cases
- Adjust relevance judgments

## Troubleshooting

### Common Issues

1. **Import errors**: Install dependencies with `pip install -r requirements.txt`
2. **Database errors**: Delete `document_index.db` and restart
3. **Memory issues**: Reduce chunk count or use smaller embedding model
4. **API startup**: Check port 8000 availability

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Profiling
```python
import time
start = time.time()
results = hybrid.hybrid_search(query, k=10)
print(f"Search took {time.time() - start:.2f}s")
```

## Future Improvements

- [ ] Query expansion and reformulation
- [ ] Learning-to-rank for score combination
- [ ] Semantic keyword matching
- [ ] Chunk-level relevance feedback
- [ ] Multi-modal search capabilities
- [ ] Real-time index updates
- [ ] Distributed search architecture

## References

- [SQLite FTS5 Documentation](https://www.sqlite.org/fts5.html)
- [Rank-BM25 Library](https://pypi.org/project/rank-bm25/)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
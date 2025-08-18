# RAG Pipeline for arXiv cs.CL Papers

A complete Retrieval-Augmented Generation (RAG) pipeline for searching through arXiv Computational Linguistics research papers. This system enables semantic search across 50 PDF papers using state-of-the-art embedding models and FAISS indexing.

## Features

- **PDF Text Extraction**: Automated text extraction from research papers using PyMuPDF
- **Smart Chunking**: Token-aware text chunking with configurable overlap
- **Dense Embeddings**: High-quality embeddings using sentence-transformers
- **Fast Search**: FAISS-powered similarity search
- **FastAPI Service**: RESTful API for integration
- **Interactive Demo**: Jupyter notebook for exploration

## Project Structure

```
├── PDFs/                   # Input PDF files (50 papers)
├── src/                    # Source code modules
│   ├── pdf_processor.py    # PDF text extraction
│   ├── chunker.py         # Text chunking with token counting
│   ├── embedder.py        # Embedding generation and FAISS indexing
│   ├── retriever.py       # Search and retrieval system
│   └── generate_report.py # Report generation utility
├── data/                   # Processed data and indices
├── outputs/               # Generated reports and results
├── main.py               # FastAPI service
├── rag_demo.ipynb       # Interactive demonstration
└── requirements.txt     # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# Step 1: Extract text from PDFs
cd src
python pdf_processor.py

# Step 2: Create text chunks
python chunker.py

# Step 3: Generate embeddings and build FAISS index
python embedder.py

# Step 4: Generate retrieval report
python generate_report.py
```

### 3. Start the FastAPI Service

```bash
# Return to root directory
cd ..
python main.py
```

The API will be available at `http://localhost:8000` with documentation at `http://localhost:8000/docs`.

### 4. Try the Interactive Demo

```bash
jupyter notebook rag_demo.ipynb
```

## API Usage

### Search Endpoint

```bash
# Basic search
curl "http://localhost:8000/search?q=transformer%20architecture&k=3"

# Search with context
curl "http://localhost:8000/search/context?q=attention%20mechanism&k=3&context_window=1"
```

### Example Response

```json
{
  "query": "transformer architecture",
  "results": [
    {
      "document_id": "1.pdf",
      "chunk_id": "1.pdf_chunk_45",
      "chunk_index": 45,
      "text": "The Transformer architecture revolutionized natural language processing...",
      "similarity_score": 0.8234,
      "token_count": 487,
      "rank": 1
    }
  ],
  "total_results": 3,
  "processing_time_ms": 23.45
}
```

## Configuration

### Chunking Parameters

Edit `src/chunker.py` to adjust:
- `chunk_size`: Target tokens per chunk (default: 512)
- `overlap_size`: Overlap between chunks (default: 50)

### Embedding Model

Edit `src/embedder.py` to change:
- `model_name`: Sentence transformer model (default: "all-MiniLM-L6-v2")

### Search Parameters

- `k`: Number of results to return (1-20)
- `context_window`: Additional context chunks (0-3)

## Generated Files

After running the pipeline, you'll have:

- `data/processed_documents.json`: Extracted text from all PDFs
- `data/chunks.json`: Processed text chunks with metadata
- `data/faiss_index.bin`: FAISS search index
- `data/chunk_metadata.pkl`: Chunk metadata for retrieval
- `data/embeddings.npy`: Raw embedding vectors
- `outputs/retrieval_report.md`: Performance analysis report

## Performance

- **Index Size**: ~2K-5K chunks (depending on paper length)
- **Search Speed**: <50ms for top-3 results
- **Memory Usage**: ~500MB for loaded index
- **Embedding Dimension**: 384 (all-MiniLM-L6-v2)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check and system status |
| `/search` | GET | Basic semantic search |
| `/search/context` | GET | Search with surrounding context |
| `/stats` | GET | Index and collection statistics |

## Troubleshooting

### Common Issues

1. **Missing FAISS index**: Run the pipeline steps in order
2. **Memory errors**: Reduce batch size in `embedder.py`
3. **Slow searches**: Check if index is properly loaded
4. **Empty results**: Verify PDF text extraction worked

### Dependencies

- Python 3.8+
- PyMuPDF for PDF processing
- sentence-transformers for embeddings
- FAISS for similarity search
- FastAPI for web service

## Example Queries

- "transformer architecture"
- "attention mechanism" 
- "BERT language model"
- "machine translation systems"
- "natural language understanding"

## License

This project is for educational and research purposes. Please respect the licenses of the original arXiv papers.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request
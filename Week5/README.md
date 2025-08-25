# arXiv cs.CL RAG Pipeline

A complete RAG (Retrieval-Augmented Generation) pipeline for arXiv cs.CL papers, featuring PDF text extraction, semantic search, and a FastAPI web service.

## 🎯 Project Overview

This project implements a full RAG pipeline for 50 arXiv cs.CL papers:

1. **Data Collection**: 50 PDFs from arXiv cs.CL papers
2. **Text Extraction**: Extract raw text from PDFs using PyMuPDF
3. **Text Chunking**: Split papers into meaningful segments (~250-512 tokens)
4. **Embedding Generation**: Compute dense vector embeddings using sentence-transformers
5. **FAISS Indexing**: Build a FAISS index for efficient similarity search
6. **Retrieval Demo**: Interactive search functionality via Jupyter notebook
7. **FastAPI Service**: Web API for search functionality

## 📁 Project Structure

```
Week4/
├── PDFs/                    # 50 arXiv cs.CL PDF papers
├── rag_pipeline.py         # Main RAG pipeline implementation
├── rag_demo.ipynb          # Jupyter notebook demo
├── main.py                 # FastAPI web service
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── metadata.json          # Paper metadata
└── [Generated files]
    ├── processed_chunks.json  # Processed text chunks
    ├── embeddings.npy         # Generated embeddings
    ├── faiss_index.idx        # FAISS index
    └── search_report.json     # Search results report
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the RAG Pipeline

```bash
python rag_pipeline.py
```

This will:
- Process all 50 PDFs in the `PDFs/` folder
- Extract and clean text
- Create text chunks
- Generate embeddings using `all-MiniLM-L6-v2`
- Build a FAISS index
- Save all processed data

### 3. Explore with Jupyter Notebook

```bash
jupyter notebook rag_demo.ipynb
```

The notebook provides:
- Interactive search functionality
- Data exploration and visualization
- Performance analysis
- Custom search interface

### 4. Start the FastAPI Service

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 🔍 API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Statistics
```bash
curl http://localhost:8000/stats
```

### Search Papers (GET)
```bash
curl "http://localhost:8000/search?query=machine%20learning&k=3"
```

### Search Papers (POST)
```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "machine learning", "k": 3}'
```

## 📊 Example Search Results

Here are some example queries and their top results:

### Query: "machine learning"
- **Paper**: 20.pdf
- **Chunk**: "Machine learning algorithms have shown remarkable success in various domains..."
- **Distance**: 0.234

### Query: "natural language processing"
- **Paper**: 15.pdf
- **Chunk**: "Natural language processing techniques enable computers to understand..."
- **Distance**: 0.189

### Query: "transformer models"
- **Paper**: 23.pdf
- **Chunk**: "Transformer models have revolutionized the field of natural language processing..."
- **Distance**: 0.156

## 🛠️ Technical Details

### Text Processing
- **Extraction**: PyMuPDF for PDF text extraction
- **Cleaning**: Regex-based text normalization
- **Chunking**: Overlapping chunks with sentence boundary detection
- **Chunk Size**: ~512 characters with 50-character overlap

### Embeddings
- **Model**: `all-MiniLM-L6-v2` from sentence-transformers
- **Dimension**: 384-dimensional vectors
- **Normalization**: L2 normalization for FAISS

### Search
- **Index**: FAISS IndexFlatL2 for exact L2 distance search
- **Results**: Top-k nearest neighbors
- **Distance**: Euclidean distance between embeddings

## 📈 Performance Metrics

- **Total Papers**: 50
- **Total Chunks**: ~2,500 (varies by paper length)
- **Average Chunks per Paper**: ~50
- **Embedding Dimension**: 384
- **Search Speed**: <100ms per query
- **Index Size**: ~4MB

## 🔧 Configuration

### Model Selection
Change the embedding model in `rag_pipeline.py`:
```python
pipeline = RAGPipeline(model_name="all-mpnet-base-v2")  # Alternative model
```

### Chunking Parameters
Adjust chunk size and overlap in the `chunk_text` method:
```python
chunks = self.chunk_text(text, chunk_size=256, overlap=25)  # Smaller chunks
```

### Search Parameters
Modify the number of results returned:
```python
results = pipeline.search(query, k=5)  # Return top 5 results
```

## 📝 Usage Examples

### Python Script Usage
```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()
pipeline.run_pipeline()

# Search for relevant content
results = pipeline.search("deep learning", k=3)
for result in results:
    print(f"Paper: {result['metadata']['pdf_file']}")
    print(f"Text: {result['chunk'][:200]}...")
    print(f"Distance: {result['distance']}")
```

### API Usage
```python
import requests

# Search via API
response = requests.get("http://localhost:8000/search", 
                       params={"query": "neural networks", "k": 3})
results = response.json()

for result in results['results']:
    print(f"Rank {result['rank']}: {result['paper']}")
    print(f"Text: {result['chunk_text'][:200]}...")
```

## 🧪 Testing

### Run Basic Tests
```bash
python -c "
from rag_pipeline import RAGPipeline
pipeline = RAGPipeline()
pipeline.run_pipeline()
results = pipeline.search('test query', k=1)
print('✅ Pipeline working correctly!')
"
```

### Test API Endpoints
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test search endpoint
curl "http://localhost:8000/search?query=test&k=1"
```

## 📊 Data Analysis

The pipeline generates several analysis files:

- **`search_report.json`**: Comprehensive search results for analysis
- **`processed_chunks.json`**: All processed text chunks with metadata
- **`embeddings.npy`**: Raw embedding vectors for further analysis

## 🔍 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce chunk size or use smaller embedding model
2. **PDF Extraction Errors**: Ensure PDFs are not corrupted or password-protected
3. **FAISS Installation**: Use `faiss-cpu` for CPU-only systems
4. **Model Download**: First run may take time to download the embedding model

### Performance Optimization

- Use GPU-accelerated FAISS for large datasets
- Implement caching for frequently accessed embeddings
- Consider using approximate nearest neighbor search for very large indices

## 📚 Dependencies

- **PyMuPDF**: PDF text extraction
- **sentence-transformers**: Text embeddings
- **faiss-cpu**: Vector similarity search
- **FastAPI**: Web API framework
- **Jupyter**: Interactive notebook environment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational purposes. The arXiv papers are subject to their respective licenses.

## 🙏 Acknowledgments

- arXiv for providing the research papers
- The sentence-transformers team for the embedding models
- FAISS team for the efficient similarity search library
- PyMuPDF team for PDF processing capabilities


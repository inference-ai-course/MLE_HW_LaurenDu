from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import sys
import os

# Add current directory to path to import our RAG pipeline
sys.path.append('.')

from rag_pipeline import RAGPipeline
from hybrid_search import HybridRetriever

# Initialize FastAPI app
app = FastAPI(
    title="arXiv cs.CL RAG Search API",
    description="A RAG (Retrieval-Augmented Generation) search API for arXiv cs.CL papers",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline and hybrid retriever
pipeline = None
hybrid_retriever = None

class SearchQuery(BaseModel):
    query: str
    k: Optional[int] = 3

class HybridSearchQuery(BaseModel):
    query: str
    k: Optional[int] = 3
    alpha: Optional[float] = 0.5
    keyword_method: Optional[str] = "fts"  # "fts" or "bm25"
    merge_method: Optional[str] = "weighted"  # "weighted" or "rrf"

class SearchResult(BaseModel):
    rank: int
    paper: str
    chunk_id: int
    distance: float
    chunk_text: str

class HybridSearchResult(BaseModel):
    rank: int
    paper: str
    chunk_id: int
    score: float
    chunk_text: str
    search_type: str
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int

class HybridSearchResponse(BaseModel):
    query: str
    results: List[HybridSearchResult]
    total_results: int
    method: str
    alpha: Optional[float] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup"""
    global pipeline, hybrid_retriever
    try:
        pipeline = RAGPipeline()
        pipeline.run_pipeline()
        
        # Initialize hybrid retriever
        hybrid_retriever = HybridRetriever(pipeline)
        
        print("✅ RAG pipeline initialized successfully!")
        print(f"📚 Loaded {len(pipeline.chunks)} chunks from {len(set(m['pdf_file'] for m in pipeline.chunk_metadata))} papers")
        print("🔍 Hybrid search capabilities ready!")
    except Exception as e:
        print(f"❌ Error initializing RAG pipeline: {e}")
        raise e

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "arXiv cs.CL RAG Search API",
        "version": "1.0.0",
        "endpoints": {
            "/search": "POST - Search for relevant paper chunks (vector search)",
            "/hybrid_search": "POST - Hybrid search combining vector and keyword search",
            "/health": "GET - Health check",
            "/stats": "GET - Get pipeline statistics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "hybrid_search_loaded": hybrid_retriever is not None,
        "chunks_loaded": len(pipeline.chunks) if pipeline else 0,
        "papers_loaded": len(set(m['pdf_file'] for m in pipeline.chunk_metadata)) if pipeline else 0
    }

@app.get("/stats")
async def get_stats():
    """Get pipeline statistics"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    # Calculate statistics
    paper_stats = {}
    for metadata in pipeline.chunk_metadata:
        paper_id = metadata['pdf_file']
        if paper_id not in paper_stats:
            paper_stats[paper_id] = 0
        paper_stats[paper_id] += 1
    
    chunk_lengths = [len(chunk) for chunk in pipeline.chunks]
    
    return {
        "total_papers": len(paper_stats),
        "total_chunks": len(pipeline.chunks),
        "average_chunks_per_paper": len(pipeline.chunks) / len(paper_stats),
        "embedding_dimension": pipeline.embedding_dim,
        "chunk_length_stats": {
            "min": min(chunk_lengths),
            "max": max(chunk_lengths),
            "mean": sum(chunk_lengths) / len(chunk_lengths)
        },
        "papers_with_most_chunks": sorted(paper_stats.items(), key=lambda x: x[1], reverse=True)[:5]
    }

@app.post("/search", response_model=SearchResponse)
async def search_papers(search_query: SearchQuery):
    """
    Search for relevant paper chunks based on a query
    
    Args:
        search_query: The search query and number of results to return
        
    Returns:
        SearchResponse with ranked results
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Perform search
        results = pipeline.search(search_query.query, search_query.k)
        
        # Convert to response format
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                rank=result['rank'],
                paper=result['metadata']['pdf_file'],
                chunk_id=result['metadata']['chunk_id'],
                distance=result['distance'],
                chunk_text=result['chunk']
            ))
        
        return SearchResponse(
            query=search_query.query,
            results=search_results,
            total_results=len(search_results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/search")
async def search_get(query: str, k: int = 3):
    """
    GET endpoint for search (alternative to POST)
    
    Args:
        query: Search query
        k: Number of results to return (default: 3)
        
    Returns:
        Search results
    """
    search_query = SearchQuery(query=query, k=k)
    return await search_papers(search_query)

@app.post("/hybrid_search", response_model=HybridSearchResponse)
async def hybrid_search_papers(search_query: HybridSearchQuery):
    """
    Hybrid search endpoint combining vector and keyword search
    
    Args:
        search_query: The hybrid search query with parameters
        
    Returns:
        HybridSearchResponse with ranked results
    """
    if pipeline is None or hybrid_retriever is None:
        raise HTTPException(status_code=503, detail="Hybrid search not initialized")
    
    try:
        # Validate parameters
        if search_query.keyword_method not in ["fts", "bm25"]:
            raise HTTPException(status_code=400, detail="keyword_method must be 'fts' or 'bm25'")
        
        if search_query.merge_method not in ["weighted", "rrf"]:
            raise HTTPException(status_code=400, detail="merge_method must be 'weighted' or 'rrf'")
        
        if not 0 <= search_query.alpha <= 1:
            raise HTTPException(status_code=400, detail="alpha must be between 0 and 1")
        
        # Perform hybrid search
        results = hybrid_retriever.hybrid_search(
            query=search_query.query,
            k=search_query.k,
            alpha=search_query.alpha,
            keyword_method=search_query.keyword_method,
            merge_method=search_query.merge_method
        )
        
        # Convert to response format
        hybrid_results = []
        for result in results:
            hybrid_results.append(HybridSearchResult(
                rank=result['rank'],
                paper=result['metadata']['pdf_file'],
                chunk_id=result['metadata']['chunk_id'],
                score=result['score'],
                chunk_text=result['chunk'],
                search_type=result['search_type'],
                vector_score=result.get('vector_score'),
                keyword_score=result.get('keyword_score')
            ))
        
        method_desc = f"{search_query.merge_method}_{search_query.keyword_method}"
        
        return HybridSearchResponse(
            query=search_query.query,
            results=hybrid_results,
            total_results=len(hybrid_results),
            method=method_desc,
            alpha=search_query.alpha if search_query.merge_method == "weighted" else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")

@app.get("/hybrid_search")
async def hybrid_search_get(
    query: str, 
    k: int = 3, 
    alpha: float = 0.5,
    keyword_method: str = "fts",
    merge_method: str = "weighted"
):
    """
    GET endpoint for hybrid search (alternative to POST)
    
    Args:
        query: Search query
        k: Number of results to return (default: 3)
        alpha: Weight for vector search (default: 0.5)
        keyword_method: Keyword search method - 'fts' or 'bm25' (default: 'fts')
        merge_method: Score merging method - 'weighted' or 'rrf' (default: 'weighted')
        
    Returns:
        Hybrid search results
    """
    search_query = HybridSearchQuery(
        query=query, 
        k=k, 
        alpha=alpha,
        keyword_method=keyword_method,
        merge_method=merge_method
    )
    return await hybrid_search_papers(search_query)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


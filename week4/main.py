from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import os
import sys

sys.path.append('./src')
from retriever import RAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="arXiv cs.CL RAG API",
    description="Retrieval-Augmented Generation API for searching arXiv Computational Linguistics papers",
    version="1.0.0"
)

retriever = None

@app.on_event("startup")
async def startup_event():
    global retriever
    try:
        logger.info("Initializing RAG retriever...")
        retriever = RAGRetriever(data_directory="./data")
        logger.info("RAG retriever initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {str(e)}")
        raise

class SearchResponse(BaseModel):
    query: str
    results: List[Dict]
    total_results: int
    processing_time_ms: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    index_size: Optional[int] = None

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "arXiv cs.CL RAG API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search?q=your_query&k=3",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    try:
        index_size = retriever.index.ntotal if retriever.index else 0
        return HealthResponse(
            status="healthy",
            message="RAG system is operational",
            index_size=index_size
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/search", response_model=SearchResponse)
async def search_papers(
    q: str = Query(..., description="Search query", min_length=1),
    k: int = Query(3, description="Number of results to return", ge=1, le=20)
):
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        import time
        start_time = time.time()
        
        results = retriever.search(q.strip(), k=k)
        
        processing_time = (time.time() - start_time) * 1000
        
        formatted_results = []
        for result in results:
            formatted_result = {
                "document_id": result["document_id"],
                "chunk_id": result["chunk_id"],
                "chunk_index": result["chunk_index"],
                "text": result["text"],
                "similarity_score": round(result["similarity_score"], 4),
                "token_count": result["token_count"],
                "rank": result["rank"]
            }
            formatted_results.append(formatted_result)
        
        return SearchResponse(
            query=q,
            results=formatted_results,
            total_results=len(formatted_results),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Search failed for query '{q}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/search/context", response_model=SearchResponse)
async def search_papers_with_context(
    q: str = Query(..., description="Search query", min_length=1),
    k: int = Query(3, description="Number of results to return", ge=1, le=20),
    context_window: int = Query(1, description="Context window size", ge=0, le=3)
):
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        import time
        start_time = time.time()
        
        results = retriever.search_with_context(q.strip(), k=k, context_window=context_window)
        
        processing_time = (time.time() - start_time) * 1000
        
        formatted_results = []
        for result in results:
            formatted_result = {
                "document_id": result["document_id"],
                "chunk_id": result["chunk_id"],
                "chunk_index": result["chunk_index"],
                "text": result["text"],
                "context_text": result.get("context_text", ""),
                "context_chunks": result.get("context_chunks", 1),
                "similarity_score": round(result["similarity_score"], 4),
                "token_count": result["token_count"],
                "rank": result["rank"]
            }
            formatted_results.append(formatted_result)
        
        return SearchResponse(
            query=q,
            results=formatted_results,
            total_results=len(formatted_results),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Context search failed for query '{q}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Context search failed: {str(e)}")

@app.get("/stats")
async def get_statistics():
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    try:
        index_size = retriever.index.ntotal if retriever.index else 0
        metadata_size = len(retriever.chunk_metadata) if retriever.chunk_metadata else 0
        
        unique_docs = set()
        if retriever.chunk_metadata:
            unique_docs = {chunk["document_id"] for chunk in retriever.chunk_metadata}
        
        return {
            "total_chunks": index_size,
            "total_documents": len(unique_docs),
            "average_chunks_per_document": round(index_size / len(unique_docs), 1) if unique_docs else 0,
            "index_size": index_size,
            "metadata_size": metadata_size
        }
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    if not os.path.exists("./data/faiss_index.bin"):
        print("❌ FAISS index not found!")
        print("Please run the pipeline first:")
        print("1. cd src && python pdf_processor.py")
        print("2. python chunker.py")
        print("3. python embedder.py")
        sys.exit(1)
    
    print("🚀 Starting arXiv cs.CL RAG API...")
    print("📖 API Documentation available at: http://localhost:8000/docs")
    print("🔍 Example search: http://localhost:8000/search?q=transformer&k=3")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
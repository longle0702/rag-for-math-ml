"""FastAPI backend for RAG system."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json

from app.rag_pipeline import get_pipeline
from app.embeddings import get_embedding_stats

# Models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class SourceCitation(BaseModel):
    source: str
    page: int
    relevance: float
    preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceCitation]
    cost: float


class StatsResponse(BaseModel):
    embedding_stats: dict
    chat_stats: dict
    index_info: dict


# Initialize FastAPI app
app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API for Math ML",
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


# Routes

@app.on_event("startup")
async def startup():
    """Initialize pipeline on startup."""
    pipeline = get_pipeline()
    try:
        pipeline.load_index()
    except ValueError:
        # Build index if it doesn't exist
        print("Building new index...")
        pipeline.build_index()


@app.get("/health")
async def health():
    """Health check endpoint."""
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "index_loaded": pipeline.loaded,
        "num_chunks": len(pipeline.chunks) if pipeline.loaded else 0
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query and return answer with sources.
    
    Args:
        request: Query request with question
    
    Returns:
        QueryResponse with answer and sources
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    pipeline = get_pipeline()
    
    if not pipeline.loaded:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    
    try:
        result = pipeline.answer_question(request.question, top_k=request.top_k or 5)
        
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[SourceCitation(**source) for source in result["sources"]],
            cost=result["cost"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics and cost tracking."""
    pipeline = get_pipeline()
    embedding_stats = get_embedding_stats()
    chat_stats = pipeline.get_chat_stats()
    
    return StatsResponse(
        embedding_stats=embedding_stats,
        chat_stats=chat_stats,
        index_info={
            "num_chunks": len(pipeline.chunks),
            "index_loaded": pipeline.loaded
        }
    )


@app.get("/evaluate")
async def evaluate_questions():
    """
    Evaluate system on questions.json.
    
    Returns:
        Evaluation results
    """
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from app.config import QUESTIONS_PATH
    
    with open(QUESTIONS_PATH, 'r') as f:
        questions = json.load(f)
    
    pipeline = get_pipeline()
    
    results = []
    for q in questions[:10]:  # Limit to first 10 for demo
        result = pipeline.answer_question(q["question"])
        results.append({
            "question": q["question"],
            "expected_answer": q.get("expected_answer", "N/A"),
            "generated_answer": result["answer"],
            "sources": result["sources"],
            "cost": result["cost"]
        })
    
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

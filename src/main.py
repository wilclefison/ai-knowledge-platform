from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(
    title="AI Knowledge Platform API",
    description="Enterprise RAG Engine with Hybrid Search and Evals",
    version="0.1.0"
)

class HealthResponse(BaseModel):
    status: str
    version: str

class QueryRequest(BaseModel):
    query: str = Field(..., example="Quais são as cláusulas de rescisão contratual?")
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = Field(default=True)

class DocumentCitation(BaseModel):
    document_name: str
    page_number: int
    score: float
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[DocumentCitation]
    latency_ms: float
    model: str

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_knowledge_base(payload: QueryRequest):
    # Stub inicial estruturado - será implementado com pipeline híbrido
    return {
        "answer": "O sistema está inicializado e pronto para indexação e consultas estruturadas.",
        "citations": [
            {
                "document_name": "manual_empresa.pdf",
                "page_number": 1,
                "score": 0.98,
                "snippet": "Exemplo de trecho recuperado via Hybrid Search + Reranker."
            }
        ],
        "latency_ms": 120.5,
        "model": "gpt-4o-mini"
    }

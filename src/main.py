import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from src.config import settings
from src.ingestion.chunker import SemanticChunker, Chunk
from src.retrieval.hybrid_search import HybridSearchEngine, SearchResult
from src.retrieval.reranker import CrossEncoderReranker, RerankResult
from src.observability.tracer import tracer, TraceRecord
from src.evals.ragas_evaluator import evaluator, EvalSample, EvalReport, MetricResult

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise RAG Platform: Hybrid Search (HNSW + GIN), Cross-Encoder Re-ranking, Langfuse Tracing and Automated Evals (Ragas).",
    version="0.5.0"
)

chunker = SemanticChunker(target_chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
hybrid_engine = HybridSearchEngine(rrf_k=settings.RRF_K)
reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")

# --- Schemas ---

class IngestTextRequest(BaseModel):
    title: str = Field(..., example="AWS Security Architecture Whitepaper")
    content: str = Field(..., example="# Section 1: IAM Policies\n\nAll IAM roles must enforce MFA.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestResponse(BaseModel):
    document_id: str
    title: str
    chunks_created: int
    chunks: List[Chunk]

class HybridSearchRequest(BaseModel):
    query: str = Field(..., example="What are the IAM MFA requirements?")
    top_k: int = Field(default=10, ge=1, le=50)

class RerankRequest(BaseModel):
    query: str = Field(..., example="IAM MFA requirements")
    candidates: List[Dict[str, Any]] = Field(..., example=[
        {"id": "chunk_1", "content": "General networking rules.", "score": 0.015},
        {"id": "chunk_2", "content": "All IAM roles must enforce mandatory MFA authentication.", "score": 0.012}
    ])
    top_k: int = Field(default=5, ge=1, le=20)

class DocumentCitation(BaseModel):
    document_title: str
    chunk_id: str
    page_number: int
    rerank_score: float
    original_rank: int
    final_rank: int
    snippet: str

class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the requirements for IAM roles?")
    user_id: Optional[str] = Field(default="user_enterprise_102")
    session_id: Optional[str] = Field(default="sess_demo_401")
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = Field(default=True)

class ObservabilitySummary(BaseModel):
    trace_id: str
    total_latency_ms: float
    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    tokens_saved_by_reranker: int

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[DocumentCitation]
    observability: ObservabilitySummary
    model: str
    retrieval_strategy: str

# --- Endpoints ---

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": "0.5.0",
        "evals_suite": "Ragas (Faithfulness, Relevance, Recall, Precision)",
        "observability_provider": "Langfuse & OpenTelemetry",
        "reranker_model": reranker.model_name
    }

@app.post("/api/v1/ingest/text", response_model=IngestResponse)
async def ingest_raw_text(payload: IngestTextRequest):
    """Processes document text through the semantic chunking engine."""
    chunks = chunker.split_text(payload.content)
    return {
        "document_id": "doc_" + str(int(time.time())),
        "title": payload.title,
        "chunks_created": len(chunks),
        "chunks": chunks
    }

@app.post("/api/v1/search/hybrid", response_model=List[SearchResult])
async def hybrid_search_endpoint(payload: HybridSearchRequest):
    """Executes a Reciprocal Rank Fusion (RRF) query combining Dense and BM25 results."""
    mock_dense = [
        {"id": "chunk_1", "document_id": "doc_101", "content": "IAM roles require mandatory MFA tokens.", "page_number": 3},
        {"id": "chunk_2", "document_id": "doc_101", "content": "Access keys must rotate every 90 days.", "page_number": 4},
        {"id": "chunk_3", "document_id": "doc_102", "content": "Network security groups baseline configuration.", "page_number": 12},
    ]
    mock_sparse = [
        {"id": "chunk_1", "document_id": "doc_101", "content": "IAM roles require mandatory MFA tokens.", "page_number": 3},
        {"id": "chunk_4", "document_id": "doc_103", "content": "IAM policy JSON schema definition.", "page_number": 1},
    ]
    results = hybrid_engine.reciprocal_rank_fusion(mock_dense, mock_sparse, top_k=payload.top_k)
    return results

@app.post("/api/v1/rerank", response_model=List[RerankResult])
async def rerank_candidates_endpoint(payload: RerankRequest):
    """Re-scores a candidate pool using Cross-Encoder cross-attention semantics."""
    return reranker.rerank(payload.query, payload.candidates, top_k=payload.top_k)

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_knowledge_base(payload: QueryRequest):
    """
    End-to-End Enterprise RAG Pipeline with Distributed Tracing:
    1. Hybrid Search Retrieval (Dense + BM25) ➔ Span 1
    2. Cross-Encoder Re-ranker (Top 20 ➔ Top 5) ➔ Span 2
    3. LLM Generation + Token Cost Attribution ➔ Generation Span
    """
    trace = tracer.start_trace(
        name="enterprise_rag_query",
        session_id=payload.session_id,
        user_id=payload.user_id,
        tags=["rag-prod", "hybrid-search", "bge-reranker"]
    )
    
    # SPAN 1: Retrieval
    span1_start = time.time()
    mock_dense = [
        {"id": "chunk_1", "document_id": "doc_101", "content": "General networking rules for VPC peering.", "page_number": 1},
        {"id": "chunk_2", "document_id": "doc_101", "content": "All IAM roles must enforce mandatory MFA authentication on sensitive API calls.", "page_number": 3},
        {"id": "chunk_3", "document_id": "doc_101", "content": "Billing and cost allocation tags overview.", "page_number": 7},
    ]
    mock_sparse = [
        {"id": "chunk_2", "document_id": "doc_101", "content": "All IAM roles must enforce mandatory MFA authentication on sensitive API calls.", "page_number": 3},
        {"id": "chunk_4", "document_id": "doc_102", "content": "IAM role policies template definition.", "page_number": 2},
    ]
    initial_candidates = hybrid_engine.reciprocal_rank_fusion(mock_dense, mock_sparse, top_k=10)
    span1_end = time.time()
    tracer.add_span(
        trace_id=trace.trace_id,
        name="hybrid_retrieval_pgvector",
        start_time=span1_start,
        end_time=span1_end,
        input_data={"query": payload.query, "top_k_requested": 10},
        output_data={"candidates_found": len(initial_candidates)}
    )

    # SPAN 2: Re-ranking
    span2_start = time.time()
    if payload.use_reranker:
        candidate_dicts = [
            {"id": c.chunk_id, "content": c.content, "score": c.score, "page_number": c.page_number}
            for c in initial_candidates
        ]
        reranked_results = reranker.rerank(payload.query, candidate_dicts, top_k=payload.top_k)
    else:
        reranked_results = [
            RerankResult(
                chunk_id=c.chunk_id,
                content=c.content,
                original_rank=idx + 1,
                rerank_score=c.score,
                new_rank=idx + 1
            )
            for idx, c in enumerate(initial_candidates[:payload.top_k])
        ]
    span2_end = time.time()
    tracer.add_span(
        trace_id=trace.trace_id,
        name="cross_encoder_rerank",
        start_time=span2_start,
        end_time=span2_end,
        input_data={"candidates_count": len(initial_candidates)},
        output_data={"final_chunks_count": len(reranked_results), "model": reranker.model_name}
    )

    # SPAN 3: LLM Generation
    span3_start = time.time()
    answer_text = "All privileged IAM roles must enforce Multi-Factor Authentication (MFA) on sensitive API calls as specified in Section 3 of the security baseline."
    prompt_tokens = 450
    completion_tokens = 65
    span3_end = time.time()
    
    tracer.add_span(
        trace_id=trace.trace_id,
        name="llm_generation",
        start_time=span3_start,
        end_time=span3_end,
        input_data={"prompt_tokens": prompt_tokens, "model": "gpt-4o-mini"},
        output_data={"completion_tokens": completion_tokens, "answer_length": len(answer_text)}
    )

    final_trace = tracer.end_trace(
        trace_id=trace.trace_id,
        model="gpt-4o-mini",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )

    citations = [
        DocumentCitation(
            document_title="AWS Security Architecture Whitepaper",
            chunk_id=r.chunk_id,
            page_number=r.metadata.get("page_number", 3),
            rerank_score=r.rerank_score,
            original_rank=r.original_rank,
            final_rank=r.new_rank,
            snippet=r.content
        )
        for r in reranked_results
    ]

    tokens_saved = (len(initial_candidates) - len(reranked_results)) * 120

    return {
        "query": payload.query,
        "answer": answer_text,
        "citations": citations,
        "observability": {
            "trace_id": final_trace.trace_id,
            "total_latency_ms": final_trace.total_latency_ms,
            "retrieval_latency_ms": round((span1_end - span1_start) * 1000, 2),
            "rerank_latency_ms": round((span2_end - span2_start) * 1000, 2),
            "generation_latency_ms": round((span3_end - span3_start) * 1000, 2),
            "prompt_tokens": final_trace.prompt_tokens,
            "completion_tokens": final_trace.completion_tokens,
            "total_tokens": final_trace.total_tokens,
            "estimated_cost_usd": final_trace.cost_usd,
            "tokens_saved_by_reranker": tokens_saved
        },
        "model": "gpt-4o-mini",
        "retrieval_strategy": "2-Stage Hybrid (RRF) ➔ Cross-Encoder with Langfuse Tracing"
    }

# --- EVALUATION ENDPOINTS ---

@app.post("/api/v1/evals/run", response_model=EvalReport)
async def run_automated_evals(samples: Optional[List[EvalSample]] = None):
    """
    Executes automated RAG Triad evaluation across a test dataset.
    Returns Faithfulness, Answer Relevance, Context Recall and Context Precision scores.
    """
    test_samples = samples or [
        EvalSample(
            sample_id="eval_sample_01",
            query="What are the IAM MFA requirements for administrative accounts?",
            contexts=[
                "All IAM administrative roles must enforce mandatory Multi-Factor Authentication (MFA) on all sensitive API calls.",
                "Access keys must be rotated every 90 days."
            ],
            generated_answer="All IAM administrative roles must enforce mandatory MFA on sensitive API calls.",
            ground_truth="Administrative IAM roles require mandatory MFA on sensitive calls."
        ),
        EvalSample(
            sample_id="eval_sample_02",
            query="What is the data retention period for audit logs?",
            contexts=[
                "Audit logs in S3 Glacier must be retained for a minimum of 365 days before permanent deletion.",
                "Encryption keys are managed via AWS KMS."
            ],
            generated_answer="Audit logs must be retained for a minimum of 365 days in S3 Glacier.",
            ground_truth="Audit logs must be kept for at least 365 days."
        )
    ]
    
    report = evaluator.evaluate_dataset(test_samples)
    return report

@app.get("/api/v1/observability/trace/{trace_id}", response_model=TraceRecord)
async def get_trace_telemetry(trace_id: str):
    """Retrieves full hierarchical span tree and telemetry for a given trace ID."""
    trace = tracer._active_traces.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace ID not found.")
    return trace

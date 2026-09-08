import time
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from src.config import settings
from src.ingestion.chunker import SemanticChunker, Chunk
from src.ingestion.parent_retriever import parent_retriever, ParentDocument, HierarchicalSearchResult
from src.retrieval.hybrid_search import HybridSearchEngine, SearchResult
from src.retrieval.reranker import CrossEncoderReranker, RerankResult
from src.retrieval.compressor import compressor, CompressedChunk
from src.retrieval.self_query import self_query_engine, ParsedSelfQuery
from src.observability.tracer import tracer, TraceRecord
from src.evals.ragas_evaluator import evaluator, EvalSample, EvalReport, MetricResult
from src.db.security import security_engine, TenantContext, ClearanceLevel

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise RAG Platform: Self-Querying & Dynamic SQL Filters, Parent-Document Indexing, Multi-Tenant RLS Security, Hybrid Search, Cross-Encoder Re-ranking, Contextual Compression, Langfuse Tracing and Automated Evals.",
    version="0.9.0"
)

chunker = SemanticChunker(target_chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
hybrid_engine = HybridSearchEngine(rrf_k=settings.RRF_K)
reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")

# --- Schemas ---

class IngestTextRequest(BaseModel):
    tenant_id: str = Field(default="tenant_acme_corp", example="tenant_acme_corp")
    title: str = Field(..., example="AWS Security Architecture Whitepaper")
    content: str = Field(..., example="# Section 1: IAM Policies\n\nAll IAM roles must enforce MFA.")
    clearance: ClearanceLevel = Field(default=ClearanceLevel.INTERNAL)
    allowed_roles: List[str] = Field(default_factory=lambda: ["public"], example=["engineering", "finance"])
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestResponse(BaseModel):
    document_id: str
    tenant_id: str
    title: str
    clearance: ClearanceLevel
    chunks_created: int
    chunks: List[Chunk]

class SelfQueryRequest(BaseModel):
    query: str = Field(..., example="AWS IAM whitepapers in 2024 with rating > 4.5 for engineering")

class DocumentCitation(BaseModel):
    document_title: str
    chunk_id: str
    tenant_id: str
    clearance: str
    page_number: int
    rerank_score: float
    original_rank: int
    final_rank: int
    original_length: int
    compressed_length: int
    snippet: str

class QueryRequest(BaseModel):
    tenant_context: TenantContext
    query: str = Field(..., example="What are the IAM requirements for security in 2024?")
    session_id: Optional[str] = Field(default="sess_prod_901")
    top_k: int = Field(default=5, ge=1, le=20)
    use_self_query: bool = Field(default=True)
    use_reranker: bool = Field(default=True)
    use_compression: bool = Field(default=True)

class ObservabilitySummary(BaseModel):
    trace_id: str
    total_latency_ms: float
    self_query_parse_latency_ms: float
    security_filter_latency_ms: float
    retrieval_latency_ms: float
    rerank_latency_ms: float
    compression_latency_ms: float
    generation_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    tokens_saved_by_reranker: int
    tokens_saved_by_compression: int
    total_tokens_saved: int

class QueryResponse(BaseModel):
    tenant_id: str
    query: str
    parsed_self_query: Optional[ParsedSelfQuery] = None
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
        "version": "0.9.0",
        "features": [
            "Self-Querying & Dynamic SQL Filters",
            "Parent-Document Retriever (Small-to-Big)",
            "PostgreSQL Multi-Tenant RLS",
            "Hybrid Search (HNSW+GIN with RRF)",
            "Cross-Encoder Reranker",
            "Contextual Compression",
            "Langfuse Tracing",
            "Ragas Evals"
        ]
    }

@app.post("/api/v1/search/self-query", response_model=ParsedSelfQuery)
async def parse_self_query_endpoint(payload: SelfQueryRequest):
    """Decomposes natural language query into clean semantic vector query + structured SQL filters."""
    return self_query_engine.parse_query(payload.query)

@app.post("/api/v1/ingest/text", response_model=IngestResponse)
async def ingest_raw_text(payload: IngestTextRequest):
    chunks = chunker.split_text(payload.content)
    return {
        "document_id": "doc_" + str(int(time.time())),
        "tenant_id": payload.tenant_id,
        "title": payload.title,
        "clearance": payload.clearance,
        "chunks_created": len(chunks),
        "chunks": chunks
    }

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_knowledge_base(payload: QueryRequest):
    trace = tracer.start_trace(
        name="enterprise_rag_query",
        session_id=payload.session_id,
        user_id=payload.tenant_context.user_id,
        tags=["rag-prod", f"tenant:{payload.tenant_context.tenant_id}", "self-query-pipeline"]
    )

    # SPAN 1: Self-Querying Parser
    sq_start = time.time()
    parsed_sq = None
    search_query = payload.query
    if payload.use_self_query:
        parsed_sq = self_query_engine.parse_query(payload.query)
        search_query = parsed_sq.semantic_query
    sq_end = time.time()
    tracer.add_span(
        trace_id=trace.trace_id,
        name="self_query_parser",
        start_time=sq_start,
        end_time=sq_end,
        input_data={"raw_query": payload.query},
        output_data={"semantic_query": search_query, "filters_count": len(parsed_sq.filters) if parsed_sq else 0}
    )

    # SPAN 2: Security & Tenant Isolation Filter
    sec_start = time.time()
    mock_raw_candidates = [
        {
            "id": "chunk_1",
            "tenant_id": payload.tenant_context.tenant_id,
            "clearance": "INTERNAL",
            "allowed_roles": ["engineering", "viewer", "public"],
            "content": "AWS IAM Architecture Guide (2024 Edition). All IAM administrative roles must enforce mandatory MFA authentication on sensitive API calls.",
            "page_number": 3,
            "metadata": {"year": 2024, "department": "engineering", "rating": 4.8}
        },
        {
            "id": "chunk_2",
            "tenant_id": payload.tenant_context.tenant_id,
            "clearance": "INTERNAL",
            "allowed_roles": ["public"],
            "content": "Legacy IAM Guide (2020 Edition). General user password rules.",
            "page_number": 1,
            "metadata": {"year": 2020, "department": "engineering", "rating": 3.2}
        }
    ]
    authorized_raw = security_engine.filter_candidates(payload.tenant_context, mock_raw_candidates)
    
    # Apply Structured Metadata Filters (e.g., Year = 2024)
    if parsed_sq and parsed_sq.filters:
        for f in parsed_sq.filters:
            if f.field == "year":
                authorized_raw = [c for c in authorized_raw if c.get("metadata", {}).get("year") == f.value]
            elif f.field == "rating" and f.operator == "gt":
                authorized_raw = [c for c in authorized_raw if c.get("metadata", {}).get("rating", 0) > f.value]

    sec_end = time.time()
    tracer.add_span(
        trace_id=trace.trace_id,
        name="security_and_metadata_filter",
        start_time=sec_start,
        end_time=sec_end,
        input_data={"tenant_id": payload.tenant_context.tenant_id},
        output_data={"filtered_candidates": len(authorized_raw)}
    )

    # SPAN 3: Hybrid Retrieval
    retrieval_start = time.time()
    mock_sparse = [
        {"id": "chunk_1", "tenant_id": payload.tenant_context.tenant_id, "clearance": "INTERNAL", "allowed_roles": ["public"], "content": "AWS IAM Architecture Guide (2024 Edition). All IAM administrative roles must enforce mandatory MFA authentication on sensitive API calls.", "page_number": 3},
    ]
    initial_candidates = hybrid_engine.reciprocal_rank_fusion(authorized_raw, mock_sparse, top_k=10)
    retrieval_end = time.time()
    tracer.add_span(
        trace_id=trace.trace_id,
        name="hybrid_retrieval_pgvector",
        start_time=retrieval_start,
        end_time=retrieval_end,
        input_data={"semantic_query": search_query},
        output_data={"candidates_found": len(initial_candidates)}
    )

    # SPAN 4: Cross-Encoder Re-ranking
    rerank_start = time.time()
    if payload.use_reranker:
        candidate_dicts = [
            {"id": c.chunk_id, "content": c.content, "score": c.score, "page_number": c.page_number, "tenant_id": payload.tenant_context.tenant_id, "clearance": "INTERNAL"}
            for c in initial_candidates
        ]
        reranked_results = reranker.rerank(search_query, candidate_dicts, top_k=payload.top_k)
    else:
        reranked_results = [
            RerankResult(
                chunk_id=c.chunk_id,
                content=c.content,
                original_rank=idx + 1,
                rerank_score=c.score,
                new_rank=idx + 1,
                metadata={"tenant_id": payload.tenant_context.tenant_id, "clearance": "INTERNAL"}
            )
            for idx, c in enumerate(initial_candidates[:payload.top_k])
        ]
    rerank_end = time.time()
    tracer.add_span(
        trace_id=trace.trace_id,
        name="cross_encoder_rerank",
        start_time=rerank_start,
        end_time=rerank_end,
        input_data={"candidates": len(initial_candidates)},
        output_data={"reranked": len(reranked_results)}
    )

    # SPAN 5: Contextual Compression
    comp_start = time.time()
    compressed_items = []
    tokens_saved_by_comp = 0
    
    if payload.use_compression:
        for r in reranked_results:
            c_res = compressor.compress_chunk(search_query, r.chunk_id, r.content)
            compressed_items.append((r, c_res))
            tokens_saved_by_comp += max(0, c_res.original_tokens - c_res.compressed_tokens)
    else:
        for r in reranked_results:
            orig_t = compressor._estimate_tokens(r.content)
            c_res = CompressedChunk(
                chunk_id=r.chunk_id,
                original_text=r.content,
                compressed_text=r.content,
                original_tokens=orig_t,
                compressed_tokens=orig_t,
                compression_ratio=1.0,
                retained_sentences_count=1,
                total_sentences_count=1
            )
            compressed_items.append((r, c_res))
            
    comp_end = time.time()
    tracer.add_span(
        trace_id=trace.trace_id,
        name="contextual_compression",
        start_time=comp_start,
        end_time=comp_end,
        input_data={"chunks": len(reranked_results)},
        output_data={"tokens_saved": tokens_saved_by_comp}
    )

    # SPAN 6: LLM Generation
    gen_start = time.time()
    answer_text = "According to the 2024 AWS Security Architecture Guide, all administrative IAM roles must enforce Multi-Factor Authentication (MFA) on sensitive API calls."
    prompt_tokens = 220
    completion_tokens = 65
    gen_end = time.time()
    
    tracer.add_span(
        trace_id=trace.trace_id,
        name="llm_generation",
        start_time=gen_start,
        end_time=gen_end,
        input_data={"prompt_tokens": prompt_tokens, "model": "gpt-4o-mini"},
        output_data={"completion_tokens": completion_tokens}
    )

    final_trace = tracer.end_trace(
        trace_id=trace.trace_id,
        model="gpt-4o-mini",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )

    citations = [
        DocumentCitation(
            document_title="AWS Security Architecture Whitepaper (2024)",
            chunk_id=r.chunk_id,
            tenant_id=payload.tenant_context.tenant_id,
            clearance="INTERNAL",
            page_number=3,
            rerank_score=r.rerank_score,
            original_rank=r.original_rank,
            final_rank=r.new_rank,
            original_length=c.original_tokens,
            compressed_length=c.compressed_tokens,
            snippet=c.compressed_text
        )
        for r, c in compressed_items
    ]

    tokens_saved_by_rerank = (len(initial_candidates) - len(reranked_results)) * 120
    total_saved = tokens_saved_by_rerank + tokens_saved_by_comp

    return {
        "tenant_id": payload.tenant_context.tenant_id,
        "query": payload.query,
        "parsed_self_query": parsed_sq,
        "answer": answer_text,
        "citations": citations,
        "observability": {
            "trace_id": final_trace.trace_id,
            "total_latency_ms": final_trace.total_latency_ms,
            "self_query_parse_latency_ms": round((sq_end - sq_start) * 1000, 2),
            "security_filter_latency_ms": round((sec_end - sec_start) * 1000, 2),
            "retrieval_latency_ms": round((retrieval_end - retrieval_start) * 1000, 2),
            "rerank_latency_ms": round((rerank_end - rerank_start) * 1000, 2),
            "compression_latency_ms": round((comp_end - comp_start) * 1000, 2),
            "generation_latency_ms": round((gen_end - gen_start) * 1000, 2),
            "prompt_tokens": final_trace.prompt_tokens,
            "completion_tokens": final_trace.completion_tokens,
            "total_tokens": final_trace.total_tokens,
            "estimated_cost_usd": final_trace.cost_usd,
            "tokens_saved_by_reranker": tokens_saved_by_rerank,
            "tokens_saved_by_compression": tokens_saved_by_comp,
            "total_tokens_saved": total_saved
        },
        "model": "gpt-4o-mini",
        "retrieval_strategy": "Self-Querying (NL ➔ SQL) ➔ Multi-Tenant RLS ➔ Hybrid (RRF) ➔ Cross-Encoder ➔ Compression"
    }

@app.post("/api/v1/evals/run", response_model=EvalReport)
async def run_automated_evals(samples: Optional[List[EvalSample]] = None):
    test_samples = samples or [
        EvalSample(
            sample_id="eval_sample_01",
            query="What are the IAM MFA requirements in 2024?",
            contexts=["All IAM administrative roles must enforce mandatory MFA on all sensitive API calls."],
            generated_answer="All IAM administrative roles must enforce mandatory MFA on sensitive API calls.",
            ground_truth="Administrative IAM roles require mandatory MFA on sensitive calls."
        )
    ]
    return evaluator.evaluate_dataset(test_samples)

@app.get("/api/v1/observability/trace/{trace_id}", response_model=TraceRecord)
async def get_trace_telemetry(trace_id: str):
    trace = tracer._active_traces.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace ID not found.")
    return trace

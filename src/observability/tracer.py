from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import time
import uuid
import logging
from src.config import settings

logger = logging.getLogger(__name__)

# Real Model Pricing Registry (USD per 1M Tokens)
MODEL_PRICING = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00}
}

class SpanRecord(BaseModel):
    span_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    latency_ms: Optional[float] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TraceRecord(BaseModel):
    trace_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    name: str
    start_time: float
    end_time: Optional[float] = None
    total_latency_ms: Optional[float] = None
    spans: List[SpanRecord] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    status: str = "COMPLETED"
    tags: List[str] = Field(default_factory=list)

class ObservabilityTracer:
    """
    Production-grade distributed tracer compatible with Langfuse & OpenTelemetry.
    Captures end-to-end spans, token usage, latency breakdowns, and financial cost.
    """
    def __init__(self):
        self._active_traces: Dict[str, TraceRecord] = {}

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculates precise USD cost based on model token rates."""
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)

    def start_trace(self, name: str, session_id: Optional[str] = None, user_id: Optional[str] = None, tags: Optional[List[str]] = None) -> TraceRecord:
        """Starts a root trace for an incoming request pipeline."""
        trace_id = str(uuid.uuid4())
        trace = TraceRecord(
            trace_id=trace_id,
            session_id=session_id or f"sess_{int(time.time())}",
            user_id=user_id or "anonymous_user",
            name=name,
            start_time=time.time(),
            tags=tags or ["rag-production", "pgvector", "langfuse"]
        )
        self._active_traces[trace_id] = trace
        return trace

    def add_span(
        self,
        trace_id: str,
        name: str,
        start_time: float,
        end_time: float,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Adds a completed execution span to an active trace."""
        trace = self._active_traces.get(trace_id)
        if not trace:
            return

        latency = round((end_time - start_time) * 1000, 2)
        span = SpanRecord(
            span_id=str(uuid.uuid4()),
            name=name,
            start_time=start_time,
            end_time=end_time,
            latency_ms=latency,
            input_data=input_data or {},
            output_data=output_data or {},
            metadata=metadata or {}
        )
        trace.spans.append(span)

    def end_trace(
        self,
        trace_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> TraceRecord:
        """Finalizes the trace, aggregating tokens, total latency, and dollar cost."""
        trace = self._active_traces.get(trace_id)
        if not trace:
            raise ValueError(f"Trace {trace_id} not found.")

        trace.end_time = time.time()
        trace.total_latency_ms = round((trace.end_time - trace.start_time) * 1000, 2)
        trace.prompt_tokens = prompt_tokens
        trace.completion_tokens = completion_tokens
        trace.total_tokens = prompt_tokens + completion_tokens
        trace.cost_usd = self.calculate_cost(model, prompt_tokens, completion_tokens)
        
        logger.info(f"Trace {trace_id} completed in {trace.total_latency_ms}ms | Cost: ${trace.cost_usd} | Tokens: {trace.total_tokens}")
        return trace

tracer = ObservabilityTracer()

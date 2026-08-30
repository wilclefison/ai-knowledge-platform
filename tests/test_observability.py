import pytest
import time
from src.observability.tracer import ObservabilityTracer

def test_tracer_cost_calculation():
    tracer = ObservabilityTracer()
    
    # 1,000,000 prompt tokens + 1,000,000 completion tokens on gpt-4o-mini
    # Input: $0.15 / 1M, Output: $0.60 / 1M => Total = $0.75
    cost = tracer.calculate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == 0.75

def test_trace_lifecycle_and_spans():
    tracer = ObservabilityTracer()
    
    trace = tracer.start_trace(name="test_pipeline", session_id="s123", user_id="u456")
    assert trace.trace_id in tracer._active_traces
    assert trace.name == "test_pipeline"
    
    # Add Span 1
    t0 = time.time()
    time.sleep(0.01)
    t1 = time.time()
    tracer.add_span(trace.trace_id, name="retrieval_step", start_time=t0, end_time=t1)
    
    # End Trace
    final_trace = tracer.end_trace(trace.trace_id, model="gpt-4o-mini", prompt_tokens=500, completion_tokens=100)
    
    assert len(final_trace.spans) == 1
    assert final_trace.total_tokens == 600
    assert final_trace.cost_usd > 0.0
    assert final_trace.total_latency_ms >= 10.0

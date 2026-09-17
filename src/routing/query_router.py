from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import time
import re
import logging

logger = logging.getLogger(__name__)

class RouteDestination(str, Enum):
    DIRECT_LLM = "DIRECT_LLM"
    VECTOR_RAG = "VECTOR_RAG"
    TEXT_TO_SQL = "TEXT_TO_SQL"
    AGENTIC_MULTI_HOP = "AGENTIC_MULTI_HOP"

class RoutingDecision(BaseModel):
    query: str
    destination: RouteDestination
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    fallback_destination: RouteDestination = RouteDestination.VECTOR_RAG
    latency_ms: float = 0.0

class AgenticQueryRouter:
    """
    Intelligent Autonomous Query Router for Enterprise GenAI.
    Inspects user intent, query complexity, and keywords to dynamically route
    traffic between direct conversational generation, structured SQL, or vector RAG.
    """
    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold

    def classify_route(self, query: str) -> RoutingDecision:
        """
        Analyzes query intent and determines the optimal execution engine.
        """
        start_time = time.time()
        q_lower = query.lower().strip()

        # 1. Check for Conversational / Direct LLM Queries
        greeting_patterns = [r'^(ol[aá]|oi|hello|hi|hey|bom dia|boa tarde|boa noite|como vai|help|quem é você)', r'\b(obrigado|thanks|thank you|valeu)\b']
        if any(re.search(pat, q_lower) for pat in greeting_patterns) and len(q_lower.split()) <= 6:
            latency = round((time.time() - start_time) * 1000, 2)
            return RoutingDecision(
                query=query,
                destination=RouteDestination.DIRECT_LLM,
                confidence_score=0.98,
                reasoning="Query matches conversational greetings or general pleasantries with zero database dependency.",
                fallback_destination=RouteDestination.VECTOR_RAG,
                latency_ms=latency
            )

        # 2. Check for Analytical / Quantitative SQL Queries
        sql_keywords = [
            "quantos", "quantas", "total", "soma", "média", "quantidade", 
            "count", "sum", "average", "qual o valor", "número de", "relatório financeiro"
        ]
        if any(kw in q_lower for kw in sql_keywords) and not ("como fazer" in q_lower or "o que é" in q_lower):
            latency = round((time.time() - start_time) * 1000, 2)
            return RoutingDecision(
                query=query,
                destination=RouteDestination.TEXT_TO_SQL,
                confidence_score=0.92,
                reasoning="Query demands quantitative aggregations, counts, or tabular metrics best served by relational SQL execution.",
                fallback_destination=RouteDestination.VECTOR_RAG,
                latency_ms=latency
            )

        # 3. Check for Complex Multi-Hop Reasoning
        multi_hop_triggers = ["compare", "comparar", "diferença entre", "além disso", "e também", "qual a relação entre"]
        if any(trigger in q_lower for trigger in multi_hop_triggers) and len(q_lower.split()) > 10:
            latency = round((time.time() - start_time) * 1000, 2)
            return RoutingDecision(
                query=query,
                destination=RouteDestination.AGENTIC_MULTI_HOP,
                confidence_score=0.88,
                reasoning="Composite query requires sub-problem decomposition and iterative multi-step context synthesis.",
                fallback_destination=RouteDestination.VECTOR_RAG,
                latency_ms=latency
            )

        # 4. Default: Vector Knowledge Base RAG (Hybrid Search + Cross-Encoder)
        latency = round((time.time() - start_time) * 1000, 2)
        return RoutingDecision(
            query=query,
            destination=RouteDestination.VECTOR_RAG,
            confidence_score=0.95,
            reasoning="Unstructured knowledge inquiry requiring semantic embedding retrieval and context grounding.",
            fallback_destination=RouteDestination.DIRECT_LLM,
            latency_ms=latency
        )

query_router = AgenticQueryRouter()

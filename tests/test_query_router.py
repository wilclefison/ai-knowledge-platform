import pytest
from src.routing.query_router import AgenticQueryRouter, RouteDestination

def test_router_detects_conversational_greetings():
    router = AgenticQueryRouter()
    res = router.classify_route("Bom dia, como você pode me ajudar?")
    assert res.destination == RouteDestination.DIRECT_LLM
    assert res.confidence_score >= 0.90

def test_router_detects_quantitative_sql_queries():
    router = AgenticQueryRouter()
    res = router.classify_route("Qual a quantidade total de documentos cadastrados?")
    assert res.destination == RouteDestination.TEXT_TO_SQL
    assert res.confidence_score >= 0.85

def test_router_detects_vector_rag_queries():
    router = AgenticQueryRouter()
    res = router.classify_route("Quais são os requisitos de MFA para administradores no guia de segurança da AWS?")
    assert res.destination == RouteDestination.VECTOR_RAG
    assert res.confidence_score >= 0.90

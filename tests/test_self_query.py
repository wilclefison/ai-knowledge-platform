import pytest
from src.retrieval.self_query import SelfQueryEngine

def test_self_query_parses_year_and_numeric_ratings():
    engine = SelfQueryEngine()
    query = "Find AWS IAM whitepapers in 2024 with rating > 4.5 for engineering"
    
    parsed = engine.parse_query(query)
    
    # Assert semantic query is cleaned of metadata noise
    assert "in 2024" not in parsed.semantic_query
    assert "rating > 4.5" not in parsed.semantic_query
    assert "for engineering" not in parsed.semantic_query
    assert "AWS IAM whitepapers" in parsed.semantic_query
    
    # Assert filters extracted
    filters_dict = {f.field: f for f in parsed.filters}
    assert "year" in filters_dict
    assert filters_dict["year"].value == 2024
    
    assert "rating" in filters_dict
    assert filters_dict["rating"].value == 4.5
    assert filters_dict["rating"].operator == "gt"
    
    assert "department" in filters_dict
    assert filters_dict["department"].value == "engineering"
    
    # Assert SQL WHERE generation
    assert "(metadata->>'year')::numeric = :sq_param_0" in parsed.sql_where_clause
    assert "(metadata->>'rating')::numeric > :sq_param_1" in parsed.sql_where_clause

def test_self_query_handles_pure_semantic_query():
    engine = SelfQueryEngine()
    query = "How do I configure VPC security groups?"
    parsed = engine.parse_query(query)
    
    assert parsed.semantic_query == query
    assert len(parsed.filters) == 0
    assert parsed.sql_where_clause == "1=1"

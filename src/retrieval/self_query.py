from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import re
import logging

logger = logging.getLogger(__name__)

class ComparisonFilter(BaseModel):
    field: str
    operator: str = Field(..., example="eq", description="eq, neq, gt, gte, lt, lte, in, contains")
    value: Union[str, int, float, List[str]]

class ParsedSelfQuery(BaseModel):
    raw_query: str
    semantic_query: str = Field(..., description="Clean text query for dense vector embedding generation")
    filters: List[ComparisonFilter] = Field(default_factory=list)
    sql_where_clause: str = ""
    sql_parameters: Dict[str, Any] = Field(default_factory=dict)

class SelfQueryEngine:
    """
    Production-grade Self-Querying & Dynamic Metadata SQL Generation Engine.
    Decomposes natural language queries into pure semantic vector queries + structured SQL metadata filters.
    """
    SQL_OPERATOR_MAP = {
        "eq": "=",
        "neq": "!=",
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
        "contains": "ILIKE"
    }

    def parse_query(self, query: str) -> ParsedSelfQuery:
        """
        Decomposes natural language query into clean semantic search string and structured filters.
        """
        filters: List[ComparisonFilter] = []
        cleaned_query = query

        # 1. Detect Year Filters (e.g., 'in 2024', 'from 2023')
        year_match = re.search(r'\b(in|from|year)\s+(20\d{2})\b', cleaned_query, re.IGNORECASE)
        if year_match:
            year_val = int(year_match.group(2))
            filters.append(ComparisonFilter(field="year", operator="eq", value=year_val))
            cleaned_query = re.sub(r'\b(in|from|year)\s+20\d{2}\b', '', cleaned_query, flags=re.IGNORECASE)

        # 2. Detect Rating / Score Filters (e.g., 'rating > 4.5', 'score greater than 9')
        score_gt = re.search(r'\b(rating|score|evaluation)\s*(>|greater than|above)\s*([0-9.]+)', cleaned_query, re.IGNORECASE)
        if score_gt:
            val = float(score_gt.group(3))
            filters.append(ComparisonFilter(field="rating", operator="gt", value=val))
            cleaned_query = re.sub(r'\b(rating|score|evaluation)\s*(>|greater than|above)\s*[0-9.]+', '', cleaned_query, flags=re.IGNORECASE)

        # 3. Detect Department / Category Filters (e.g., 'in engineering', 'for finance')
        dept_match = re.search(r'\b(in|for|department)\s+(engineering|finance|security|legal|hr)\b', cleaned_query, re.IGNORECASE)
        if dept_match:
            dept_val = dept_match.group(2).lower()
            filters.append(ComparisonFilter(field="department", operator="eq", value=dept_val))
            cleaned_query = re.sub(r'\b(in|for|department)\s+(engineering|finance|security|legal|hr)\b', '', cleaned_query, flags=re.IGNORECASE)

        # Clean excess whitespaces from semantic query
        semantic_query = " ".join(cleaned_query.split())
        if not semantic_query:
            semantic_query = query

        # Build parameterized SQL where clause
        sql_clauses = []
        sql_params = {}

        for idx, f in enumerate(filters):
            param_name = f"sq_param_{idx}"
            sql_op = self.SQL_OPERATOR_MAP.get(f.operator, "=")
            
            if f.field in ["year", "rating"]:
                sql_clauses.append(f"(metadata->>'{f.field}')::numeric {sql_op} :{param_name}")
                sql_params[param_name] = f.value
            elif f.operator == "contains":
                sql_clauses.append(f"metadata->>'{f.field}' ILIKE :{param_name}")
                sql_params[param_name] = f"%{f.value}%"
            else:
                sql_clauses.append(f"metadata->>'{f.field}' {sql_op} :{param_name}")
                sql_params[param_name] = str(f.value)

        sql_where = " AND ".join(sql_clauses) if sql_clauses else "1=1"

        return ParsedSelfQuery(
            raw_query=query,
            semantic_query=semantic_query,
            filters=filters,
            sql_where_clause=sql_where,
            sql_parameters=sql_params
        )

self_query_engine = SelfQueryEngine()

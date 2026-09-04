import pytest
from src.db.security import SecurityFilterEngine, TenantContext, ClearanceLevel

def test_security_engine_blocks_cross_tenant_chunks():
    engine = SecurityFilterEngine()
    context = TenantContext(
        tenant_id="tenant_acme",
        user_id="alice",
        roles=["engineering"],
        clearance=ClearanceLevel.INTERNAL
    )
    
    candidates = [
        {"id": "c1", "tenant_id": "tenant_acme", "clearance": "INTERNAL", "allowed_roles": ["engineering"], "content": "ACME API Spec."},
        {"id": "c2", "tenant_id": "tenant_rival_corp", "clearance": "INTERNAL", "allowed_roles": ["engineering"], "content": "Rival proprietary source code."}
    ]
    
    filtered = engine.filter_candidates(context, candidates)
    
    assert len(filtered) == 1
    assert filtered[0]["id"] == "c1"
    assert filtered[0]["tenant_id"] == "tenant_acme"

def test_security_engine_enforces_clearance_levels():
    engine = SecurityFilterEngine()
    context = TenantContext(
        tenant_id="tenant_acme",
        user_id="bob",
        roles=["viewer"],
        clearance=ClearanceLevel.INTERNAL  # Level 1
    )
    
    candidates = [
        {"id": "c1", "tenant_id": "tenant_acme", "clearance": "INTERNAL", "allowed_roles": ["public"], "content": "Public handbook."},
        {"id": "c2", "tenant_id": "tenant_acme", "clearance": "RESTRICTED", "allowed_roles": ["public"], "content": "Restricted salary data."}  # Level 3
    ]
    
    filtered = engine.filter_candidates(context, candidates)
    
    # c2 must be blocked because user clearance is only INTERNAL (Level 1)
    assert len(filtered) == 1
    assert filtered[0]["id"] == "c1"

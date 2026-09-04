from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ClearanceLevel(str, Enum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"

class TenantContext(BaseModel):
    tenant_id: str = Field(..., example="tenant_acme_corp")
    user_id: str = Field(..., example="usr_alice_101")
    roles: List[str] = Field(default_factory=lambda: ["viewer"], example=["finance_analyst", "engineering"])
    clearance: ClearanceLevel = Field(default=ClearanceLevel.INTERNAL)

class SecurityFilterEngine:
    """
    Multi-Tenant Access Control and Row-Level Security (RLS) Engine for Vector Retrieval.
    Enforces strict metadata isolation in SQL queries to prevent cross-tenant vector leakage.
    """
    CLEARANCE_HIERARCHY = {
        ClearanceLevel.PUBLIC: 0,
        ClearanceLevel.INTERNAL: 1,
        ClearanceLevel.CONFIDENTIAL: 2,
        ClearanceLevel.RESTRICTED: 3
    }

    def build_sql_filter(self, context: TenantContext) -> Dict[str, Any]:
        """
        Builds parameterized SQL WHERE clause enforcing tenant isolation and role matching.
        """
        user_clearance_num = self.CLEARANCE_HIERARCHY.get(context.clearance, 1)
        allowed_clearances = [
            lvl.value for lvl, num in self.CLEARANCE_HIERARCHY.items()
            if num <= user_clearance_num
        ]

        return {
            "sql_where": """
                tenant_id = :tenant_id
                AND clearance = ANY(:allowed_clearances)
                AND (allowed_roles && :user_roles OR 'public' = ANY(allowed_roles))
            """,
            "params": {
                "tenant_id": context.tenant_id,
                "allowed_clearances": allowed_clearances,
                "user_roles": context.roles
            }
        }

    def filter_candidates(
        self,
        context: TenantContext,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validates in-memory candidate chunks against tenant boundaries and security clearances.
        """
        user_clearance_num = self.CLEARANCE_HIERARCHY.get(context.clearance, 1)
        user_roles_set = set(r.lower() for r in context.roles)
        
        authorized_candidates = []
        
        for item in candidates:
            # 1. Tenant boundary validation (Hard check)
            item_tenant = item.get("tenant_id", "default")
            if item_tenant != context.tenant_id:
                logger.warning(f"Security Alert: Blocked cross-tenant vector leak attempt! Target Tenant: {context.tenant_id}, Chunk Tenant: {item_tenant}")
                continue

            # 2. Clearance level check
            item_clearance_str = item.get("clearance", ClearanceLevel.INTERNAL.value)
            try:
                item_clearance = ClearanceLevel(item_clearance_str)
                item_clearance_num = self.CLEARANCE_HIERARCHY.get(item_clearance, 1)
            except ValueError:
                item_clearance_num = 1

            if item_clearance_num > user_clearance_num:
                logger.info(f"Access Denied: Chunk {item.get('id')} requires {item_clearance_str}, user clearance is {context.clearance.value}")
                continue

            # 3. Role-based access control (RBAC)
            allowed_roles = set(r.lower() for r in item.get("allowed_roles", ["public"]))
            if "public" in allowed_roles or bool(user_roles_set.intersection(allowed_roles)):
                authorized_candidates.append(item)
            else:
                logger.info(f"Access Denied: User roles {context.roles} not in chunk roles {allowed_roles}")

        return authorized_candidates

security_engine = SecurityFilterEngine()

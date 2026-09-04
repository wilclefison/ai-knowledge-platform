-- Schema definition for Enterprise AI Knowledge Platform
-- PostgreSQL 16 + pgvector with Multi-Tenant Row-Level Security (RLS)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Document Registry
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(64) NOT NULL,
    title VARCHAR(255) NOT NULL,
    clearance VARCHAR(32) DEFAULT 'INTERNAL',
    allowed_roles TEXT[] DEFAULT ARRAY['public'],
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents(tenant_id);

-- Document Chunks Table
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id VARCHAR(64) NOT NULL,
    content TEXT NOT NULL,
    page_number INT DEFAULT 1,
    chunk_index INT NOT NULL,
    clearance VARCHAR(32) DEFAULT 'INTERNAL',
    allowed_roles TEXT[] DEFAULT ARRAY['public'],
    
    -- Dense Embedding (1536 dims for text-embedding-3-small or OpenAI ada-002)
    embedding vector(1536),
    
    -- Sparse Lexical TSVector for BM25 search
    tsv_content tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for Ultra-low Latency Hybrid Search
-- 1. HNSW Index for Cosine Vector Similarity
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 2. GIN Index for Millisecond Full-Text Search (Sparse BM25)
CREATE INDEX IF NOT EXISTS idx_chunks_tsv 
ON document_chunks 
USING GIN (tsv_content);

-- 3. Composite Tenant & Metadata Index
CREATE INDEX IF NOT EXISTS idx_chunks_tenant_sec 
ON document_chunks(tenant_id, clearance);

-- PostgreSQL Row-Level Security (RLS) Policies
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;

-- Tenant Isolation RLS Policy
DROP POLICY IF EXISTS tenant_isolation_policy ON document_chunks;
CREATE POLICY tenant_isolation_policy ON document_chunks
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

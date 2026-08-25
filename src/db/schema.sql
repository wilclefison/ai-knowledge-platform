-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents Master Table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    source_uri VARCHAR(512),
    file_type VARCHAR(50) DEFAULT 'markdown',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Document Chunks for Hybrid Retrieval
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    tsv_content tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    page_number INT DEFAULT 1,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 1. HNSW Index for Fast Cosine Similarity Vector Search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 2. GIN Index for Fast BM25/Full-Text Sparse Search
CREATE INDEX IF NOT EXISTS idx_chunks_tsv 
ON document_chunks 
USING gin (tsv_content);

-- 3. Document ID Index for Foreign Key Filtering
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id 
ON document_chunks (document_id);

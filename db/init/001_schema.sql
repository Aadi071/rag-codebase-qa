-- Enable pgvector. The application creates the `chunks` table itself (via
-- store.db.init_schema) so the vector() dimension always matches the chosen
-- embedding backend (768 for local bge-base, 1536 for OpenAI).
CREATE EXTENSION IF NOT EXISTS vector;

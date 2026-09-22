import os

DB_DSN = os.getenv("DB_DSN", "postgresql://nba:nba@localhost:5432/nba")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:2b")

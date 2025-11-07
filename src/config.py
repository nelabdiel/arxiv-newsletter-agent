# src/config.py

"""Config & telemetry hardening."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Disable common telemetry by default (must be set before imports elsewhere)
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGGRAPH_CLI_NO_ANALYTICS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

load_dotenv()  # allow .env overrides

@dataclass
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma3")
    query: str = os.getenv("QUERY", "all:quantum")
    since_hours: int = int(os.getenv("SINCE_HOURS", "24"))
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    cluster_threshold: float = float(os.getenv("CLUSTER_THRESHOLD", "0.35"))
    max_papers_per_cluster: int = int(os.getenv("MAX_PAPERS_PER_CLUSTER", "6"))
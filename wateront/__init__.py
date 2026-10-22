"""WaterOnt: layered knowledge graph + GraphRAG for explainable asset management."""
from .config import Config
from .kg import KnowledgeGraph
from .pipeline import build_knowledge_graph

__all__ = ["Config", "KnowledgeGraph", "build_knowledge_graph"]
__version__ = "0.1.0"

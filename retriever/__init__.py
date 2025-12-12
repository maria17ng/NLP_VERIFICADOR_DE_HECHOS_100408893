"""
Pipeline Avanzado de Recuperación para RAG.

Implementa estrategias avanzadas de recuperación:
- Filtrado por metadatos enriquecidos
- Búsqueda híbrida (semántica + keywords)
- Reranking con cross-encoder
- Diversificación de fuentes
- Scoring de relevancia con thresholds

Autor: Proyecto Final NLP - UC3M
Fecha: Diciembre 2025
"""

from retriever.advanced_retriever import AdvancedRetriever
from retriever.config import RetrievalConfig
from retriever.diversity_selector import DiversitySelector
from retriever.hybrid_search import HybridSearcher
from retriever.metadata_filter import MetadataFilter

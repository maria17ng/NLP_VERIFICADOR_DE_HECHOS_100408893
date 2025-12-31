from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    """Configuración para el pipeline de recuperación."""

    # Búsqueda vectorial inicial
    k_initial: int = 50

    # Filtrado de metadatos
    use_metadata_filter: bool = True
    metadata_boost: float = 0.2  # Boost para docs con metadatos coincidentes

    # Búsqueda híbrida
    use_hybrid_search: bool = True
    keyword_weight: float = 0.3  # Peso de keyword matching (0-1)
    team_preference_boost: float = 0.2 # Boost para priorizar documentos del mismo equipo

    # Reranking
    use_reranker: bool = True
    rerank_top_k: int = 20  # Docs a reranquear

    # Diversidad
    use_diversity: bool = True
    max_chunks_per_source: int = 3  # Máx chunks del mismo documento
    diversity_penalty: float = 0.1  # Penalización por repetición

    # Thresholds
    min_relevance_score: float = 0.3  # Score mínimo para incluir doc
    min_rerank_score: float = -5.0  # Score mínimo de reranker

    # Resultado final
    final_top_k: int = 5

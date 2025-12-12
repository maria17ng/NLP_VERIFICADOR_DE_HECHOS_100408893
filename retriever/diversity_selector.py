from typing import List, Dict, Tuple
from collections import defaultdict

from langchain_core.documents import Document
from retriever.config import RetrievalConfig
from utils.utils import setup_logger


class DiversitySelector:
    """
    Selecciona documentos diversos para evitar redundancia.

    Penaliza chunks que provienen del mismo documento fuente
    o tienen contenido muy similar.
    """

    def __init__(self, config: RetrievalConfig):
        """
        Inicializa el selector de diversidad.

        Args:
            config: Configuración del pipeline
        """
        self.config = config
        self.logger = setup_logger('DiversitySelector', level='DEBUG')

    def diversify(self, scored_documents: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        Aplica diversificación a los documentos puntuados.

        Args:
            scored_documents: Lista de (documento, score)

        Returns:
            Lista diversificada de (documento, adjusted_score)
        """
        if not self.config.use_diversity:
            return scored_documents

        # Contar chunks por fuente
        source_counts: Dict[str, int] = defaultdict(int)
        diversified = []

        for doc, score in scored_documents:
            source = doc.metadata.get('parent_source', doc.metadata.get('source', 'unknown'))

            # Aplicar penalización si ya tenemos muchos chunks de esta fuente
            count = source_counts[source]
            if count >= self.config.max_chunks_per_source:
                penalty = self.config.diversity_penalty * count
                adjusted_score = max(score - penalty, 0.0)
            else:
                adjusted_score = score

            diversified.append((doc, adjusted_score))
            source_counts[source] += 1

        # Reordenar por adjusted score
        diversified.sort(key=lambda x: x[1], reverse=True)

        # Log de diversidad
        unique_sources = len(source_counts)
        self.logger.debug(
            f"Diversidad: {unique_sources} fuentes únicas, "
            f"máx {max(source_counts.values())} chunks por fuente"
        )

        return diversified

from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np

from langchain_core.documents import Document
from retriever.config import RetrievalConfig
from utils.utils import setup_logger


class DiversitySelector:
    """
    Selecciona documentos diversos para evitar redundancia.

    Implementa MMR (Maximal Marginal Relevance) para balancear
    relevancia y diversidad de forma genérica.
    """

    def __init__(self, config: RetrievalConfig):
        """
        Inicializa el selector de diversidad.

        Args:
            config: Configuración del pipeline
        """
        self.config = config
        self.logger = setup_logger('DiversitySelector', level='DEBUG')
        self.embeddings_cache = {}  # Cache para embeddings de documentos

    def diversify(self, scored_documents: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        Aplica diversificación MMR (Maximal Marginal Relevance) a los documentos.

        MMR balancea relevancia (score) y diversidad (similitud entre docs),
        seleccionando docs que sean relevantes pero diversos entre sí.

        Args:
            scored_documents: Lista de (documento, score)

        Returns:
            Lista diversificada de (documento, score)
        """
        if not self.config.use_diversity or len(scored_documents) <= 1:
            return scored_documents

        # Usar MMR con lambda=0.7 (70% relevancia, 30% diversidad)
        mmr_lambda = 0.7

        # Ordenar por score inicial
        sorted_docs = sorted(scored_documents, key=lambda x: x[1], reverse=True)

        # Selección MMR
        selected = []
        remaining = sorted_docs.copy()

        # Seleccionar el primero (más relevante)
        if remaining:
            selected.append(remaining.pop(0))

        # Seleccionar el resto maximizando MMR
        while remaining and len(selected) < len(scored_documents):
            mmr_scores = []
            for doc, rel_score in remaining:
                # Diversidad: mínima similitud con los ya seleccionados
                max_sim = max(
                    self._similarity(doc, sel_doc)
                    for sel_doc, _ in selected
                )
                # MMR score
                mmr_score = mmr_lambda * rel_score - (1 - mmr_lambda) * max_sim
                mmr_scores.append((doc, rel_score, mmr_score))

            # Seleccionar el de mayor MMR
            mmr_scores.sort(key=lambda x: x[2], reverse=True)
            best_doc, best_rel_score, _ = mmr_scores[0]
            selected.append((best_doc, best_rel_score))
            remaining = [(d, s) for d, s in remaining if d != best_doc]

        # Aplicar caps por fuente como filtro final
        source_counts: Dict[str, int] = defaultdict(int)
        diversified = []
        for doc, score in selected:
            source = doc.metadata.get('parent_source', doc.metadata.get('source', 'unknown'))
            if source_counts[source] < self.config.max_chunks_per_source:
                diversified.append((doc, score))
                source_counts[source] += 1

        unique_sources = len(source_counts)
        self.logger.debug(
            f"Diversidad MMR: {unique_sources} fuentes únicas, "
            f"máx {max(source_counts.values()) if source_counts else 0} chunks por fuente"
        )

        return diversified

    def _similarity(self, doc1: Document, doc2: Document) -> float:
        """
        Calcula similitud textual simple entre dos documentos.

        Usa Jaccard sobre tokens como aproximación rápida y genérica.
        No requiere embeddings externos.

        Args:
            doc1: Primer documento
            doc2: Segundo documento

        Returns:
            Similitud entre 0.0 y 1.0
        """
        import re

        def tokenize(text: str) -> set:
            return set(re.findall(r'\b\w{3,}\b', text.lower()))

        tokens1 = tokenize(doc1.page_content)
        tokens2 = tokenize(doc2.page_content)

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union) if union else 0.0

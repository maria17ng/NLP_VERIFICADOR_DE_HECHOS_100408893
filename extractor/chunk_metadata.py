from typing import Dict, Any
from langchain_core.documents import Document

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from extractor import MetadataExtractor


class ChunkMetadataEnricher:
    """
    Enriquece metadatos de chunks individuales.

    Añade información específica del chunk para mejorar
    la recuperación y el reranking.
    """

    def __init__(self):
        """Inicializa el enriquecedor de metadatos."""
        self.extractor = MetadataExtractor()

    def enrich_chunk(self, chunk: Document, parent_metadata: Dict[str, Any]) -> Document:
        """
        Enriquece metadatos de un chunk.

        Args:
            chunk: Chunk a enriquecer
            parent_metadata: Metadatos del documento padre

        Returns:
            Chunk con metadatos enriquecidos
        """
        # Heredar metadatos del padre
        chunk.metadata.update({
            'parent_' + k: v
            for k, v in parent_metadata.items()
            if k in ['title', 'source', 'content_type']
        })

        # Extraer metadatos específicos del chunk
        chunk = self.extractor.extract_metadata(chunk)

        # Calcular relevancia potencial
        chunk.metadata['relevance_score'] = self._calculate_relevance_score(chunk)

        return chunk

    @staticmethod
    def _calculate_relevance_score(chunk: Document) -> float:
        """
        Calcula un score de relevancia potencial del chunk.

        Basado en densidad de información y presencia de entidades.

        Args:
            chunk: Chunk a evaluar

        Returns:
            Score de relevancia (0-1)
        """
        score = 0.0
        metadata = chunk.metadata

        # Densidad de información
        score += metadata.get('info_density', 0.0) * 0.4

        # Presencia de fechas
        if metadata.get('dates'):
            score += 0.2

        # Presencia de entidades
        has_entities = any([
            metadata.get('persons'),
            metadata.get('organizations'),
            metadata.get('locations')
        ])
        if has_entities:
            score += 0.2

        # Longitud razonable (ni muy corto ni muy largo)
        text_length = len(chunk.page_content)
        if 200 <= text_length <= 1500:
            score += 0.2

        return min(score, 1.0)

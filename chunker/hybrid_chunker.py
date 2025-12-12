from typing import List
from langchain_core.documents import Document

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  spaCy no disponible. Usando fallback a chunking simple.")

from chunker.semantic_chunker import SemanticChunker


class HybridChunker:
    """
    Chunker híbrido que crea chunks de múltiples tamaños.

    Crea chunks pequeños (granulares) y grandes (contextuales)
    para optimizar la recuperación en diferentes escenarios.
    """

    def __init__(self, small_chunk_size: int = 512, large_chunk_size: int = 1500, chunk_overlap: int = 100):
        """
        Inicializa el chunker híbrido.

        Args:
            small_chunk_size: Tamaño de chunks pequeños
            large_chunk_size: Tamaño de chunks grandes
            chunk_overlap: Solapamiento entre chunks
        """
        self.small_chunker = SemanticChunker(
            chunk_size=small_chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=50,
            max_chunk_size=small_chunk_size * 2
        )

        self.large_chunker = SemanticChunker(
            chunk_size=large_chunk_size,
            chunk_overlap=chunk_overlap * 2,
            min_chunk_size=200,
            max_chunk_size=large_chunk_size * 2
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Divide documentos en chunks de múltiples tamaños.

        Args:
            documents: Documentos a dividir

        Returns:
            Lista de chunks (pequeños y grandes)
        """
        all_chunks = []

        # Crear chunks pequeños
        small_chunks = self.small_chunker.split_documents(documents)
        for chunk in small_chunks:
            chunk.metadata['chunk_size_type'] = 'small'
            all_chunks.append(chunk)

        # Crear chunks grandes
        large_chunks = self.large_chunker.split_documents(documents)
        for chunk in large_chunks:
            chunk.metadata['chunk_size_type'] = 'large'
            all_chunks.append(chunk)

        return all_chunks

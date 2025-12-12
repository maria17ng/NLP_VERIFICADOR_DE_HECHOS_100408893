import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  spaCy no disponible. Usando fallback a chunking simple.")


class SemanticChunker:
    """
    Chunker semántico que respeta límites naturales del texto.

    Attributes:
        chunk_size: Tamaño objetivo de chunk en caracteres
        chunk_overlap: Solapamiento entre chunks
        respect_sentences: Si respetar límites de oraciones
        min_chunk_size: Tamaño mínimo de chunk
        max_chunk_size: Tamaño máximo de chunk
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, respect_sentences: bool = True,
                 min_chunk_size: int = 100, max_chunk_size: int = 2000):
        """
        Inicializa el chunker semántico.

        Args:
            chunk_size: Tamaño objetivo de chunk
            chunk_overlap: Solapamiento entre chunks
            respect_sentences: Si respetar límites de oraciones
            min_chunk_size: Tamaño mínimo
            max_chunk_size: Tamaño máximo
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_sentences = respect_sentences
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # Cargar modelo de spaCy si está disponible
        self.nlp = None
        if SPACY_AVAILABLE and respect_sentences:
            try:
                self.nlp = spacy.load("es_core_news_sm", disable=["ner", "lemmatizer"])
            except OSError:
                print("⚠️  Modelo spaCy 'es_core_news_sm' no encontrado.")
                print("   Instalar con: python -m spacy download es_core_news_sm")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Divide documentos en chunks semánticos.

        Args:
            documents: Lista de documentos a dividir

        Returns:
            Lista de chunks
        """
        all_chunks = []

        for doc in documents:
            chunks = self.split_text(doc.page_content, doc.metadata)
            all_chunks.extend(chunks)

        return all_chunks

    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Divide texto en chunks semánticos.

        Args:
            text: Texto a dividir
            metadata: Metadatos del documento original

        Returns:
            Lista de chunks como documentos
        """
        if not text:
            return []

        if metadata is None:
            metadata = {}

        # Estrategia según disponibilidad de spaCy
        if self.nlp and self.respect_sentences:
            chunks = self._split_by_sentences(text)
        else:
            chunks = self._split_by_paragraphs(text)

        # Crear documentos de chunks
        chunk_docs = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'chunk_method': 'semantic'
            })

            chunk_docs.append(Document(
                page_content=chunk_text,
                metadata=chunk_metadata
            ))

        return chunk_docs

    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Divide texto respetando límites de oraciones.

        Args:
            text: Texto a dividir

        Returns:
            Lista de chunks
        """
        # Procesar texto con spaCy
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        return self._group_sentences(sentences)

    def _group_sentences(self, sentences: List[str]) -> List[str]:
        """
        Agrupa oraciones en chunks respetando tamaño objetivo.

        Args:
            sentences: Lista de oraciones

        Returns:
            Lista de chunks
        """
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # Si la oración sola excede el máximo, dividirla
            if sentence_size > self.max_chunk_size:
                # Guardar chunk actual si existe
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Dividir oración larga
                sub_chunks = self._split_long_sentence(sentence)
                chunks.extend(sub_chunks)
                continue

            # Si añadir esta oración excede el tamaño objetivo
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Guardar chunk actual
                chunks.append(' '.join(current_chunk))

                # Calcular overlapping
                overlap_sentences = self._calculate_overlap(current_chunk)
                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in current_chunk)

            # Añadir oración al chunk actual
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 por espacio

        # Añadir último chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)

        return chunks

    def _calculate_overlap(self, sentences: List[str]) -> List[str]:
        """
        Calcula qué oraciones incluir en el overlap.

        Args:
            sentences: Oraciones del chunk anterior

        Returns:
            Oraciones para el overlap
        """
        overlap_sentences = []
        overlap_size = 0

        # Tomar oraciones desde el final hasta alcanzar el overlap
        for sentence in reversed(sentences):
            sentence_size = len(sentence)
            if overlap_size + sentence_size <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_size += sentence_size + 1
            else:
                break

        return overlap_sentences

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """
        Divide una oración muy larga por comas o puntos y comas.

        Args:
            sentence: Oración a dividir

        Returns:
            Lista de fragmentos
        """
        # Intentar dividir por comas o puntos y comas
        fragments = re.split(r'[,;]\s+', sentence)

        chunks = []
        current_chunk = []
        current_size = 0

        for fragment in fragments:
            fragment_size = len(fragment)

            if current_size + fragment_size > self.chunk_size and current_chunk:
                chunks.append(', '.join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(fragment)
            current_size += fragment_size + 2  # +2 por ", "

        if current_chunk:
            chunks.append(', '.join(current_chunk))

        return chunks

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """
        Divide texto por párrafos cuando spaCy no está disponible.

        Args:
            text: Texto a dividir

        Returns:
            Lista de chunks
        """
        # Dividir por párrafos (doble salto de línea)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = []
        current_size = 0

        for paragraph in paragraphs:
            para_size = len(paragraph)

            # Si el párrafo solo excede el máximo, dividirlo
            if para_size > self.max_chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Usar fallback a RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                sub_chunks = splitter.split_text(paragraph)
                chunks.extend(sub_chunks)
                continue

            # Si añadir este párrafo excede el tamaño
            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))

                # Overlap: incluir último párrafo si cabe
                if para_size <= self.chunk_overlap:
                    current_chunk = [paragraph]
                    current_size = para_size
                else:
                    current_chunk = []
                    current_size = 0

            if not current_chunk or current_size == 0:
                current_chunk.append(paragraph)
                current_size = para_size
            else:
                current_chunk.append(paragraph)
                current_size += para_size + 2  # +2 por \n\n

        # Último chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)

        return chunks

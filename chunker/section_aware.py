import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  spaCy no disponible. Usando fallback a chunking simple.")

from chunker import SemanticChunker


class SectionAwareChunker(SemanticChunker):
    """
    Chunker que respeta límites de secciones del documento.

    No parte chunks a mitad de una sección identificada.
    """

    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Divide texto respetando secciones.

        Args:
            text: Texto a dividir
            metadata: Metadatos

        Returns:
            Lista de chunks
        """
        if metadata is None:
            metadata = {}

        # Detectar secciones
        sections = self._detect_sections(text)

        # Si no hay secciones, usar método estándar
        if len(sections) <= 1:
            return super().split_text(text, metadata)

        # Dividir cada sección independientemente
        all_chunks = []

        for section_title, section_content in sections:
            section_metadata = metadata.copy()
            section_metadata['section_title'] = section_title

            # Dividir sección
            section_chunks = super().split_text(section_content, section_metadata)
            all_chunks.extend(section_chunks)

        return all_chunks

    @staticmethod
    def _detect_sections(text: str) -> List[Tuple[str, str]]:
        """
        Detecta secciones en el texto.

        Args:
            text: Texto a analizar

        Returns:
            Lista de tuplas (título_sección, contenido)
        """
        sections = []
        current_section = ('Inicio', '')

        lines = text.split('\n')
        section_pattern = re.compile(r'^(#{1,3}\s+|\d+\.\s+)(.+)$')

        for line in lines:
            match = section_pattern.match(line.strip())

            if match:
                # Guardar sección anterior
                if current_section[1].strip():
                    sections.append(current_section)

                # Nueva sección
                title = match.group(2).strip()
                current_section = (title, '')
            else:
                # Añadir a contenido actual
                current_section = (current_section[0], current_section[1] + line + '\n')

        # Añadir última sección
        if current_section[1].strip():
            sections.append(current_section)

        return sections if len(sections) > 1 else [('Documento completo', text)]

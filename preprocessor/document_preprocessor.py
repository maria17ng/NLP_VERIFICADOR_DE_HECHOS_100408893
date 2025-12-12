import re
import unicodedata
from typing import List, Dict, Any
from langchain_core.documents import Document


class DocumentPreprocessor:
    """
    Preprocesador de documentos para limpieza y normalización.

    Attributes:
        remove_urls: Si eliminar URLs del texto
        remove_emails: Si eliminar emails del texto
        normalize_whitespace: Si normalizar espacios en blanco
        fix_encoding: Si corregir problemas de encoding
        min_paragraph_length: Longitud mínima de párrafo en caracteres
    """

    def __init__(self, remove_urls: bool = True, remove_emails: bool = True, normalize_whitespace: bool = True,
                 fix_encoding: bool = True, min_paragraph_length: int = 50):
        """
        Inicializa el preprocesador.

        Args:
            remove_urls: Si eliminar URLs
            remove_emails: Si eliminar emails
            normalize_whitespace: Si normalizar espacios
            fix_encoding: Si corregir encoding
            min_paragraph_length: Longitud mínima de párrafo
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.fix_encoding = fix_encoding
        self.min_paragraph_length = min_paragraph_length

    def preprocess_document(self, document: Document) -> Document:
        """
        Preprocesa un documento completo.

        Args:
            document: Documento a preprocesar

        Returns:
            Documento preprocesado
        """
        # Limpiar texto
        cleaned_text = self.clean_text(document.page_content)

        # Normalizar
        normalized_text = self.normalize_text(cleaned_text)

        # Detectar estructura
        structure = self.detect_structure(normalized_text)

        # Actualizar metadata con información de estructura
        document.metadata.update({
            'has_title': structure['has_title'],
            'num_sections': structure['num_sections'],
            'num_paragraphs': structure['num_paragraphs']
        })

        # Actualizar contenido
        document.page_content = normalized_text

        return document

    def clean_text(self, text: str) -> str:
        """
        Limpia el texto eliminando elementos no deseados.

        Args:
            text: Texto a limpiar

        Returns:
            Texto limpio
        """
        if not text:
            return ""

        # Corregir encoding si es necesario
        if self.fix_encoding:
            text = self._fix_encoding_issues(text)

        # Eliminar URLs
        if self.remove_urls:
            text = self._remove_urls(text)

        # Eliminar emails
        if self.remove_emails:
            text = self._remove_emails(text)

        # Eliminar caracteres de control
        text = self._remove_control_characters(text)

        # Normalizar espacios
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        return text

    def normalize_text(self, text: str) -> str:
        """
        Normaliza el texto aplicando transformaciones estándar.

        Args:
            text: Texto a normalizar

        Returns:
            Texto normalizado
        """
        # Normalización unicode (NFD -> NFC)
        text = unicodedata.normalize('NFC', text)

        # Corregir problemas comunes de puntuación
        text = self._fix_punctuation(text)

        # Eliminar líneas muy cortas (probablemente headers/footers)
        text = self._remove_short_lines(text)

        return text

    def detect_structure(self, text: str) -> Dict[str, Any]:
        """
        Detecta la estructura del documento.

        Args:
            text: Texto del documento

        Returns:
            Diccionario con información de estructura
        """
        structure = {
            'has_title': False,
            'num_sections': 0,
            'num_paragraphs': 0,
            'sections': []
        }

        lines = text.split('\n')

        # Detectar título (primera línea no vacía si es corta y sin punto final)
        for line in lines:
            if line.strip():
                if len(line.strip()) < 100 and not line.strip().endswith('.'):
                    structure['has_title'] = True
                break

        # Detectar secciones (líneas cortas en mayúsculas o que empiezan con números)
        section_pattern = re.compile(r'^(\d+\.|\d+\)|\#|\*{1,3})\s*(.+)$')
        for line in lines:
            stripped = line.strip()
            if section_pattern.match(stripped) or (
                    len(stripped) < 80 and
                    stripped.isupper() and
                    len(stripped.split()) > 1
            ):
                structure['num_sections'] += 1
                structure['sections'].append(stripped)

        # Detectar párrafos (bloques de texto separados por líneas vacías)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        structure['num_paragraphs'] = len([
            p for p in paragraphs
            if len(p) >= self.min_paragraph_length
        ])

        return structure

    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Extrae secciones del texto con sus títulos.

        Args:
            text: Texto del documento

        Returns:
            Lista de secciones con título y contenido
        """
        sections = []
        current_section = {'title': 'Introducción', 'content': ''}

        lines = text.split('\n')
        section_pattern = re.compile(r'^(\d+\.|\d+\)|\#\#?|\*{1,3})\s*(.+)$')

        for line in lines:
            stripped = line.strip()
            match = section_pattern.match(stripped)

            # Si es un título de sección
            if match or (
                    len(stripped) < 80 and
                    stripped.isupper() and
                    len(stripped.split()) > 2
            ):
                # Guardar sección anterior si tiene contenido
                if current_section['content'].strip():
                    sections.append(current_section.copy())

                # Iniciar nueva sección
                title = match.group(2) if match else stripped
                current_section = {'title': title, 'content': ''}
            else:
                # Añadir al contenido de la sección actual
                current_section['content'] += line + '\n'

        # Añadir última sección
        if current_section['content'].strip():
            sections.append(current_section)

        return sections

    # ========== Métodos auxiliares privados ==========
    @staticmethod
    def _fix_encoding_issues(text: str) -> str:
        """Corrige problemas comunes de encoding."""
        # Reemplazos comunes de encoding mal interpretado
        replacements = {
            'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
            'Ã±': 'ñ', 'Ã¼': 'ü', 'Â': '', 'Ã': '',
            'â€œ': '"', 'â€': '"', 'â€™': "'", 'â€"': '—',
            '\u00a0': ' ',  # Non-breaking space
        }

        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)

        return text

    @staticmethod
    def _remove_urls(text: str) -> str:
        """Elimina URLs del texto."""
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        return url_pattern.sub('', text)

    @staticmethod
    def _remove_emails(text: str) -> str:
        """Elimina direcciones de email."""
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        return email_pattern.sub('', text)

    @staticmethod
    def _remove_control_characters(text: str) -> str:
        """Elimina caracteres de control excepto saltos de línea y tabulaciones."""
        return ''.join(
            char for char in text
            if unicodedata.category(char)[0] != 'C' or char in '\n\t\r'
        )

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normaliza espacios en blanco."""
        # Múltiples espacios -> un espacio
        text = re.sub(r'[ \t]+', ' ', text)

        # Múltiples saltos de línea -> máximo dos
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Eliminar espacios al inicio/final de líneas
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        return text.strip()

    @staticmethod
    def _fix_punctuation(text: str) -> str:
        """Corrige problemas comunes de puntuación."""
        # Espacio antes de puntuación
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)

        # Sin espacio después de puntuación
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)

        # Múltiples signos de puntuación
        text = re.sub(r'([!?.]){2,}', r'\1', text)

        return text

    @staticmethod
    def _remove_short_lines(text: str) -> str:
        """Elimina líneas muy cortas que probablemente sean headers/footers."""
        lines = text.split('\n')
        filtered_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Mantener líneas vacías para estructura
            if not stripped:
                filtered_lines.append(line)
                continue

            # Mantener si es suficientemente larga
            if len(stripped) >= 30:
                filtered_lines.append(line)
                continue

            # Mantener si parece un título (no termina en punto)
            if not stripped.endswith('.') and len(stripped.split()) > 2:
                filtered_lines.append(line)
                continue

            # Mantener si está entre líneas largas (párrafo continuo)
            prev_long = i > 0 and len(lines[i - 1].strip()) > 30
            next_long = i < len(lines) - 1 and len(lines[i + 1].strip()) > 30
            if prev_long and next_long:
                filtered_lines.append(line)

        return '\n'.join(filtered_lines)

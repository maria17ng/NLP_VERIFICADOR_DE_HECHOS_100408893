import re
from langchain_core.documents import Document
from preprocessor import DocumentPreprocessor


class WikipediaPreprocessor(DocumentPreprocessor):
    """
    Preprocesador especializado para artículos de Wikipedia.

    Elimina secciones estándar de Wikipedia que no aportan valor:
    - Referencias
    - Enlaces externos
    - Véase también
    - Bibliografía
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wiki_sections_to_remove = [
            'Referencias', 'referencias',
            'Enlaces externos', 'enlaces externos',
            'Véase también', 'véase también',
            'Bibliografía', 'bibliografía',
            'Notas', 'notas',
            'Further reading', 'See also', 'External links', 'References'
        ]

    def preprocess_document(self, document: Document) -> Document:
        """
        Preprocesa documento de Wikipedia.

        Args:
            document: Documento a preprocesar

        Returns:
            Documento preprocesado
        """
        # Eliminar secciones de Wikipedia
        text = self._remove_wiki_sections(document.page_content)
        document.page_content = text

        # Aplicar preprocesamiento general
        return super().preprocess_document(document)

    def _remove_wiki_sections(self, text: str) -> str:
        """Elimina secciones estándar de Wikipedia."""
        for section in self.wiki_sections_to_remove:
            # Buscar el inicio de la sección
            pattern = re.compile(
                rf'^(=+\s*{section}\s*=+|{section}\s*$)',
                re.MULTILINE | re.IGNORECASE
            )

            match = pattern.search(text)
            if match:
                # Eliminar desde esa sección hasta el final
                text = text[:match.start()].strip()

        return text

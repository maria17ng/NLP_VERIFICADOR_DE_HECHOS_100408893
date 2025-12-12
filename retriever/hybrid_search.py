import re
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from retriever.config import RetrievalConfig
from utils.utils import setup_logger


class HybridSearcher:
    """
    Combina búsqueda semántica con keyword matching.

    Implementa BM25-like scoring sobre el contenido de documentos
    para complementar la búsqueda vectorial.
    """

    def __init__(self, config: RetrievalConfig):
        """
        Inicializa el buscador híbrido.

        Args:
            config: Configuración del pipeline
        """
        self.config = config
        self.logger = setup_logger('HybridSearcher', level='DEBUG')

    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """
        Extrae keywords significativas de un texto.

        Prioriza números (fechas, años) que son críticos para fact-checking.

        Args:
            text: Texto a procesar

        Returns:
            Lista de keywords (números primero, luego palabras)
        """
        # Normalizar y tokenizar
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)

        # Filtrar stopwords comunes y palabras cortas
        stopwords = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'de', 'del', 'al', 'y', 'o', 'en', 'con', 'por', 'para',
            'que', 'es', 'su', 'se', 'como', 'más', 'fue', 'son',
            'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to',
            'of', 'for', 'is', 'are', 'was', 'were', 'be', 'been'
        }

        # Extraer años/fechas (4 dígitos) - ALTA PRIORIDAD para fact-checking
        years = re.findall(r'\b(1[89]\d{2}|20\d{2})\b', text)

        # Extraer palabras clave
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]

        # Agregar sinónimos para palabras clave de fact-checking
        key_terms_map = {
            'fundado': ['fundación', 'creado', 'registrado', 'establecido'],
            'fundación': ['fundado', 'creado', 'registrado'],
            'registrado': ['fundado', 'fundación', 'creado'],
        }

        expanded_keywords = []
        for kw in keywords:
            expanded_keywords.append(kw)
            if kw in key_terms_map:
                expanded_keywords.extend(key_terms_map[kw])

        # PRIORIDAD: Años primero, luego keywords expandidas
        return years + expanded_keywords

    def calculate_keyword_score(self, claim: str, document: Document) -> float:
        """
        Calcula score de keyword matching entre claim y documento.

        Da peso extra a años/fechas que son críticos para fact-checking.

        Args:
            claim: Afirmación
            document: Documento a evaluar

        Returns:
            Score de 0.0 a 1.0
        """
        import re

        claim_keywords = self.extract_keywords(claim)

        if not claim_keywords:
            return 0.0

        # Buscar en contenido y metadatos
        doc_text = document.page_content.lower()

        # Agregar texto de metadatos relevantes
        if 'keywords' in document.metadata:
            doc_text += ' ' + str(document.metadata['keywords']).lower()
        if 'title' in document.metadata:
            doc_text += ' ' + str(document.metadata['title']).lower()

        # Extraer años del claim y documento
        claim_years = set(re.findall(r'\b(1[89]\d{2}|20\d{2})\b', claim.lower()))
        doc_years = set(re.findall(r'\b(1[89]\d{2}|20\d{2})\b', doc_text))

        # Contar matches
        matches = 0
        year_matches = 0

        for keyword in claim_keywords:
            # Verificar si es un año
            is_year = re.match(r'^\d{4}$', keyword)

            count = doc_text.count(keyword.lower())
            if count > 0:
                if is_year:
                    # BONUS para años: peso x3
                    year_matches += min(count, 3) * 3
                else:
                    # Keywords normales
                    matches += min(count, 3)

        # Score total con bonus por años
        total_score = (matches + year_matches) / (len(claim_keywords) * 2)

        # BONUS adicional si hay overlap de años entre claim y documento
        if claim_years and claim_years.intersection(doc_years):
            total_score *= 1.5  # 50% bonus por match de años

        return min(total_score, 1.0)

    def hybrid_score(self, claim: str, documents: List[Document],
                     semantic_scores: Optional[List[float]] = None) -> List[Tuple[Document, float]]:
        """
        Combina scores semánticos con keyword matching.

        Args:
            claim: Afirmación
            documents: Lista de documentos
            semantic_scores: Scores semánticos previos (opcional)

        Returns:
            Lista de tuplas (documento, hybrid_score)
        """
        if not self.config.use_hybrid_search:
            default_score = 1.0 / len(documents) if documents else 0.0
            return [(doc, default_score) for doc in documents]

        hybrid_results = []

        for i, doc in enumerate(documents):
            keyword_score = self.calculate_keyword_score(claim, doc)

            # Combinar con score semántico si existe
            if semantic_scores and i < len(semantic_scores):
                semantic_score = semantic_scores[i]
            else:
                semantic_score = 1.0  # Asumir score neutral

            # Weighted combination
            hybrid = (
                    (1 - self.config.keyword_weight) * semantic_score +
                    self.config.keyword_weight * keyword_score
            )

            hybrid_results.append((doc, hybrid))

        self.logger.debug(
            f"Híbrido: {len([s for _, s in hybrid_results if s > 0.5])} docs con score > 0.5"
        )

        return hybrid_results

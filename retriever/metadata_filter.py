import re
from typing import List, Dict, Any, Tuple, Set

from langchain_core.documents import Document
from retriever.config import RetrievalConfig
from utils.utils import setup_logger


class MetadataFilter:
    """
    Filtra y puntúa documentos basándose en metadatos enriquecidos.

    Analiza coincidencias de:
    - Fechas mencionadas en el claim
    - Entidades (personas, organizaciones, lugares)
    - Keywords relevantes
    - Tipo de contenido
    """

    def __init__(self, config: RetrievalConfig):
        """
        Inicializa el filtro de metadatos.

        Args:
            config: Configuración del pipeline
        """
        self.config = config
        self.logger = setup_logger('MetadataFilter', level='DEBUG')

    @staticmethod
    def extract_entities_from_claim(claim: str) -> Dict[str, Set[str]]:
        """
        Extrae entidades básicas del claim (sin NER complejo).

        Args:
            claim: Afirmación a analizar

        Returns:
            Diccionario con tipos de entidades extraídas
        """
        entities = {
            'dates': set(),
            'numbers': set(),
            'keywords': set()
        }

        # Extraer fechas (años, fechas completas)
        date_patterns = [
            r'\b(19|20)\d{2}\b',  # Años 1900-2099
            r'\b\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\b',  # "6 de marzo de 1902"
            r'\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4}\b'
        ]

        for pattern in date_patterns:
            matches = re.finditer(pattern, claim, re.IGNORECASE)
            entities['dates'].update(m.group() for m in matches)

        # Extraer números (para datos estadísticos)
        number_pattern = r'\b\d+\b'
        numbers = re.finditer(number_pattern, claim)
        entities['numbers'].update(m.group() for m in numbers)

        # Keywords significativas (palabras con mayúsculas, > 4 chars)
        words = claim.split()
        keywords = [w.strip('.,!?;:') for w in words
                    if len(w) > 4 and not w.lower() in {'donde', 'cuando', 'quien', 'cual', 'como'}]
        entities['keywords'] = set(keywords)

        return entities

    @staticmethod
    def calculate_metadata_score(claim_entities: Dict[str, Set[str]], doc_metadata: Dict[str, Any]) -> float:
        """
        Calcula un score de coincidencia entre el claim y metadatos del documento.

        Args:
            claim_entities: Entidades extraídas del claim
            doc_metadata: Metadatos del documento

        Returns:
            Score de 0.0 a 1.0
        """
        score = 0.0
        factors = 0

        # 1. Coincidencia de fechas, da score incluso si no coincide la fecha exactamente
        # De forma que permita recuperar documentos con fechas diferentes y detectar asi contradicciones
        if claim_entities['dates'] and 'dates' in doc_metadata:
            doc_dates_str = str(doc_metadata['dates']).lower()
            date_matches = sum(1 for date in claim_entities['dates']
                               if str(date).lower() in doc_dates_str)
            if date_matches > 0:
                # Coincidencia exacta
                score += 0.4
                factors += 1
            elif len(doc_dates_str) > 0:
                # Documento tiene fechas pero no coinciden
                score += 0.15
                factors += 1

        # 2. Coincidencia de keywords
        if claim_entities['keywords'] and 'keywords' in doc_metadata:
            doc_keywords_str = str(doc_metadata['keywords']).lower()
            keyword_matches = sum(1 for kw in claim_entities['keywords']
                                  if kw.lower() in doc_keywords_str)
            if keyword_matches > 0:
                keyword_score = min(keyword_matches / len(claim_entities['keywords']), 1.0)
                score += keyword_score * 0.3
                factors += 1

        # 3. Coincidencia de entidades (personas, organizaciones)
        entity_fields = ['persons', 'organizations', 'locations']
        for field in entity_fields:
            if field in doc_metadata:
                doc_entities_str = str(doc_metadata[field]).lower()
                # Buscar keywords en entidades del documento
                entity_matches = sum(1 for kw in claim_entities['keywords']
                                     if kw.lower() in doc_entities_str)
                if entity_matches > 0:
                    score += 0.2
                    factors += 1
                    break  # Solo contar una vez

        # 4. Tipo de contenido (preferir histórico para fechas, biográfico para personas)
        if 'content_type' in doc_metadata:
            content_type = doc_metadata['content_type']
            if claim_entities['dates'] and content_type == 'historical':
                score += 0.1
                factors += 1
            elif any(kw[0].isupper() for kw in claim_entities['keywords']) and content_type == 'biographical':
                score += 0.1
                factors += 1

        # Normalizar
        if factors > 0:
            return min(score, 1.0)

        return 0.0

    def filter_and_score(self, claim: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Filtra documentos y asigna scores basándose en metadatos.

        Args:
            claim: Afirmación a verificar
            documents: Lista de documentos recuperados

        Returns:
            Lista de tuplas (documento, metadata_score)
        """
        if not self.config.use_metadata_filter:
            return [(doc, 0.0) for doc in documents]

        claim_entities = self.extract_entities_from_claim(claim)

        self.logger.debug(f"Entidades extraídas del claim: {claim_entities}")

        scored_docs = []
        for doc in documents:
            metadata_score = self.calculate_metadata_score(
                claim_entities,
                doc.metadata
            )
            scored_docs.append((doc, metadata_score))

        self.logger.debug(
            f"Documentos con metadata score > 0: "
            f"{sum(1 for _, s in scored_docs if s > 0)}/{len(scored_docs)}"
        )

        return scored_docs

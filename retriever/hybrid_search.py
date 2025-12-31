import re
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from retriever.config import RetrievalConfig
from utils.utils import setup_logger

try:
    from rank_bm25 import BM25Okapi

    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


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
        self.bm25 = None  # Se inicializa cuando se llama a build_bm25_index
        self.bm25_corpus = []  # Documentos indexados

        if not HAS_BM25:
            self.logger.warning("⚠️  rank_bm25 no disponible. Instalar con: pip install rank-bm25")
            self.logger.warning("    Usando keyword matching simple como fallback")

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

    @staticmethod
    def _tokenize_with_ngrams(text: str, max_n: int = 3, max_tokens: int = 512) -> List[str]:
        """Tokeniza texto en uni/bi/trigramas para preservar frases clave."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return []

        # Limitar para evitar explosión combinatoria
        tokens = tokens[:max_tokens]

        ngram_tokens: List[str] = []
        length = len(tokens)
        for n in range(1, max_n + 1):
            if length < n:
                break
            for i in range(length - n + 1):
                if n == 1:
                    ngram_tokens.append(tokens[i])
                else:
                    ngram_tokens.append('_'.join(tokens[i:i + n]))

        return ngram_tokens

    def build_bm25_index(self, documents: List[Document]) -> None:
        """
        Construye índice BM25 sobre colección de documentos.

        IMPORTANTE: Debe llamarse una vez con todos los documentos candidatos
        antes de usar bm25_score().

        Args:
            documents: Lista de documentos a indexar
        """
        if not HAS_BM25:
            self.logger.debug("BM25 no disponible, saltando indexación")
            return

        if not documents:
            self.logger.debug("BM25: no hay documentos que indexar en este lote")
            self.bm25 = None
            self.bm25_corpus = []
            return

        # Reiniciar corpus e índice para evitar reutilizar resultados de lotes previos
        self.bm25_corpus = documents

        # Tokenizar cada documento
        tokenized_corpus = []
        for doc in documents:
            # Combinar contenido + metadata relevante
            text = doc.page_content.lower()

            # Agregar metadata de contexto
            if 'title' in doc.metadata:
                text += ' ' + str(doc.metadata.get('title', '')).lower()
            if 'keywords' in doc.metadata:
                text += ' ' + str(doc.metadata.get('keywords', '')).lower()
            if 'entidades' in doc.metadata:
                entidades = doc.metadata.get('entidades', [])
                if entidades:
                    text += ' ' + ' '.join([str(e).lower() for e in entidades])

            # Tokenizar preservando uni/bi/trigramas
            tokens = self._tokenize_with_ngrams(text)
            tokenized_corpus.append(tokens)

        # Construir índice BM25 para el lote actual
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.logger.debug(f"✅ Índice BM25 construido con {len(tokenized_corpus)} documentos para esta consulta")

    def bm25_score(self, query: str) -> List[float]:
        """
        Calcula scores BM25 para la query sobre el corpus indexado.

        Args:
            query: Query de búsqueda

        Returns:
            Lista de scores BM25 (uno por documento en bm25_corpus)
        """
        if not HAS_BM25 or self.bm25 is None:
            # Fallback: retornar scores uniformes
            return [1.0] * len(self.bm25_corpus) if self.bm25_corpus else []

        # Tokenizar query alineado con el corpus
        query_tokens = self._tokenize_with_ngrams(query)

        # Calcular scores BM25
        scores = self.bm25.get_scores(query_tokens)

        # Normalizar scores a [0, 1]
        max_score = max(scores) if scores.size > 0 and max(scores) > 0 else 1.0
        normalized_scores = [float(s / max_score) for s in scores]

        return normalized_scores

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
        Combina scores semánticos con BM25 (o keyword matching de fallback).

        ESTRATEGIA
        - Si BM25 disponible: 50% semántico + 50% BM25
        - Si no: 70% semántico + 30% keyword matching simple

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

        # Construir índice BM25 para este lote (no reutilizar de consultas previas)
        if HAS_BM25 and documents:
            self.build_bm25_index(documents)

        # Obtener scores BM25 (o None si no disponible)
        bm25_scores = None
        if HAS_BM25 and self.bm25 is not None:
            bm25_scores = self.bm25_score(claim)
            if bm25_scores:
                self.logger.debug(
                    f"🔍 BM25: scores calculados (max={max(bm25_scores):.3f}, min={min(bm25_scores):.3f})"
                )

        hybrid_results = []

        for i, doc in enumerate(documents):
            # Determinar score de keywords (BM25 o fallback)
            if bm25_scores is not None and i < len(bm25_scores):
                keyword_score = bm25_scores[i]
                method = "BM25"
                weight = 0.5
            else:
                keyword_score = self.calculate_keyword_score(claim, doc)
                method = "keyword"
                weight = self.config.keyword_weight  # 30% por defecto

            # Combinar con score semántico si existe
            if semantic_scores and i < len(semantic_scores):
                semantic_score = semantic_scores[i]
            else:
                semantic_score = 1.0  # Asumir score neutral

            # Weighted combination
            hybrid = (1 - weight) * semantic_score + weight * keyword_score

            hybrid_results.append((doc, hybrid))

        # Log de resultados
        high_scores = len([s for _, s in hybrid_results if s > 0.5])
        method_str = "BM25" if bm25_scores is not None else "keyword matching"
        self.logger.debug(f"Híbrido ({method_str}): {high_scores} docs con score > 0.5")

        return hybrid_results

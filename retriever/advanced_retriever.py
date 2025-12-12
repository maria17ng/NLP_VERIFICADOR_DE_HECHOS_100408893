import os
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from retriever.config import RetrievalConfig
from retriever.metadata_filter import MetadataFilter
from retriever.hybrid_search import HybridSearcher
from retriever.diversity_selector import DiversitySelector
from retriever.query_decomposer import QueryDecomposer
from utils.utils import setup_logger


class AdvancedRetriever:
    """
    Pipeline completo de recuperación avanzada.

    Orquesta todos los componentes:
    1. Búsqueda vectorial inicial
    2. Filtrado por metadatos
    3. Búsqueda híbrida (semántica + keywords)
    4. Reranking con cross-encoder
    5. Diversificación de fuentes
    6. Aplicación de thresholds de relevancia
    """

    def __init__(self, vector_db: Chroma, reranker: Optional[CrossEncoder] = None,
                 config: Optional[RetrievalConfig] = None):
        """
        Inicializa el retriever avanzado.

        Args:
            vector_db: Base de datos vectorial
            reranker: Modelo de reranking (opcional)
            config: Configuración del pipeline (opcional)
        """
        self.vector_db = vector_db
        self.reranker = reranker
        self.config = config or RetrievalConfig()

        # Componentes modulares
        self.metadata_filter = MetadataFilter(self.config)
        self.hybrid_searcher = HybridSearcher(self.config)
        self.diversity_selector = DiversitySelector(self.config)
        self.query_decomposer = QueryDecomposer()

        self.logger = setup_logger('AdvancedRetriever', level='INFO')

        self.logger.info("AdvancedRetriever inicializado")
        self.logger.info(f"   • Metadata filtering: {self.config.use_metadata_filter}")
        self.logger.info(f"   • Hybrid search: {self.config.use_hybrid_search}")
        self.logger.info(f"   • Reranking: {self.config.use_reranker and reranker is not None}")
        self.logger.info(f"   • Diversity: {self.config.use_diversity}")

    @staticmethod
    def _normalize_query_for_search(query: str) -> str:
        """
        Normaliza la query para búsqueda semántica.

        IMPORTANTE: Para verificación de hechos, NO eliminamos fechas
        porque son críticas para distinguir afirmaciones correctas de incorrectas.
        En su lugar, hacemos normalización mínima para mejorar matching.

        Args:
            query: Query original

        Returns:
            Query normalizada (sin cambios drásticos)
        """
        import re
        # Solo normalizar espacios extras y capitalización inconsistente
        normalized = ' '.join(query.split())
        # NO eliminamos años - son críticos para fact-checking
        return normalized

    @staticmethod
    def _expand_query(query: str) -> List[str]:
        """
        Expande la query con sinónimos y variaciones para mejorar recall.

        ESTRATEGIA CRÍTICA para fact-checking:
        1. Query sin fecha específica (para encontrar documentos sobre el TEMA)
        2. Query con fecha original (para matching exacto)
        3. Variaciones con sinónimos

        Ejemplo:
        - "fundado en 1903" → ["Real Madrid fundación", "fundado 1903",
                               "Real Madrid creado", "registrado 1903"]

        Args:
            query: Query original

        Returns:
            Lista de queries expandidas (incluye original)
        """
        import re

        queries = []

        # Extraer componentes de la query
        query_lower = query.lower()

        # Extraer años (fechas de 4 dígitos)
        years = re.findall(r'\b(1[89]\d{2}|20\d{2})\b', query_lower)

        # Extraer entidad principal (ej: "Real Madrid", "Barcelona")
        entities = []
        for entity in ['real madrid', 'barcelona', 'atlético', 'madrid']:
            if entity in query_lower:
                entities.append(entity)

        # **CLAVE**: Query sin fecha para encontrar documentos sobre el TEMA
        if years and entities:
            # Remover la fecha de la query original
            query_without_year = query_lower
            for year in years:
                query_without_year = query_without_year.replace(year, '').strip()
            # Limpiar espacios múltiples
            query_without_year = ' '.join(query_without_year.split())
            queries.append(query_without_year)

        # Query original (con fecha)
        queries.append(query)

        # Diccionario de sinónimos para fact-checking
        synonyms = {
            'fundado': ['fundación', 'creado', 'creación', 'establecido', 'registrado', 'constituido'],
            'nació': ['nacimiento', 'nace'],
            'ganó': ['ganar', 'victoria', 'campeón', 'título'],
            'juega': ['jugar', 'disputa', 'estadio'],
        }

        # Generar variaciones con sinónimos (sin fecha)
        for word, syns in synonyms.items():
            if word in query_lower:
                for syn in syns[:2]:  # Solo top 2 sinónimos
                    # Crear query sin fecha + sinónimo
                    if years and entities:
                        expanded = f"{entities[0]} {syn}"
                        if expanded not in queries:
                            queries.append(expanded)

        # Limitar a 6 variaciones
        return queries[:6]

    def _apply_metadata_prefilter(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Pre-filtra documentos basado en metadata rica (fechas, temas LDA, entidades).

        MEJORADO: Usa temas detectados por Gensim LDA (más preciso que keywords).

        Args:
            query: Query original
            docs: Documentos candidatos

        Returns:
            Documentos filtrados y priorizados
        """
        import re

        # Extraer componentes de la query
        dates_in_query = re.findall(r'\b\d{4}\b', query)
        query_lower = query.lower()

        # Priorizar docs relevantes por metadata
        prioritized_docs = []
        other_docs = []

        for doc in docs:
            metadata = doc.metadata
            is_relevant = False
            relevance_score = 0.0

            # PRIORIDAD 1: Match de temas LDA (si disponible)
            # Los temas LDA son más precisos porque se basan en:
            # - Frecuencias de palabras reales
            # - Co-ocurrencias de términos
            # - Distribuciones latentes aprendidas del corpus
            if metadata.get('has_topics', False):
                doc_topics_str = metadata.get('topics', '')
                doc_main_topic = metadata.get('main_topic', '')

                # Buscar overlap entre términos de query y temas del documento
                for word in query_lower.split():
                    if len(word) > 3:  # Solo palabras significativas
                        if word in doc_topics_str.lower() or word in doc_main_topic.lower():
                            is_relevant = True
                            relevance_score += 0.5
                            break

            # PRIORIDAD 2: Docs con fechas (si query tiene fecha)
            if dates_in_query and metadata.get('tiene_fechas', False):
                is_relevant = True
                relevance_score += 0.3

            # PRIORIDAD 3: Docs con entidades relevantes
            if metadata.get('num_entidades', 0) > 0:
                relevance_score += 0.1

            # PRIORIDAD 4: Docs con hechos clave (fechas + acciones)
            if metadata.get('hechos_clave', ''):
                relevance_score += 0.2

            if is_relevant or relevance_score > 0:
                prioritized_docs.append((doc, relevance_score))
            else:
                other_docs.append((doc, 0.0))

        # Ordenar docs priorizados por score
        prioritized_docs.sort(key=lambda x: x[1], reverse=True)

        # Combinar: primero relevantes (sin score), luego otros
        result = [doc for doc, _ in prioritized_docs] + [doc for doc, _ in other_docs]

        if prioritized_docs:
            self.logger.debug(
                f"      🎯 Pre-filtro metadata (LDA): {len(prioritized_docs)} docs relevantes de {len(docs)}")

        return result

    def retrieve(self, query: str, return_scores: bool = False) -> Tuple[List[Document], Optional[List[float]]]:
        """
        Recupera documentos relevantes con pipeline avanzado.

        Pipeline completo:
        1. Vector search (k_initial docs) - con query normalizada
        2. Metadata filtering & scoring
        3. Hybrid search (semantic + keywords)
        4. Reranking con cross-encoder - con query original
        5. Threshold filtering
        6. Diversity selection
        7. Top-K final

        Args:
            query: Consulta para búsqueda
            return_scores: Si retornar scores finales

        Returns:
            Tupla con:
            - Lista de documentos recuperados
            - Lista de scores (si return_scores=True)
        """
        self.logger.info(f"Iniciando recuperación avanzada para: '{query[:60]}...'")

        # === FASE 0: Query Decomposition ===
        # Descomponer query en sub-queries (sin fecha, con fecha, keywords)
        sub_queries = self.query_decomposer.decompose(query)
        self.logger.debug(f"🔍 Query descompuesta en {len(sub_queries)} sub-queries:")
        for i, sq in enumerate(sub_queries, 1):
            self.logger.debug(f"   {i}. {sq}")

        # === FASE 1: Búsqueda Vectorial Inicial con Query Decomposition ===
        # Buscar con cada sub-query (prioridad a query sin fecha)
        if len(sub_queries) > 1:
            self.logger.debug(f"🔧 Usando {len(sub_queries)} sub-queries para mejor cobertura")
            for i, eq in enumerate(sub_queries[:3], 1):
                self.logger.debug(f"   {i}. {eq}")

        def has_numbers(s: str) -> bool:
            return bool(__import__('re').search(r"\b\d+\b", s))

        sub_queries_sorted = sorted(sub_queries, key=lambda s: (has_numbers(s), len(s)))

        # Buscar con cada sub-query y combinar resultados
        # PRIORIDAD: Primera query (sin números/fecha) tiene mayor peso
        self.logger.debug(f"[1/6] Búsqueda vectorial con query decomposition (k={self.config.k_initial})")
        all_docs_with_priority = []
        seen_ids = set()

        for priority_idx, search_query in enumerate(sub_queries_sorted):
            # k más alto para primera query (sin números/fecha) que es más importante
            k_for_query = self.config.k_initial if priority_idx == 0 else self.config.k_initial // 2
            docs = self.vector_db.similarity_search(search_query, k=k_for_query)

            # Agregar docs únicos con prioridad
            for doc in docs:
                doc_id = hash(doc.page_content[:100])
                if doc_id not in seen_ids:
                    # Dar prioridad a docs de primera query (sin fecha específica)
                    priority_score = 1.0 / (priority_idx + 1)  # 1.0, 0.5, 0.33, ...
                    all_docs_with_priority.append((doc, priority_score))
                    seen_ids.add(doc_id)

        # Ordenar por prioridad (primera query primero)
        all_docs_with_priority.sort(key=lambda x: x[1], reverse=True)

        # Tomar top k_initial después de combinar
        initial_docs = [doc for doc, _ in all_docs_with_priority[:self.config.k_initial]]

        # === FASE 1.5: Pre-filtro por metadata rica ===
        # Aplicar filtro inteligente por metadata (temas, fechas, etc.)
        initial_docs = self._apply_metadata_prefilter(query, initial_docs)

        # DEBUG: Mostrar primeros 3 chunks recuperados
        if initial_docs:
            self.logger.debug(f"📄 Primeros 3 chunks recuperados (de {len(all_docs_with_priority)} únicos):")
            for i, doc in enumerate(initial_docs[:3], 1):
                preview = doc.page_content[:100].replace('\n', ' ')
                self.logger.debug(f"   {i}. {preview}...")

        if not initial_docs:
            self.logger.warning("No se encontraron documentos en la búsqueda vectorial")
            return ([], []) if return_scores else []

        self.logger.debug(f"      → {len(initial_docs)} documentos recuperados (priorizados por relevancia temática)")

        # === FASE 2: Filtrado por Metadatos ===
        # IMPORTANTE: Usar query ORIGINAL para metadata filtering, no la última variación
        self.logger.debug("[2/6] Filtrado por metadatos")
        metadata_scored = self.metadata_filter.filter_and_score(query, initial_docs)

        # === FASE 3: Búsqueda Híbrida ===
        # IMPORTANTE: Usar query ORIGINAL para hybrid search
        self.logger.debug("[3/6] Búsqueda híbrida (semántica + keywords)")
        # Extraer docs de metadata_scored
        docs_for_hybrid = [doc for doc, _ in metadata_scored]
        hybrid_scored = self.hybrid_searcher.hybrid_score(query, docs_for_hybrid)

        # Combinar scores de metadata e hybrid
        combined_scored = []
        for i, (doc, hybrid_score) in enumerate(hybrid_scored):
            metadata_score = metadata_scored[i][1]
            # Boost combinado
            boost = metadata_score * self.config.metadata_boost
            final_score = hybrid_score + boost
            combined_scored.append((doc, final_score))

        # === FASE 4: Reranking ===
        # IMPORTANTE: Usar query ORIGINAL para reranking
        if self.config.use_reranker and self.reranker:
            self.logger.debug(f"[4/6] Reranking (top {self.config.rerank_top_k})")

            # Tomar top K para reranquear
            combined_scored.sort(key=lambda x: x[1], reverse=True)
            top_for_rerank = combined_scored[:self.config.rerank_top_k]

            # Garantizar candidatos de la sub-query sin números estén presentes
            # (si se perdieron en el combinado por pesos)
            topic_only_query = sub_queries_sorted[0] if sub_queries_sorted else query
            topic_only_docs = self.vector_db.similarity_search(topic_only_query, k=self.config.k_initial // 2)
            # Añadir algunos candidatos únicos de topic-only
            existing_set = {id(doc) for doc, _ in top_for_rerank}
            for doc in topic_only_docs[:10]:
                if id(doc) not in existing_set:
                    top_for_rerank.append((doc, self.config.min_relevance_score))
                    existing_set.add(id(doc))

            # Aplicar reranker con query ORIGINAL (incluye fecha para comparación)
            pairs = [[query, doc.page_content] for doc, _ in top_for_rerank]
            rerank_scores = self.reranker.predict(pairs)

            # Combinar scores
            reranked = []
            for (doc, prev_score), rerank_score in zip(top_for_rerank, rerank_scores):
                # Normalizar rerank score (suele estar en rango -10 a 10)
                normalized_rerank = (rerank_score + 10) / 20
                # Combinar: 70% rerank, 30% score previo
                combined = 0.7 * normalized_rerank + 0.3 * prev_score
                reranked.append((doc, combined, rerank_score))

            # Aplicar threshold de reranker
            reranked = [
                (doc, score, rs) for doc, score, rs in reranked
                if rs >= self.config.min_rerank_score
            ]

            self.logger.debug(
                f"      → {len(reranked)} documentos tras threshold "
                f"(min_score={self.config.min_rerank_score})"
            )

            # Mantener solo doc. y score combinado
            scored_docs = [(doc, score) for doc, score, _ in reranked]
        else:
            self.logger.debug("[4/6] Reranking deshabilitado, usando scores híbridos")
            scored_docs = combined_scored

        # === FASE 5: Threshold de Relevancia ===
        self.logger.debug(f"[5/6] Aplicando threshold mínimo ({self.config.min_relevance_score})")
        filtered_docs = [
            (doc, score) for doc, score in scored_docs
            if score >= self.config.min_relevance_score
        ]

        if not filtered_docs:
            self.logger.warning(
                f"Ningún documento supera el threshold de relevancia "
                f"({self.config.min_relevance_score})"
            )
            return ([], []) if return_scores else []

        self.logger.debug(f"      → {len(filtered_docs)} documentos relevantes")

        # === FASE 6: Diversificación ===
        self.logger.debug("[6/6] Diversificación de fuentes")
        diversified_docs = self.diversity_selector.diversify(filtered_docs)

        # === RESULTADO FINAL ===
        # Ordenar por score final y tomar top K
        diversified_docs.sort(key=lambda x: x[1], reverse=True)
        final_docs = diversified_docs[:self.config.final_top_k]

        self.logger.info(
            f"✅ Pipeline completado: {len(final_docs)} documentos finales "
            f"(desde {len(initial_docs)} iniciales)"
        )

        if return_scores:
            docs = [doc for doc, _ in final_docs]
            scores = [score for _, score in final_docs]
            return docs, scores

        return [doc for doc, _ in final_docs]

    def retrieve_with_context(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Recupera documentos y construye contexto formateado.

        Compatible con la interfaz del FactChecker original.

        Args:
            query: Consulta para búsqueda

        Returns:
            Tupla con:
            - context: String formateado con evidencia y citaciones
            - metadata_list: Lista de metadatos de documentos
        """
        docs, scores = self.retrieve(query, return_scores=True)

        if not docs:
            return "", []

        # Construir contexto con citaciones
        context_parts = []
        metadata_list = []

        for i, (doc, score) in enumerate(zip(docs, scores), 1):
            filename = os.path.basename(
                doc.metadata.get("source", "Desconocido")
            )

            # Construir citación
            citation = self._build_citation(doc.metadata)

            # Header con score
            header = f"--- DOCUMENTO {i}: {filename}{citation} [Score: {score:.3f}] ---"
            clean_content = " ".join(doc.page_content.split())

            context_parts.append(f"{header}\n{clean_content}\n")

            metadata_list.append({
                'filename': filename,
                'citation': citation,
                'score': score,
                'metadata': doc.metadata
            })

        context = "\n".join(context_parts)

        return context, metadata_list

    @staticmethod
    def _build_citation(metadata: Dict[str, Any]) -> str:
        """
        Construye citación desde metadatos.

        Args:
            metadata: Metadatos del documento

        Returns:
            String de citación formateada
        """
        citation = ""

        if "page" in metadata:
            page_num = metadata['page'] + 1
            citation = f" (Pág. {page_num})"
        elif "chunk_id" in metadata:
            chunk_id = metadata['chunk_id']
            total_chunks = metadata.get('total_chunks_in_file', '?')
            citation = f" (Sec. {chunk_id}/{total_chunks})"

        return citation

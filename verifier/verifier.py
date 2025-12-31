import time
import os
import hashlib
import random
from collections import Counter
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from sentence_transformers import CrossEncoder

# Importaciones locales
from language import ProcesadorMultilingue
from utils.utils import ConfigManager, setup_logger, load_prompts
from retriever import AdvancedRetriever, RetrievalConfig
from summarizer import EvidenceSummarizer

from pathlib import Path
from dotenv import load_dotenv
dotenv_path = Path('settings/.env')
load_dotenv(dotenv_path=dotenv_path)


class FactChecker:
    """
    Verificador de hechos basado en RAG con soporte multilingüe.

    Esta clase implementa un sistema completo de verificación que:
    1. Detecta el idioma de entrada y traduce si es necesario
    2. Recupera evidencia relevante de una base de datos vectorial
    3. Usa un LLM para determinar la veracidad
    4. Retorna respuestas en el idioma original del usuario
    5. Mantiene caché de consultas para optimización

    Attributes:
        config: Gestor de configuración del sistema
        logger: Logger para registro de eventos
        prompts: Plantillas de prompts para el LLM
        linguist: Procesador multilingüe para traducción
        cache: Diccionario para almacenar consultas previas
        embeddings: Modelo de embeddings para búsqueda semántica
        vector_db: Base de datos vectorial (ChromaDB)
        reranker: Modelo de reranking para mejorar recuperación
        advanced_retriever: Pipeline avanzado de recuperación
        llm: Modelo de lenguaje para generación
        chain: Cadena de procesamiento LangChain
    """

    SUPPORTED_TEAM_KEYWORDS = {
        "real_madrid": {
            "display": "Real Madrid",
            "keywords": (
                "real madrid",
                "real madrid club de fútbol",
                "real madrid club de futbol",
                "madridista",
                "madridistas",
                "merengue",
                "merengues",
                "santiago bernabéu",
                "santiago bernabeu",
                "bernabéu",
                "bernabeu"
            )
        },
        "atletico_madrid": {
            "display": "Atlético de Madrid",
            "keywords": (
                "atlético de madrid",
                "atletico de madrid",
                "atlético",
                "atletico",
                "atleti",
                "colchonero",
                "colchoneros",
                "rojiblanco",
                "rojiblancos",
                "cívitas metropolitano",
                "civitas metropolitano",
                "wanda metropolitano",
                "metropolitano"
            )
        },
        "getafe": {
            "display": "Getafe CF",
            "keywords": (
                "getafe",
                "getafe cf",
                "getafe club de fútbol",
                "getafe club de futbol",
                "azulón",
                "azulon",
                "azulones",
                "coliseum alfonso pérez",
                "coliseum alfonso perez",
                "alfonso pérez",
                "alfonso perez"
            )
        },
        "leganes": {
            "display": "CD Leganés",
            "keywords": (
                "leganés",
                "leganes",
                "club deportivo leganés",
                "club deportivo leganes",
                "cd leganés",
                "cd leganes",
                "pepineros",
                "pepineras",
                "butarque"
            )
        },
        "rayo_vallecano": {
            "display": "Rayo Vallecano",
            "keywords": (
                "rayo vallecano",
                "rayo",
                "vallecano",
                "rayista",
                "rayistas",
                "vallecas",
                "franja roja"
            )
        }
    }

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el sistema de verificación de hechos.

        Args:
            config_path: Ruta al archivo de configuración YAML

        Raises:
            FileNotFoundError: Si no se encuentran archivos de configuración
            Exception: Si falla la carga de modelos
        """
        # Configuración y logging
        self.config = ConfigManager(config_path)
        self.logger = setup_logger(
            name="FactChecker",
            level=self.config.get('logging.level', 'INFO'),
            log_file=os.path.join(
                self.config.get_path('logs'),
                'fact_checker.log'
            ),
            console=self.config.get('logging.console_enabled', True)
        )

        self.logger.info("=" * 70)
        self.logger.info("Iniciando Sistema de Verificación de Hechos")
        self.logger.info("=" * 70)

        # Cargar prompts
        self._load_prompts()

        # Forzar modo determinista si está configurado
        self._configure_determinism()

        # Inicializar componentes
        self._init_language_processor()
        self._init_cache()
        self._init_embeddings()
        self._init_vector_db()
        self._init_reranker()
        self._init_advanced_retriever()
        self._init_llm()
        self._init_summarizer()

        self.logger.info("✅ Sistema inicializado correctamente")
        self.logger.info("=" * 70)

    def _load_prompts(self) -> None:
        """Carga las plantillas de prompts desde archivo YAML."""
        try:
            prompts_path = self.config.get_path('prompts')
            self.prompts = load_prompts(prompts_path)
            self.logger.info(f"Prompts cargados desde: {prompts_path}")
        except FileNotFoundError as e:
            self.logger.error(f"❌ Error cargando prompts: {e}")
            raise

    def _configure_determinism(self) -> None:
        """Configura semillas globales para reducir la aleatoriedad."""
        if not self.config.get('deterministic_mode.enabled', False):
            return

        seed = int(self.config.get('deterministic_mode.seed', 42))
        random.seed(seed)
        np.random.seed(seed)

        try:
            import torch  # type: ignore
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            # Torch puede no estar disponible en todos los entornos; continuar silenciosamente
            pass

        self.logger.info(f"🔒 Modo determinista activado (seed={seed})")

    def _init_language_processor(self) -> None:
        """Inicializa el procesador multilingüe."""
        try:
            lid_model_path = self.config.get_path('lid_model')
            self.linguist = ProcesadorMultilingue(model_path=lid_model_path)
            self.logger.info("Procesador multilingüe inicializado")
        except Exception as e:
            self.logger.warning(f"⚠️  Error inicializando procesador de idiomas: {e}")
            self.linguist = None

    def _init_cache(self) -> None:
        """Inicializa el sistema de caché."""
        if self.config.get('cache.enabled', True):
            self.cache = {}
            self.cache_max_size = self.config.get('cache.max_size', 1000)
            self.logger.info(f"Caché habilitado (tamaño máx: {self.cache_max_size})")
        else:
            self.cache = None
            self.logger.info("Caché deshabilitado")

    def _init_embeddings(self) -> None:
        """Inicializa el modelo de embeddings."""
        try:
            # Verificar si usar OpenAI o HuggingFace
            provider = self.config.get('models.embeddings.provider', 'huggingface')

            if provider == 'openai':
                # Usar OpenAI embeddings
                api_key = self.config.get('models.openai.api_key') or os.getenv('OPENAI_KEY')
                if not api_key:
                    self.logger.warning("⚠️  OpenAI API key no encontrada, usando HuggingFace como fallback")
                    provider = 'huggingface'
                else:
                    model_name = self.config.get('models.embeddings.openai_model', 'text-embedding-3-small')
                    self.logger.info(f"Cargando OpenAI embeddings: {model_name}")
                    self.embeddings = OpenAIEmbeddings(
                        model=model_name,
                        openai_api_key=api_key
                    )
                    self.logger.info(f"✅ Embeddings cargados: OpenAI {model_name}")
                    return

            # Fallback o default: HuggingFace
            if provider == 'huggingface':
                model_name = self.config.get('models.embeddings.name')
                self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
                self.logger.info(f"Embeddings cargados: {model_name}")
        except Exception as e:
            self.logger.error(f"❌ Error cargando embeddings: {e}")
            raise

    def _init_vector_db(self) -> None:
        """Inicializa la conexión a la base de datos vectorial."""
        try:
            db_path = self.config.get_path('vector_store')

            if os.path.exists(db_path):
                self.vector_db = Chroma(
                    persist_directory=db_path,
                    embedding_function=self.embeddings
                )
                # Obtener número de documentos
                collection = self.vector_db._collection
                doc_count = collection.count()
                self.logger.info(f"✅ Base de datos vectorial conectada: {db_path}")
                self.logger.info(f"\tDocumentos en BD: {doc_count}")
            else:
                self.vector_db = None
                self.logger.warning(f"⚠️  No se encontró la base de datos vectorial en: {db_path}")
                self.logger.warning("   Por favor, ejecuta ingest_data.py primero")

        except Exception as e:
            self.logger.error(f"❌ Error conectando a la base de datos: {e}")
            self.vector_db = None

    def _init_reranker(self) -> None:
        """Inicializa el modelo de reranking."""
        try:
            model_name = self.config.get('models.reranker.name')
            self.reranker = CrossEncoder(model_name)
            self.logger.info(f"Reranker cargado: {model_name}")
        except Exception as e:
            self.logger.warning(f"⚠️  Error cargando reranker: {e}")
            self.reranker = None

    def _init_advanced_retriever(self) -> None:
        """Inicializa el pipeline avanzado de recuperación."""
        if not self.vector_db:
            self.logger.warning("⚠️  No se puede inicializar AdvancedRetriever sin vector_db")
            self.advanced_retriever = None
            return

        try:
            # Configuración del retriever desde config.yaml
            rag_config = self.config.get_rag_config()

            retrieval_config = RetrievalConfig(
                k_initial=rag_config.get('similarity_search', {}).get('k', 50),
                use_metadata_filter=rag_config.get('advanced_retrieval', {}).get('use_metadata_filter', True),
                metadata_boost=rag_config.get('advanced_retrieval', {}).get('metadata_boost', 0.2),
                use_hybrid_search=rag_config.get('advanced_retrieval', {}).get('use_hybrid_search', True),
                keyword_weight=rag_config.get('advanced_retrieval', {}).get('keyword_weight', 0.3),
                use_reranker=self.reranker is not None,
                rerank_top_k=rag_config.get('advanced_retrieval', {}).get('rerank_top_k', 20),
                use_diversity=rag_config.get('advanced_retrieval', {}).get('use_diversity', True),
                max_chunks_per_source=rag_config.get('advanced_retrieval', {}).get('max_chunks_per_source', 3),
                diversity_penalty=rag_config.get('advanced_retrieval', {}).get('diversity_penalty', 0.1),
                min_relevance_score=rag_config.get('advanced_retrieval', {}).get('min_relevance_score', 0.3),
                min_rerank_score=rag_config.get('advanced_retrieval', {}).get('min_rerank_score', -5.0),
                final_top_k=rag_config.get('reranking', {}).get('top_k', 5)
            )

            self.advanced_retriever = AdvancedRetriever(
                vector_db=self.vector_db,
                reranker=self.reranker,
                config=retrieval_config
            )

            self.logger.info("Pipeline avanzado de recuperación inicializado")

        except Exception as e:
            self.logger.error(f"❌ Error inicializando AdvancedRetriever: {e}")
            self.advanced_retriever = None

    def _init_llm(self) -> None:
        """Inicializa el modelo de lenguaje (OpenAI u Ollama) y la cadena de procesamiento."""
        # Inicializar atributos por defecto
        self.llm = None
        self.chain = None

        try:
            llm_config = self.config.get_model_config('llm')
            openai_config = self.config.config.get('models', {}).get('openai', {})
            deterministic_mode = self.config.get('deterministic_mode.enabled', False)

            # Selector: OpenAI si está habilitado, sino Ollama
            if openai_config.get('enabled', False):
                # OpenAI
                api_key = openai_config.get('api_key') or os.getenv('OPENAI_KEY')
                if not api_key:
                    raise ValueError(
                        "OpenAI enabled pero no se encontró api_key. "
                        "Añade api_key en config.yaml o variable OPENAI_KEY."
                    )

                base_temperature = openai_config.get('temperature', 0.1)
                temperature = 0.0 if deterministic_mode else base_temperature
                top_p = openai_config.get('top_p', 1.0)
                top_p = max(min(top_p, 1.0), 1e-5)

                self.llm = ChatOpenAI(
                    model=openai_config.get('model', 'gpt-4o-mini'),
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=openai_config.get('max_tokens', 800),
                    api_key=api_key
                )
                self.logger.info(f"LLM inicializado: OpenAI {openai_config.get('model', 'gpt-4o-mini')}")
                self.logger.info(
                    f"  Temperatura: {temperature}{' (determinista)' if deterministic_mode else ''}"
                )
            else:
                # Ollama (comportamiento original)
                llm_params = {
                    'model': llm_config.get('name', 'llama3.2'),
                    'temperature': 0.0 if deterministic_mode else llm_config.get('temperature', 0.1),
                    'format': llm_config.get('format', 'json'),
                    'keep_alive': llm_config.get('keep_alive', '5m')
                }

                # Si hay URL base y API key (para UC3M)
                if 'base_url' in llm_config:
                    llm_params['base_url'] = llm_config['base_url']
                if 'api_key' in llm_config:
                    llm_params['api_key'] = llm_config['api_key']

                self.llm = ChatOllama(**llm_params)
                self.logger.info(f"LLM inicializado: Ollama {llm_params['model']}")
                self.logger.info(
                    f"  Temperatura: {llm_params['temperature']}{' (determinista)' if deterministic_mode else ''}"
                )

            # Verificar que LLM se inicializó
            if self.llm is None:
                raise RuntimeError("El LLM no se inicializó correctamente")

            # Inicializar cadena
            self.logger.info("Creando cadena de procesamiento...")

            if 'verification_prompt' not in self.prompts:
                raise KeyError("No se encontró 'verification_prompt' en los prompts cargados")

            prompt_template = ChatPromptTemplate.from_template(
                self.prompts['verification_prompt']
            )
            self.chain = prompt_template | self.llm | JsonOutputParser()

            # Verificar que la cadena se creó
            if self.chain is None:
                raise RuntimeError("La cadena de procesamiento no se inicializó correctamente")

            self.logger.info("✅ Cadena de procesamiento inicializada correctamente")

        except Exception as e:
            self.logger.error(f"❌ Error inicializando LLM: {e}")
            import traceback
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

    def _init_summarizer(self) -> None:
        """Inicializa el generador de resúmenes."""
        try:
            # Pasar el LLM al summarizer para resúmenes abstractivos
            self.summarizer = EvidenceSummarizer(llm=self.llm)
            self.logger.info("Generador de resúmenes inicializado")
        except Exception as e:
            self.logger.warning(f"⚠️  Error inicializando summarizer: {e}")
            self.summarizer = None

    def retrieve_context(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Recupera evidencia relevante de la base de datos vectorial.

        Este método usa el pipeline avanzado de recuperación:
        1. Búsqueda vectorial inicial (k documentos)
        2. Filtrado por metadatos enriquecidos
        3. Búsqueda híbrida (semántica + keywords)
        4. Reranking con cross-encoder
        5. Diversificación de fuentes
        6. Aplicación de thresholds de relevancia

        Args:
            query: Consulta para búsqueda de evidencia

        Returns:
            Tupla con:
            - context: String formateado con evidencia y citaciones
            - metadata: Lista de metadatos de documentos recuperados
        """
        if not self.vector_db:
            self.logger.warning("No hay base de datos vectorial disponible")
            return "", []

        # Usar AdvancedRetriever si está disponible
        if self.advanced_retriever:
            try:
                self.logger.info("Usando pipeline avanzado de recuperacion")
                context, metadata_list = self.advanced_retriever.retrieve_with_context(query)
                self.logger.debug(f"Contexto construido con {len(metadata_list)} fragmentos (pipeline avanzado)")

                if metadata_list:
                    self.logger.debug("Documentos recuperados:")
                    for i, meta in enumerate(metadata_list[:5], 1):
                        filename = meta.get("filename", "Unknown")
                        citation = meta.get("citation", "")
                        self.logger.debug(f"\t {i}.{filename} {citation}")
                return context, metadata_list
            except Exception as e:
                self.logger.error(f"❌ Error en AdvancedRetriever, usando fallback: {e}")
                # Continuar con método básico como fallback
                import traceback
                traceback.print_exception(e)

        # === FALLBACK: Método básico original ===
        self.logger.warning("⚠️  Usando pipeline de recuperación básico (fallback)")

        # Configuración RAG
        rag_config = self.config.get_rag_config()
        k_initial = rag_config.get('similarity_search', {}).get('k', 50)
        top_k_rerank = rag_config.get('reranking', {}).get('top_k', 5)

        # 1. Búsqueda vectorial inicial
        self.logger.debug(f"Buscando documentos similares (k={k_initial})")
        docs = self.vector_db.similarity_search(query, k=k_initial)

        if not docs:
            self.logger.warning("No se encontraron documentos relevantes")
            return "", []

        self.logger.debug(f"\tEncontrados {len(docs)} documentos iniciales")

        # 2. Reranking (si está disponible)
        if self.reranker:
            self.logger.debug("Aplicando reranking")
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.reranker.predict(pairs)
            scored_docs = sorted(
                zip(docs, scores),
                key=lambda x: x[1],
                reverse=True
            )
            top_docs = [doc for doc, score in scored_docs[:top_k_rerank]]
            self.logger.debug(
                f"\tTop {top_k_rerank} documentos tras reranking"
            )
        else:
            top_docs = docs[:top_k_rerank]

        # 3. Construcción del contexto con citaciones
        context_parts = []
        metadata_list = []

        for doc in top_docs:
            filename = os.path.basename(doc.metadata.get("source", "Desconocido"))

            # Citación granular
            citation = self._build_citation(doc.metadata)

            header = f"--- DOCUMENTO: {filename}{citation} ---"
            clean_content = " ".join(doc.page_content.split())

            context_parts.append(f"{header}\n{clean_content}\n")
            metadata_list.append({
                'filename': filename,
                'citation': citation,
                'metadata': doc.metadata
            })

        context = "\n".join(context_parts)
        self.logger.debug(f"Contexto construido con {len(top_docs)} fragmentos")

        return context, metadata_list

    @staticmethod
    def _build_citation(metadata: Dict[str, Any]) -> str:
        """
        Construye una citación precisa basada en los metadatos del documento.

        Args:
            metadata: Metadatos del documento

        Returns:
            String con la citación formateada
        """
        citation = ""

        # Si es PDF (tiene número de página)
        if "page" in metadata:
            page_num = metadata['page'] + 1  # PyPDF usa índice 0
            citation = f" (Pág. {page_num})"

        # Si es TXT con chunks (secciones)
        elif "chunk_id" in metadata:
            chunk_id = metadata['chunk_id']
            total_chunks = metadata.get('total_chunks_in_file', '?')
            citation = f" (Sec. {chunk_id}/{total_chunks})"

        return citation

    def _calculate_confidence(self, verdict: str, context: str, metadata_list: List[Dict[str, Any]],
                              claim: str = "", explanation: str = "") -> int:
        """
        Calcula un nivel de confianza basado en la evidencia recuperada.

        El nivel de confianza se basa en:
        - Similitud semántica entre claim y explicación (coseno de embeddings)
        - Scores de los documentos recuperados (calidad de la evidencia)
        - Número de fuentes diversas
        - Penalización por explicaciones genéricas
        - Coherencia entre fuentes

        Args:
            verdict: Veredicto del sistema (VERDADERO, FALSO, etc.)
            context: Contexto recuperado
            metadata_list: Metadatos de documentos recuperados
            claim: Claim original del usuario
            explanation: Explicación generada por el LLM

        Returns:
            Nivel de confianza de 0 a 5
        """
        if verdict == "NO SE PUEDE VERIFICAR" or not context:
            return 0

        confidence_score = 0.0  # Score flotante (0-5)

        # FACTOR 1: Similitud semántica claim-explicación (0-2 puntos)
        if claim and explanation and hasattr(self, 'embeddings'):
            try:
                # Generar embeddings para claim y explicación
                claim_embedding = self.embeddings.embed_query(claim)
                explanation_embedding = self.embeddings.embed_query(explanation)

                # Calcular similitud coseno usando numpy
                claim_vec = np.array(claim_embedding)
                expl_vec = np.array(explanation_embedding)

                cosine_sim = np.dot(claim_vec, expl_vec) / (
                        np.linalg.norm(claim_vec) * np.linalg.norm(expl_vec)
                )

                # Convertir similitud (0-1) a puntos (0-2)
                # Alta similitud = explicación específica y relevante
                similarity_points = max(0, min(2.0, float(cosine_sim) * 2.5))
                confidence_score += similarity_points

                self.logger.debug(f"Similitud coseno: {cosine_sim:.4f} → +{similarity_points:.2f} puntos")

            except Exception as e:
                self.logger.debug(f"No se pudo calcular similitud semántica: {e}")
                # Fallback: dar 1 punto base
                confidence_score += 1.0
                self.logger.debug(f"  +1.0 punto (fallback)")

        # FACTOR 2: Calidad de los documentos recuperados (0-2 puntos)
        if metadata_list:
            # Extraer scores de los metadatos (si existen)
            scores = []
            for meta in metadata_list:
                if isinstance(meta, dict) and 'score' in meta:
                    scores.append(meta['score'])

            self.logger.debug(f"Scores extraídos: {scores} (de {len(metadata_list)} docs)")

            if scores:
                avg_score = sum(scores) / len(scores)
                self.logger.debug(f"Score promedio: {avg_score:.3f}")

                # Score promedio > 0.7 = alta relevancia
                if avg_score >= 0.7:
                    confidence_score += 2.0
                    self.logger.debug(f"  +2.0 puntos (score >= 0.7)")
                elif avg_score >= 0.6:
                    confidence_score += 1.5
                    self.logger.debug(f"  +1.5 puntos (score >= 0.6)")
                elif avg_score >= 0.5:
                    confidence_score += 1.0
                    self.logger.debug(f"  +1.0 punto (score >= 0.5)")
                else:
                    confidence_score += 0.5
                    self.logger.debug(f"  +0.5 puntos (score < 0.5)")
            else:
                # Sin scores, dar 1 punto base si hay documentos
                confidence_score += 1.0
                self.logger.debug(f"  +1.0 punto (sin scores, fallback)")

        # FACTOR 3: Número de fuentes únicas (0-1 punto)
        num_sources = len(set(meta.get('source', '') for meta in metadata_list if isinstance(meta, dict)))
        if num_sources >= 3:
            confidence_score += 1.0
        elif num_sources >= 2:
            confidence_score += 0.5

        # PENALIZACIÓN: Explicaciones genéricas/vagas
        if explanation:
            import re
            vague_indicators = [
                r"confirma la información",
                r"según la evidencia",
                r"evidencia confirma",
                r"información mencionada"
            ]

            explanation_lower = explanation.lower()
            vague_count = sum(1 for pattern in vague_indicators
                              if re.search(pattern, explanation_lower))

            if vague_count >= 2:
                confidence_score *= 0.7  # Penalizar 30%
                self.logger.debug(f"Penalización vaga (2+ patrones): x0.7 → {confidence_score:.2f}")
            elif vague_count == 1:
                confidence_score *= 0.85  # Penalizar 15%
                self.logger.debug(f"Penalización vaga (1 patrón): x0.85 → {confidence_score:.2f}")

        # Convertir a escala 0-5 (entero)
        final_confidence = max(0, min(5, round(confidence_score)))
        self.logger.debug(f"📊 Confianza final: {confidence_score:.2f} → {final_confidence}/5")

        return final_confidence

    @staticmethod
    def _reduce_context(claim: str, context: str, max_sentences: int = 10) -> str:
        """
        Reduce el contexto a las frases más ancla para el claim.

        Criterios genéricos (no dependientes de dominio):
        - Coincidencia de entidades del claim (palabras capitalizadas de 2+ términos o tokens significativos)
        - Coincidencia de verbo/acción común (fundó/creó/ganó/...)
        - Presencia de números de 4 dígitos (años)

        Args:
            claim: Afirmación del usuario (en español preferentemente)
            context: Contexto completo concatenado de múltiples documentos
            max_sentences: Máximo de frases a retornar

        Returns:
            Subconjunto de frases relevantes concatenadas, preservando encabezados de documento
        """
        import re

        if not context:
            return context

        # Extraer entidades simples del claim (palabras capitalizadas compuestas o tokens > 3 chars)
        entities = set()
        entities.update(re.findall(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)\b", claim))
        # Añadir keywords en minúscula significativas del claim
        entities.update([w for w in re.findall(r"\b\w{4,}\b", claim.lower()) if not w.isdigit()])

        # Verbos/acciones comunes en verificación de hechos (genérico en español)
        action_terms = {
            'fundado', 'fundación', 'creado', 'creación', 'establecido', 'registrado',
            'ganó', 'ganar', 'victoria', 'campeón', 'título', 'obtuvo', 'logró', 'consiguió',
            'nació', 'muerte', 'falleció', 'es', 'fue', 'fueron', 'son'
        }

        # Dividir en bloques por documento para preservar encabezados
        blocks = re.split(r"(--- DOCUMENTO\s+\d+: .*?---)", context)
        reduced_parts: List[str] = []
        selected_count = 0

        def score_sentence(s: str) -> int:
            s_low = s.lower()
            score = 0
            # Entidades/keywords
            score += sum(1 for e in entities if e and e.lower() in s_low)
            # Verbos/acciones
            score += sum(1 for a in action_terms if a in s_low)
            # Años
            if re.search(r"\b(1[89]\d{2}|20\d{2})\b", s):
                score += 2
            return score

        for i in range(0, len(blocks), 2):
            header = blocks[i]
            body = blocks[i + 1] if i + 1 < len(blocks) else ''
            if header and header.strip().startswith('--- DOCUMENTO'):
                reduced_parts.append(header.strip())
                continue

            text = header if header else body
            if not text or text.strip().startswith('--- DOCUMENTO'):
                continue

            # Dividir a oraciones de forma sencilla
            sentences = re.split(r"(?<=[\.!?])\s+", text)
            # Puntuar y seleccionar top
            scored = [(s, score_sentence(s)) for s in sentences if len(s.strip()) > 20]
            scored.sort(key=lambda x: x[1], reverse=True)

            for s, sc in scored:
                if sc <= 0:
                    continue
                reduced_parts.append(s.strip())
                selected_count += 1
                if selected_count >= max_sentences:
                    break
            if selected_count >= max_sentences:
                break

        # Si no se seleccionó nada, devolver primeras frases razonables
        if selected_count == 0:
            sentences = re.split(r"(?<=[\.!?])\s+", context)
            fallback = [s.strip() for s in sentences if len(s.strip()) > 20][:max_sentences]
            return "\n".join(fallback)

        return "\n".join(reduced_parts)

    @staticmethod
    def _match_keyword_group(text: str, keyword_map: Dict[str, Dict[str, Tuple[str, ...]]]) -> Optional[Dict[str, str]]:
        """Detecta si el texto contiene alguna palabra clave definida en un diccionario."""
        if not text:
            return None

        text_lower = text.lower()
        for key, data in keyword_map.items():
            for keyword in data.get('keywords', ()):  # type: ignore[arg-type]
                if keyword and keyword in text_lower:
                    return {
                        "key": key,
                        "display": data.get('display', key.replace('_', ' ').title()),
                        "matched": keyword
                    }
        return None

    def _apply_domain_guard(self, claim: str) -> Optional[Dict[str, Any]]:
        """Aplica una guard clause cuando el claim se sale del dominio cubierto."""
        supported_hit = self._match_keyword_group(claim, self.SUPPORTED_TEAM_KEYWORDS)
        if supported_hit:
            return None

        self.logger.warning("🛑 Claim fuera de dominio detectado")

        message = (
            "El corpus actual solo cubre clubes madrileños (Real Madrid, Atlético, Getafe, "
            "Leganés y Rayo Vallecano). No hay evidencia interna para verificar hechos sobre."
        )

        return {
            "veredicto": "NO SE PUEDE VERIFICAR",
            "nivel_confianza": 0,
            "fuente_documento": "Fuera del dominio del corpus",
            "explicacion_corta": message,
            "evidencia_citada": "Ninguna",
            "fragmentos_evidencia": [],
            "fuentes": [],
            "resumen_evidencia": "No disponible"
        }

    def _apply_structured_rules(self, claim: str, context: str,
                                metadata_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Aplica reglas determinísticas antes de invocar al LLM."""
        override = self._detect_foundation_year_override(claim, context, metadata_list)
        if override:
            return override
        return None

    def _detect_foundation_year_override(self, claim: str, context: str,
                                         metadata_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detecta contradicciones explícitas sobre el año de fundación."""
        import re

        if not claim or not context:
            return None

        claim_lower = claim.lower()
        if not any(token in claim_lower for token in ("fundad", "fundación", "fundacion", "registrad", "registro")):
            return None

        claim_year_match = re.search(r"(1[89]\d{2}|20\d{2})", claim_lower)
        if not claim_year_match:
            return None
        claim_year = claim_year_match.group(1)

        team_hit = self._match_keyword_group(claim, self.SUPPORTED_TEAM_KEYWORDS)
        if not team_hit:
            return None

        keywords = tuple(keyword for keyword in self.SUPPORTED_TEAM_KEYWORDS[team_hit['key']]['keywords'])
        # Patrón mejorado: solo acepta años MUY cercanos (máx 30 caracteres) a palabras de fundación
        # Esto evita falsos positivos como "club fundado... que en 1947 hizo X"
        foundation_pattern = re.compile(
            r"(?:fundad[oa]|registrad[oa]|inscrito|constituy[oó]|legaliz[aó]|acta).{0,30}?((?:18|19|20)\d{2})",
            re.IGNORECASE
        )

        sentences = re.split(r"(?<=[\.!?])\s+", context)
        evidence_candidates: List[Tuple[str, str]] = []

        for sentence in sentences:
            cleaned = sentence.strip()
            if not cleaned:
                continue

            sentence_lower = cleaned.lower()
            if not any(keyword in sentence_lower for keyword in keywords):
                continue
            if not any(token in sentence_lower for token in ("fundad", "fundación", "fundacion", "registr", "acta", "legaliz")):
                continue

            for match in foundation_pattern.finditer(cleaned):
                year = match.group(1)
                # Filtro adicional: verificar que el año esté en contexto directo de fundación
                # Buscar 50 caracteres antes y después del año encontrado
                match_start = match.start(1)
                context_window = cleaned[max(0, match_start - 50):min(len(cleaned), match_start + 50)].lower()

                # Rechazar si hay palabras que indican otros eventos (no fundación)
                exclusion_keywords = ["acuerdo", "contrato", "filial", "fichaje", "traspaso", "convenio"]
                if any(excl in context_window for excl in exclusion_keywords):
                    continue

                evidence_candidates.append((year, cleaned))

        if not evidence_candidates:
            return None

        year_counter = Counter(year for year, _ in evidence_candidates)
        if not year_counter:
            return None

        best_year, _ = year_counter.most_common(1)[0]
        if best_year == claim_year:
            return None

        best_sentence = next((sentence for year, sentence in evidence_candidates if year == best_year), "")

        first_source = metadata_list[0] if metadata_list else {}
        source_label = first_source.get('filename', 'Documentación del corpus')
        citation = first_source.get('citation', '')
        if citation:
            source_label = f"{source_label}{citation}"

        self.logger.info(
            f"⚖️ Regla de fundación aplicada: {team_hit['display']} → {best_year} (claim decía {claim_year})"
        )

        return {
            "veredicto": "FALSO",
            "nivel_confianza": 4,
            "fuente_documento": source_label,
            "explicacion_corta": (
                f"La documentación indica que {team_hit['display']} se fundó oficialmente en {best_year}, "
                f"no en {claim_year}."
            ),
            "evidencia_citada": best_sentence or "La fuente primaria menciona el año correcto de fundación."
        }

    def verify(self, claim_usuario: str) -> Dict[str, Any]:
        """
        Verifica la veracidad de una afirmación.

        Este es el método principal del sistema. Proceso completo:
        1. Detecta idioma y traduce a español si es necesario
        2. Verifica caché para consultas repetidas
        3. Recupera evidencia de la base de datos
        4. Evalúa con el LLM
        5. Traduce respuesta al idioma original
        6. Retorna resultado con métricas

        Args:
            claim_usuario: Afirmación a verificar (en cualquier idioma)

        Returns:
            Diccionario con:
            - veredicto: VERDADERO, FALSO, o NO SE PUEDE VERIFICAR
            - nivel_confianza: 0-5
            - fuente_documento: Archivo(s) que respaldan el veredicto
            - explicacion_corta: Justificación del veredicto
            - evidencia_citada: Fragmento relevante de la evidencia
            - fuentes: Lista de fuentes con citaciones
            - fragmentos_evidencia: Lista de fragmentos recuperados
            - tiempo_procesamiento: Tiempo total en segundos
            - origen: LLM o CACHÉ
            - idioma_respuesta: Idioma de la respuesta
            - calidad_traduccion: % de confianza en la traducción
        """
        start_time = time.time()
        self.logger.info("=" * 70)
        self.logger.info(f"Nueva verificación: '{claim_usuario[:100]}...'")

        # --- PASO 1: PROCESAMIENTO DE IDIOMA ---
        claim_es, idioma_orig, calidad = self._process_input_language(
            claim_usuario
        )

        # --- PASO 2: GUARD CLAUSE DE DOMINIO ---
        guard_result = self._apply_domain_guard(claim_es)

        if guard_result:
            result = guard_result
            origen = "GUARD_CLAUSE"
        else:
            # --- PASO 3: VERIFICAR CACHÉ ---
            claim_hash = self._get_cache_key(claim_es)
            cached_result = self._check_cache(claim_hash)

            if cached_result:
                result = cached_result
                origen = "CACHÉ"
                self.logger.info("Resultado obtenido de caché")
            else:
                # --- PASO 4: RECUPERAR EVIDENCIA ---
                self.logger.debug(f"Buscando evidencia para: {claim_es}")
                context, metadata_list = self.retrieve_context(claim_es)

                if context:
                    context_preview = context[:500].replace('\n', ' ')
                    self.logger.debug(f"Contexto recuperado: {context_preview}")
                    self.logger.debug(f"Recuperados {len(metadata_list)} fragmentos de evidencia")
                else:
                    self.logger.warning(f"⚠️ No se recuperó ningún contexto")

                # --- PASO 4BIS: APLICAR REGLAS DETERMINÍSTICAS ---
                structured_override = self._apply_structured_rules(
                    claim_es,
                    context,
                    metadata_list
                ) if context else None

                if structured_override:
                    result = structured_override
                else:
                    # --- PASO 5: EVALUAR CON LLM ---
                    result = self._evaluate_claim(
                        claim_es,
                        claim_usuario,
                        context,
                        calidad,
                        metadata_list
                    )

                # --- PASO 5.5: AGREGAR FUENTES Y FRAGMENTOS ---
                result['fuentes'] = self._format_sources(metadata_list)
                result['fragmentos_evidencia'] = self._extract_evidence_fragments(context, metadata_list)

                # --- PASO 5.6: GENERAR RESUMEN (si hay fragmentos) ---
                if self.summarizer and result['fragmentos_evidencia']:
                    try:
                        resumen = self.summarizer.generate_summary(
                            result['fragmentos_evidencia'],
                            claim_es,
                            method="extractive",
                            max_sentences=2
                        )
                        result['resumen_evidencia'] = resumen
                        self.logger.debug(f"Resumen generado: {resumen[:100]}...")
                    except Exception as e:
                        self.logger.warning(f"⚠️  Error generando resumen: {e}")
                        result['resumen_evidencia'] = "No disponible"

                # --- PASO 6: GUARDAR EN CACHÉ ---
                self._save_to_cache(claim_hash, result)
                origen = "LLM"

        # --- PASO 6: TRADUCIR RESPUESTA ---
        final_result = self._translate_response(result, idioma_orig)

        # --- PASO 7: AÑADIR MÉTRICAS ---
        tiempo_total = round(time.time() - start_time, 3)
        final_result.update({
            "tiempo_procesamiento": f"{tiempo_total}s",
            "origen": origen,
            "calidad_traduccion": f"{int(calidad * 100)}%",
            "idioma_respuesta": idioma_orig
        })

        self.logger.info(f"✅ Verificación completada en {tiempo_total}s")
        self.logger.info(f"\tVeredicto: {final_result.get('veredicto')}")
        self.logger.info(f"\tConfianza: {final_result.get('nivel_confianza')}/5")
        self.logger.info("=" * 70)

        return final_result

    def _process_input_language(self, claim_usuario: str) -> Tuple[str, str, float]:
        """
        Procesa el idioma de entrada y traduce si es necesario.

        Args:
            claim_usuario: Afirmación original del usuario

        Returns:
            Tupla con (claim_en_español, idioma_original, calidad_traducción)
        """
        if self.linguist:
            claim_es, idioma_orig, calidad = self.linguist.procesar_entrada(
                claim_usuario
            )
            self.logger.info(f"Idioma detectado: {idioma_orig}")
            if idioma_orig != 'es':
                self.logger.info(f"\tTraducción realizada (calidad: {int(calidad * 100)}%)")
            return claim_es, idioma_orig, calidad
        else:
            # Sin procesador de idiomas, asumir español
            return claim_usuario, 'es', 1.0

    @staticmethod
    def _get_cache_key(claim: str) -> str:
        """
        Genera una clave de caché para una afirmación.

        Args:
            claim: Afirmación normalizada

        Returns:
            Hash MD5 de la afirmación
        """
        normalized = claim.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Verifica si existe un resultado en caché.

        Args:
            cache_key: Clave de caché

        Returns:
            Resultado cacheado o None
        """
        if self.cache is not None and cache_key in self.cache:
            self.logger.debug(f"Cache HIT: {cache_key[:8]}...")
            return self.cache[cache_key].copy()
        return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """
        Guarda un resultado en caché.

        Args:
            cache_key: Clave de caché
            result: Resultado a guardar
        """
        if self.cache is not None:
            # Gestión de tamaño máximo (FIFO simple)
            if len(self.cache) >= self.cache_max_size:
                # Eliminar el primer elemento
                first_key = next(iter(self.cache))
                del self.cache[first_key]

            self.cache[cache_key] = result.copy()
            self.logger.debug(f"Guardado en caché: {cache_key[:8]}...")

    @staticmethod
    def _check_context_relevance(claim: str, context: str) -> float:
        """
        Verifica si el contexto es relevante para la afirmación usando similitud básica.

        Args:
            claim: Afirmación a verificar
            context: Contexto recuperado

        Returns:
            Score de relevancia (0.0 a 1.0)
        """
        if not context or not claim:
            return 0.0

        # Extraer palabras clave de la afirmación (simple)
        claim_words = set(claim.lower().split())
        context_lower = context.lower()

        # Contar cuántas palabras clave aparecen en el contexto
        matches = sum(1 for word in claim_words if len(word) > 3 and word in context_lower)
        relevance = matches / max(len([w for w in claim_words if len(w) > 3]), 1)

        return min(relevance, 1.0)

    @staticmethod
    def _validate_llm_response(result: Dict[str, Any], claim: str, context: str) -> Dict[str, Any]:
        """
        Valida la respuesta del LLM para detectar alucinaciones.

        Verifica que la explicación del LLM sea consistente con el claim.
        Por ejemplo, si el claim dice "1902" pero la explicación habla de "1950",
        es una alucinación.

        Args:
            result: Respuesta del LLM
            claim: Afirmación original
            context: Contexto usado

        Returns:
            Resultado validado (o corregido si es necesario)
        """
        import re

        # Extraer explicación
        explicacion = result.get('explicacion_corta', '')

        # Extraer años del claim y de la explicación
        claim_years = set(re.findall(r'\b(1[89]\d{2}|20\d{2})\b', claim))
        expl_years = set(re.findall(r'\b(1[89]\d{2}|20\d{2})\b', explicacion))

        # Si la explicación menciona años que NO están en el claim, es sospechoso
        extra_years = expl_years - claim_years

        if extra_years and result.get('veredicto') == 'FALSO':
            # El LLM está comparando con un año que no está en el claim
            # Esto indica confusión - probablemente debería ser VERDADERO o NO SE PUEDE VERIFICAR

            # Verificar si el año del claim SÍ aparece en el contexto
            context_years = set(re.findall(r'\b(1[89]\d{2}|20\d{2})\b', context))

            if claim_years and claim_years.intersection(context_years):
                # El año del claim SÍ está en el contexto → debería ser VERDADERO
                return {
                    "veredicto": "VERDADERO",
                    "nivel_confianza": 4,
                    "fuente_documento": result.get('fuente_documento', 'Corregido por validación'),
                    "explicacion_corta": f"La evidencia confirma la información mencionada en la afirmación.",
                    "evidencia_citada": result.get('evidencia_citada', 'Validado por sistema')
                }

        # Si no hay problema, devolver el resultado original
        return result

    @staticmethod
    def _validate_vague_explanations(result: Dict[str, Any], claim: str, context: str) -> Dict[str, Any]:
        """
        Detecta explicaciones vagas y convierte VERDADERO → NO SE PUEDE VERIFICAR cuando corresponde.

        Ejemplos de explicaciones vagas:
        - "La evidencia confirma la información mencionada en la afirmación." (sin especificar QUÉ)
        - "La evidencia confirma que [X]" pero X NO es el sujeto principal del claim

        Args:
            result: Resultado del LLM
            claim: Claim original
            context: Contexto recuperado

        Returns:
            Resultado validado
        """
        import re

        verdict = result.get('veredicto', '')
        explanation = result.get('explicacion_corta', '')

        # Solo validar si el veredicto es VERDADERO
        if verdict != "VERDADERO":
            return result

        # Patrones de explicaciones genéricas/vagas
        vague_patterns = [
            r"la evidencia confirma la información mencionada en la afirmación",
            r"la evidencia confirma que.{0,50}$",  # Muy corta
            r"confirma la información",
            r"la evidencia valida",
        ]

        explanation_lower = explanation.lower()
        is_vague = any(re.search(pattern, explanation_lower) for pattern in vague_patterns)

        if is_vague:
            # Verificar si el claim menciona un sujeto específico que NO está en el contexto
            # Ejemplo: Claim "El Atlético de Madrid es madrileño" pero contexto solo menciona
            # "derbi madrileño" entre Real Madrid y Atlético (no describe al Atlético directamente)

            # Extraer sujeto principal del claim (primera entidad capitalizada)
            match = re.search(
                r"(El|La|Los|Las)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[a-záéíóúñ]+)*(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)",
                claim)
            if match:
                subject = match.group(2).strip()

                # Verificar si el contexto DESCRIBE al sujeto (no solo lo menciona)
                # Buscar patrones como "X es", "X fue", "X tiene", etc.
                descriptive_patterns = [
                    rf"{re.escape(subject)}\s+(es|fue|tiene|cuenta con|se fundó|fundado)",
                    rf"sobre\s+{re.escape(subject)}",
                    rf"historia\s+de\s+{re.escape(subject)}",
                ]

                has_description = any(re.search(pattern, context, re.IGNORECASE)
                                      for pattern in descriptive_patterns)

                if not has_description:
                    # ÚLTIMO CHECK: ¿El claim pregunta algo genérico como "es madrileño"?
                    # Si el contexto menciona al sujeto en contexto de Madrid, podría ser válido
                    if "madrid" in claim.lower() and subject.lower() in context.lower():
                        # Es un caso límite - dejarlo pasar
                        return result

                    # Definitivamente vago
                    return {
                        "veredicto": "NO SE PUEDE VERIFICAR",
                        "nivel_confianza": 0,
                        "fuente_documento": result.get('fuente_documento', 'Corregido por validación'),
                        "explicacion_corta": f"La evidencia menciona '{subject}' de pasada pero no describe sus características directamente.",
                        "evidencia_citada": result.get('evidencia_citada', 'Validado por sistema')
                    }

        return result

    def _evaluate_claim(self, claim_es: str, claim_original: str, context: str, translation_quality: float,
                        metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evalúa una afirmación usando el LLM.

        Args:
            claim_es: Afirmación en español
            claim_original: Afirmación original del usuario
            context: Contexto recuperado
            translation_quality: Calidad de la traducción (0-1)
            metadata_list: Metadatos de documentos recuperados

        Returns:
            Diccionario con el veredicto y detalles
        """
        # Si no hay contexto, no se puede verificar
        if not context:
            self.logger.warning("❌ No se encontró contexto relevante")
            return {
                "veredicto": "NO SE PUEDE VERIFICAR",
                "explicacion_corta": "No se encontró información relevante en la base de datos.",
                "fuente_documento": "Ninguno",
                "nivel_confianza": 0,
                "evidencia_citada": "Ninguna"
            }

        # NUEVA VALIDACIÓN: Verificar relevancia del contexto
        relevance_score = self._check_context_relevance(claim_es, context)
        self.logger.debug(f"Relevancia del contexto: {relevance_score:.2f}")

        if relevance_score < 0.30:  # Umbral más exigente de relevancia
            self.logger.warning(
                f"⚠️ Contexto poco relevante (score: {relevance_score:.2f}). Reintentando retrieval sin números...")

            # Fallback genérico: intentar recuperar usando la query sin números/fechas
            import re as _re
            claim_topic_only = _re.sub(r"\b\d+\b", " ", claim_es)
            claim_topic_only = " ".join(claim_topic_only.split())

            new_context, new_metadata = self.retrieve_context(claim_topic_only)
            if new_context:
                self.logger.debug("🔁 Fallback: contexto alternativo recuperado para evaluación")
                context = new_context
                metadata_list = new_metadata
            else:
                self.logger.warning("🔁 Fallback sin resultados. Se devuelve NO SE PUEDE VERIFICAR")
                return {
                    "veredicto": "NO SE PUEDE VERIFICAR",
                    "explicacion_corta": "La evidencia encontrada no trata sobre el tema de la afirmación.",
                    "fuente_documento": "Ninguno",
                    "nivel_confianza": 0,
                    "evidencia_citada": "Ninguna"
                }

        # Preparar prompt con advertencia de traducción si aplica
        prompt_claim = claim_es
        threshold = self.config.get(
            'language.translation_confidence_threshold',
            0.6
        )

        if translation_quality < threshold:
            self.logger.warning(
                f"⚠️  Calidad de traducción baja ({int(translation_quality * 100)}%)"
            )
            prompt_claim += (
                f" [NOTA: Posible error de traducción. "
                f"Original: '{claim_original}']"
            )

        # Usar contexto completo (top 3 chunks) sin reducción
        context_to_send = context
        self.logger.info(f"📄 Usando contexto completo sin reducción ({len(context)} caracteres)")
        self.logger.debug(f"Contexto enviado al LLM (preview): {context_to_send[:500]}...")

        # Invocar al LLM
        self.logger.info("Evaluando con LLM...")
        try:
            # Verificar que self.chain existe
            if not hasattr(self, 'chain') or self.chain is None:
                raise RuntimeError(
                    "La cadena de procesamiento (self.chain) no está disponible. "
                    "El sistema no se inicializó correctamente."
                )

            result = self.chain.invoke({
                "context": context,
                "claim": prompt_claim
            })

            # POST-PROCESAMIENTO: Validar respuesta del LLM
            # Si el LLM da una explicación que menciona fechas/datos que no están en el claim,
            # es una alucinación y debemos corregir
            result = self._validate_llm_response(result, claim_es, context)

            # NUEVA VALIDACIÓN: Detectar explicaciones vagas que indican falta de evidencia
            result = self._validate_vague_explanations(result, claim_es, context)

            # Calcular confianza basada en evidencia
            if 'nivel_confianza' not in result or result['nivel_confianza'] == 0:
                result['nivel_confianza'] = self._calculate_confidence(
                    verdict=result.get('veredicto', ''),
                    context=context,
                    metadata_list=metadata_list,
                    claim=claim_es,
                    explanation=result.get('explicacion_corta', '')
                )

            return result

        except Exception as e:
            self.logger.error(f"❌ Error en evaluación con LLM: {e}")
            import traceback
            self.logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            return {
                "error": f"Fallo del modelo: {str(e)}",
                "veredicto": "ERROR",
                "nivel_confianza": 0
            }

    def _translate_response(self, result: Dict[str, Any], target_language: str) -> Dict[str, Any]:
        """
        Traduce la respuesta al idioma objetivo.

        Args:
            result: Resultado en español
            target_language: Idioma objetivo

        Returns:
            Resultado traducido
        """
        if target_language == 'es' or not self.linguist:
            return result.copy()

        translated_result = result.copy()

        # Traducir campos de texto
        if 'explicacion_corta' in translated_result:
            translated_result['explicacion_corta'] = self.linguist.procesar_salida(
                translated_result['explicacion_corta'],
                target_language
            )

        if 'veredicto' in translated_result:
            translated_result['veredicto'] = self.linguist.procesar_salida(
                translated_result['veredicto'],
                target_language
            )

        # La evidencia citada NO se traduce para mantener fidelidad
        self.logger.debug(f"🌍 Respuesta traducida a: {target_language}")

        return translated_result

    @staticmethod
    def _format_sources(metadata_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Formatea la lista de metadatos en fuentes legibles.

        Args:
            metadata_list: Lista de metadatos de documentos

        Returns:
            Lista de diccionarios con información de fuentes
        """
        sources = []
        for meta in metadata_list:
            source = {
                'documento': meta.get('filename', 'Desconocido'),
                'citacion': meta.get('citation', ''),
                'seccion': meta.get('metadata', {}).get('chunk_id', ''),
            }
            sources.append(source)
        return sources

    @staticmethod
    def _extract_evidence_fragments(context: str, metadata_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extrae fragmentos de evidencia del contexto con sus metadatos.

        Args:
            context: Contexto completo
            metadata_list: Lista de metadatos

        Returns:
            Lista de fragmentos con información
        """
        import re

        fragments = []
        # Dividir el contexto por documentos
        doc_parts = re.split(r'(--- DOCUMENTO \d+:.*?---)', context)

        for i, meta in enumerate(metadata_list, 1):
            # Buscar el fragmento correspondiente
            fragment_text = ""
            for j in range(len(doc_parts)):
                if f"DOCUMENTO {i}:" in doc_parts[j]:
                    # El texto está en la siguiente parte
                    if j + 1 < len(doc_parts):
                        fragment_text = doc_parts[j + 1].strip()[:300]  # Primeros 300 chars
                    break

            if fragment_text:
                fragments.append({
                    'documento': meta.get('filename', 'Desconocido'),
                    'citacion': meta.get('citation', ''),
                    'fragmento': fragment_text
                })

        return fragments

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema.

        Returns:
            Diccionario con estadísticas:
            - cache_size: Número de elementos en caché
            - vector_db_docs: Número de documentos en la BD
            - config: Configuración actual
        """
        stats = {
            'cache_size': len(self.cache) if self.cache else 0,
            'cache_max_size': self.cache_max_size if self.cache else 0,
            'vector_db_connected': self.vector_db is not None,
            'reranker_available': self.reranker is not None,
            'multilingual_enabled': self.linguist is not None
        }

        if self.vector_db:
            try:
                stats['vector_db_docs'] = self.vector_db._collection.count()
            except Exception as e:
                stats['vector_db_docs'] = 'N/A'
                self.logger.warning(f"Error while {e}")

        return stats

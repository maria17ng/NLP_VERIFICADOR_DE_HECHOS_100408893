import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Importaciones locales
from utils.utils import ConfigManager, setup_logger
from preprocessor import DocumentPreprocessor, WikipediaPreprocessor
from chunker.semantic_chunker import SemanticChunker
from chunker.hybrid_chunker import HybridChunker
from chunker.section_aware import SectionAwareChunker
from extractor import MetadataExtractor, ChunkMetadataEnricher, FactMetadataExtractor, TopicExtractor
from hyde import HyDEGenerator, HyDEAugmenter, SimpleHyDEGenerator


class DocumentIngester:
    """
    Gestor de ingesta de documentos a base de datos vectorial.

    Esta clase se encarga de:
    1. Cargar documentos desde archivos (TXT, PDF)
    2. Dividirlos en fragmentos manejables (chunking)
    3. Añadir metadatos para citación precisa
    4. Almacenarlos en ChromaDB con embeddings

    Attributes:
        config: Gestor de configuración
        logger: Logger para registro de eventos
        embeddings: Modelo de embeddings
        text_splitter: Divisor de texto configurado
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el sistema de ingesta.

        Args:
            config_path: Ruta al archivo de configuración
        """
        # Configuración
        self.config = ConfigManager(config_path)

        # Logger
        self.logger = setup_logger(
            name="DocumentIngester",
            level=self.config.get('logging.level', 'INFO'),
            log_file=os.path.join(
                self.config.get_path('logs'),
                'ingest.log'
            ),
            console=self.config.get('logging.console_enabled', True)
        )

        self.logger.info("=" * 70)
        self.logger.info("📂 Iniciando Sistema de Ingesta de Documentos")
        self.logger.info("=" * 70)

        # Inicializar componentes
        self._init_embeddings()
        self._init_preprocessor()
        self._init_text_splitter()
        self._init_metadata_extractor()
        self._init_fact_metadata_extractor()
        self._init_topic_extractor()
        self._init_hyde()

    def _init_embeddings(self) -> None:
        """Inicializa el modelo de embeddings."""
        try:
            model_name = self.config.get('models.embeddings.name')
            self.logger.info(f"Cargando modelo de embeddings: {model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'trust_remote_code': True}  # Soporte para modelos con código personalizado
            )
            self.logger.info("✅ Embeddings cargados correctamente")
        except Exception as e:
            self.logger.error(f"❌ Error cargando embeddings: {e}")
            raise

    def _init_preprocessor(self) -> None:
        """Inicializa el preprocesador de documentos."""
        preproc_config = self.config.get('rag.preprocessing', {})

        if not preproc_config.get('enabled', True):
            self.preprocessor = None
            self.logger.warning(" Preprocesamiento deshabilitado")
            return

        # Usar preprocesador específico para Wikipedia si está habilitado
        if preproc_config.get('wikipedia_mode', True):
            self.preprocessor = WikipediaPreprocessor(
                remove_urls=preproc_config.get('remove_urls', True),
                remove_emails=preproc_config.get('remove_emails', True),
                normalize_whitespace=preproc_config.get('normalize_whitespace', True),
                fix_encoding=preproc_config.get('fix_encoding', True),
                min_paragraph_length=preproc_config.get('min_paragraph_length', 50)
            )
            self.logger.info("Preprocesador Wikipedia inicializado")
        else:
            self.preprocessor = DocumentPreprocessor(
                remove_urls=preproc_config.get('remove_urls', True),
                remove_emails=preproc_config.get('remove_emails', True),
                normalize_whitespace=preproc_config.get('normalize_whitespace', True),
                fix_encoding=preproc_config.get('fix_encoding', True),
                min_paragraph_length=preproc_config.get('min_paragraph_length', 50)
            )
            self.logger.info("Preprocesador estándar inicializado")

    def _init_text_splitter(self) -> None:
        """Inicializa el divisor de texto."""
        chunk_config = self.config.get('rag.chunking', {})
        strategy = chunk_config.get('strategy', 'semantic')

        if strategy == 'semantic':
            semantic_config = chunk_config.get('semantic', {})
            self.text_splitter = SemanticChunker(
                chunk_size=chunk_config.get('chunk_size', 1000),
                chunk_overlap=chunk_config.get('chunk_overlap', 200),
                respect_sentences=semantic_config.get('respect_sentences', True),
                min_chunk_size=semantic_config.get('min_chunk_size', 100),
                max_chunk_size=semantic_config.get('max_chunk_size', 2000)
            )
            self.logger.info("Chunker semántico configurado")

        elif strategy == 'hybrid':
            hybrid_config = chunk_config.get('hybrid', {})
            self.text_splitter = HybridChunker(
                small_chunk_size=hybrid_config.get('small_chunk_size', 512),
                large_chunk_size=hybrid_config.get('large_chunk_size', 1500),
                chunk_overlap=hybrid_config.get('chunk_overlap', 100)
            )
            self.logger.info("Chunker híbrido configurado")

        elif strategy == 'section_aware':
            semantic_config = chunk_config.get('semantic', {})
            self.text_splitter = SectionAwareChunker(
                chunk_size=chunk_config.get('chunk_size', 1000),
                chunk_overlap=chunk_config.get('chunk_overlap', 200),
                respect_sentences=semantic_config.get('respect_sentences', True),
                min_chunk_size=semantic_config.get('min_chunk_size', 100),
                max_chunk_size=semantic_config.get('max_chunk_size', 2000)
            )
            self.logger.info("Chunker por secciones configurado")

        else:  # 'basic' o fallback
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_config.get('chunk_size', 1000),
                chunk_overlap=chunk_config.get('chunk_overlap', 200),
                add_start_index=chunk_config.get('add_start_index', True)
            )
            self.logger.info("Chunker básico configurado")

        self.logger.info(
            f"   Tamaño objetivo: {chunk_config.get('chunk_size', 1000)}, "
            f"Overlap: {chunk_config.get('chunk_overlap', 200)}"
        )

    def _init_metadata_extractor(self) -> None:
        """Inicializa el extractor de metadatos."""
        metadata_config = self.config.get('rag.metadata_extraction', {})

        if not metadata_config.get('enabled', True):
            self.metadata_extractor = None
            self.chunk_enricher = None
            self.logger.warning("Extracción de metadatos deshabilitada")
            return

        self.metadata_extractor = MetadataExtractor(
            extract_dates=metadata_config.get('extract_dates', True),
            extract_entities=metadata_config.get('extract_entities', True),
            classify_content=metadata_config.get('classify_content', True)
        )

        self.chunk_enricher = ChunkMetadataEnricher()

        self.logger.info("Extractor de metadatos inicializado")

    def _init_fact_metadata_extractor(self) -> None:
        """Inicializa el extractor de metadata para fact-checking."""
        try:
            self.fact_metadata_extractor = FactMetadataExtractor()
            self.logger.info("✅ FactMetadataExtractor inicializado para fact-checking")
        except Exception as e:
            self.logger.warning(f"⚠️ Error inicializando FactMetadataExtractor: {e}")
            self.fact_metadata_extractor = None

    def _init_topic_extractor(self) -> None:
        """Inicializa el extractor de temas con Gensim LDA."""
        topic_config = self.config.get('rag.topic_modeling', {})

        if not topic_config.get('enabled', True):
            self.topic_extractor = None
            self.logger.warning("⚠️ Topic modeling deshabilitado")
            return

        try:
            self.topic_extractor = TopicExtractor(
                num_topics=topic_config.get('num_topics', 10),
                passes=topic_config.get('passes', 10)
            )
            self.logger.info("✅ TopicExtractor (Gensim LDA) inicializado")
        except Exception as e:
            self.logger.warning(f"⚠️ Error inicializando TopicExtractor: {e}")
            self.topic_extractor = None

    def _init_hyde(self) -> None:
        """Inicializa el generador HyDE."""
        hyde_config = self.config.get('rag.hyde', {})

        if not hyde_config.get('enabled', False):
            self.hyde_augmenter = None
            self.logger.warning("HyDE deshabilitado")
            return

        # Intentar usar generador completo, fallback a simple
        try:
            generator = HyDEGenerator(
                num_questions=hyde_config.get('num_questions', 3),
                min_chunk_length=hyde_config.get('min_chunk_length', 100)
            )
        except Exception:
            self.logger.warning("Usando HyDE simplificado (sin spaCy)")
            generator = SimpleHyDEGenerator(
                num_questions=hyde_config.get('num_questions', 3)
            )

        self.hyde_augmenter = HyDEAugmenter(
            generator=generator,
            create_question_docs=hyde_config.get('create_question_docs', True)
        )

        self.logger.info("Generador HyDE inicializado")

    def load_documents(self, data_path: str = None) -> List[Document]:
        """
        Carga documentos desde un directorio.

        Soporta archivos .txt y .pdf. Los archivos se cargan con metadatos
        básicos que luego se enriquecen durante el chunking.

        Args:
            data_path: Ruta al directorio de datos (usa config si no se especifica)

        Returns:
            Lista de documentos cargados

        Raises:
            FileNotFoundError: Si el directorio no existe
        """
        if data_path is None:
            data_path = self.config.get_path('data_raw')

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"❌ El directorio de datos no existe: {data_path}"
            )

        self.logger.info(f"Buscando documentos en: {data_path}")

        all_docs = []

        # 1. Cargar archivos TXT
        txt_docs = self._load_txt_files(data_path)
        all_docs.extend(txt_docs)

        # 2. Cargar archivos PDF
        pdf_docs = self._load_pdf_files(data_path)
        all_docs.extend(pdf_docs)

        if not all_docs:
            self.logger.warning(
                "No se encontraron documentos. "
                "Asegúrate de tener archivos .txt o .pdf en el directorio."
            )
        else:
            self.logger.info(
                f"✅ Total de documentos cargados: {len(all_docs)}"
            )

        return all_docs

    def _load_txt_files(self, data_path: str) -> List[Document]:
        """
        Carga archivos de texto plano (.txt).

        Args:
            data_path: Directorio de datos

        Returns:
            Lista de documentos de texto
        """
        try:
            loader = DirectoryLoader(
                data_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'},
                show_progress=True
            )
            docs = loader.load()
            self.logger.info(f"   Archivos TXT cargados: {len(docs)}")
            return docs
        except Exception as e:
            self.logger.warning(f"️  Error cargando archivos TXT: {e}")
            return []

    def _load_pdf_files(self, data_path: str) -> List[Document]:
        """
        Carga archivos PDF.

        Args:
            data_path: Directorio de datos

        Returns:
            Lista de documentos PDF
        """
        pdf_files = list(Path(data_path).rglob("*.pdf"))

        if not pdf_files:
            self.logger.info("   No se encontraron archivos PDF")
            return []

        self.logger.info(f"   Procesando {len(pdf_files)} archivos PDF...")

        all_docs = []
        for pdf_file in tqdm(pdf_files, desc="Cargando PDFs"):
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                self.logger.warning(
                    f"  Error cargando {pdf_file.name}: {e}"
                )

        self.logger.info(f"   ✅ Páginas de PDF cargadas: {len(all_docs)}")
        return all_docs

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Divide documentos en fragmentos (chunks) manejables.

        Pipeline completo:
        1. Preprocesamiento (limpieza, normalización)
        2. Extracción de metadatos del documento
        3. Chunking (semántico, híbrido o básico)
        4. Enriquecimiento de metadatos de chunks
        5. Generación HyDE (si está habilitado)

        Args:
            documents: Lista de documentos a fragmentar

        Returns:
            Lista de fragmentos con metadatos enriquecidos
        """
        self.logger.info("Iniciando pipeline de procesamiento de documentos...")

        # 1. Preprocesamiento
        if self.preprocessor:
            self.logger.info("\tPreprocesando documentos...")
            documents = [self.preprocessor.preprocess_document(doc) for doc in tqdm(documents, desc="Preprocesando")]

        # 2. Extracción de metadatos a nivel documento
        if self.metadata_extractor:
            self.logger.info("\tExtrayendo metadatos de documentos...")
            documents = [self.metadata_extractor.extract_metadata(doc) for doc in tqdm(documents, desc="Extrayendo metadatos")]

        # 3. Chunking
        self.logger.info("\tFragmentando documentos...")
        chunks = self.text_splitter.split_documents(documents)
        self.logger.info(f"\tFragmentos generados: {len(chunks)}")

        # 4. Añadir metadatos de ubicación (mantener compatibilidad)
        self.logger.info("\tAñadiendo metadatos de ubicación...")
        enriched_chunks = self._add_location_metadata(chunks)

        # 5. Enriquecer metadatos de chunks individuales
        if self.chunk_enricher:
            self.logger.info("\tEnriqueciendo metadatos de chunks...")
            for chunk in tqdm(enriched_chunks, desc="Enriqueciendo chunks"):
                parent_metadata = {
                    'title': chunk.metadata.get('title'),
                    'source': chunk.metadata.get('source'),
                    'content_type': chunk.metadata.get('content_type')
                }
                self.chunk_enricher.enrich_chunk(chunk, parent_metadata)

        # 5.5. Enriquecer con metadata para fact-checking
        if self.fact_metadata_extractor:
            self.logger.info("\t🔍 Extrayendo metadata para fact-checking (fechas, entidades, hechos)...")
            enriched_chunks = self.fact_metadata_extractor.enrich_documents(enriched_chunks)

        # 5.6. Entrenar y aplicar topic modeling con Gensim LDA
        if self.topic_extractor:
            self.logger.info("\t📊 Entrenando modelo LDA y detectando temas automáticamente...")
            # Entrenar modelo con todos los chunks
            self.topic_extractor.train(enriched_chunks)
            # Enriquecer chunks con temas detectados
            enriched_chunks = self.topic_extractor.enrich_documents(enriched_chunks)

        # 6. Generación HyDE
        if self.hyde_augmenter:
            self.logger.info("\tGenerando preguntas hipotéticas (HyDE)...")
            enriched_chunks = self.hyde_augmenter.augment_chunks(enriched_chunks)
            self.logger.info(f"\tChunks totales (con HyDE): {len(enriched_chunks)}")

        self.logger.info(
            f"✅ Pipeline completado: {len(enriched_chunks)} chunks finales"
        )

        return enriched_chunks

    def _add_location_metadata(
        self,
        chunks: List[Document]
    ) -> List[Document]:
        """
        Añade metadatos de ubicación a los fragmentos.

        Para archivos TXT: añade chunk_id y total_chunks_in_file
        Para archivos PDF: la información de página ya viene incluida

        Args:
            chunks: Lista de fragmentos

        Returns:
            Lista de fragmentos con metadatos enriquecidos
        """
        # Agrupar por archivo fuente
        docs_by_source: Dict[str, List[Document]] = {}

        for chunk in chunks:
            source = chunk.metadata.get('source', 'unknown')
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(chunk)

        # Procesar cada grupo
        enriched_chunks = []

        for source, doc_list in docs_by_source.items():
            # Determinar tipo de documento
            is_pdf = 'page' in doc_list[0].metadata

            if is_pdf:
                # Para PDFs, añadir información del total y section_number basado en la página
                for doc in doc_list:
                    doc.metadata['total_pages'] = max(
                        d.metadata.get('page', 0) for d in doc_list
                    ) + 1
                    # section_number = numero de página + 1 (para que empiece por 1)
                    doc.metadata['section_number'] = doc.metadata.get('page', 0) + 1
                    enriched_chunks.append(doc)
            else:
                # Para TXT, añadir chunk_id y section_number
                total_chunks = len(doc_list)
                for i, doc in enumerate(doc_list, 1):
                    doc.metadata['chunk_id'] = i
                    doc.metadata['section_number'] = i
                    doc.metadata['total_chunks_in_file'] = total_chunks
                    enriched_chunks.append(doc)

        return enriched_chunks

    def ingest(
        self,
        data_path: str = None,
        db_path: str = None,
        clear_existing: bool = False
    ) -> Chroma:
        """
        Ejecuta el proceso completo de ingesta.

        Este método orquesta todo el pipeline:
        1. Carga documentos
        2. Los fragmenta
        3. Genera embeddings
        4. Los almacena en ChromaDB

        Args:
            data_path: Directorio de datos (usa config si no se especifica)
            db_path: Ruta de la base de datos (usa config si no se especifica)
            clear_existing: Si True, elimina la BD existente antes de crear

        Returns:
            Instancia de ChromaDB con los documentos indexados

        Raises:
            Exception: Si falla algún paso del proceso
        """
        self.logger.info("Iniciando proceso de ingesta completo")

        # Rutas
        if data_path is None:
            data_path = self.config.get_path('data_raw')
        if db_path is None:
            db_path = self.config.get_path('vector_store')

        self.logger.info(f"\tDirectorio de datos: {data_path}")
        self.logger.info(f"\tBase de datos: {db_path}")

        # Limpiar BD existente si se solicita
        if clear_existing and os.path.exists(db_path):
            self.logger.warning(f"\tEliminando base de datos existente...")
            shutil.rmtree(db_path)

        # 1. Cargar documentos
        documents = self.load_documents(data_path)

        if not documents:
            raise ValueError(
                "❌ No se cargaron documentos. "
                "Verifica que el directorio contenga archivos .txt o .pdf"
            )

        # 2. Fragmentar
        chunks = self.chunk_documents(documents)

        # 3. Filtras metadatos complejos
        self.logger.info("Filtrando metadatos complejos")
        chunks = self._filter_complex_metadata(chunks)

        # 4. Crear base de datos vectoria
        self.logger.info("Creando base de datos vectorial...")

        try:
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=db_path
            )

            # Obtener estadísticas
            doc_count = vector_db._collection.count()

            self.logger.info("=" * 70)
            self.logger.info("✅ INGESTA COMPLETADA CON ÉXITO")
            self.logger.info(f"   Documentos en BD: {doc_count}")
            self.logger.info(f"   Ubicación: {db_path}")
            self.logger.info("=" * 70)

            return vector_db

        except Exception as e:
            self.logger.error(f"❌ Error creando base de datos: {e}")
            raise

    def get_stats(self, db_path: str = None) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la base de datos.

        Args:
            db_path: Ruta de la base de datos (usa config si no se especifica)

        Returns:
            Diccionario con estadísticas
        """
        if db_path is None:
            db_path = self.config.get_path('vector_store')

        if not os.path.exists(db_path):
            return {
                'exists': False,
                'path': db_path
            }

        try:
            vector_db = Chroma(
                persist_directory=db_path,
                embedding_function=self.embeddings
            )

            stats = {
                'exists': True,
                'path': db_path,
                'document_count': vector_db._collection.count(),
                'size_mb': self._get_directory_size(db_path)
            }

            return stats

        except Exception as e:
            return {
                'exists': True,
                'path': db_path,
                'error': str(e)
            }

    def _get_directory_size(self, path: str) -> float:
        """
        Calcula el tamaño de un directorio en MB.

        Args:
            path: Ruta del directorio

        Returns:
            Tamaño en megabytes
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)

        return round(total_size / (1024 * 1024), 2)

    def _filter_complex_metadata(self, chunks: List[Document]) -> List[Document]:
        """
        Filtra metadatos complejos que ChromaDB no puede almacenar.

        ChromaDB solo acepta: str, int, float, bool, None
        Convierte listas a strings separados por comas.

        Args:
            chunks: Lista de chunks con metadatos

        Returns:
            Lista de chunks con metadatos filtrados
        """
        for chunk in chunks:
            metadata = chunk.metadata.copy()

            # Convertir listas a strings
            for key, value in list(metadata.items()):
                if isinstance(value, list):
                    if value:  # Si la lista no está vacía
                        # Convertir a string separado por comas
                        metadata[key] = ', '.join(str(v) for v in value[:5])  # Limitar a 5 items
                    else:
                        # Eliminar listas vacías
                        del metadata[key]
                elif isinstance(value, dict):
                    # Eliminar diccionarios
                    del metadata[key]
                elif value is None:
                    # Mantener None
                    continue
                elif not isinstance(value, (str, int, float, bool)):
                    # Convertir otros tipos a string
                    metadata[key] = str(value)

            chunk.metadata = metadata

        return chunks

def main():
    """Función principal para ejecutar la ingesta."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingesta de documentos a base de datos vectorial"
    )
    parser.add_argument(
        '--data-path',
        type=str,
        help='Directorio con los documentos a ingestar'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        help='Ruta donde guardar la base de datos'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Eliminar base de datos existente antes de crear'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Mostrar solo estadísticas de la BD existente'
    )

    args = parser.parse_args()

    # Inicializar ingester
    ingester = DocumentIngester()

    # Si solo queremos stats
    if args.stats:
        stats = ingester.get_stats(args.db_path)
        print("📊 Estadísticas de la Base de Datos:")
        print("=" * 50)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("=" * 50)
        return

    # Ejecutar ingesta
    try:
        ingester.ingest(
            data_path=args.data_path,
            db_path=args.db_path,
            clear_existing=args.clear
        )

        # Mostrar estadísticas finales
        stats = ingester.get_stats(args.db_path)
        print("📊 Estadísticas Finales:")
        print("=" * 50)
        print(f"Documentos: {stats.get('document_count', 'N/A')}")
        print(f"Tamaño BD: {stats.get('size_mb', 'N/A')} MB")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Error durante la ingesta: {e}")
        raise


if __name__ == "__main__":
    main()

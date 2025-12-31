"""
Script de Pruebas para Pipeline Avanzado de Recuperación.

Valida cada componente del AdvancedRetriever:
- MetadataFilter
- HybridSearcher
- DiversitySelector
- Pipeline completo

Autor: Proyecto Final NLP - UC3M
Fecha: Diciembre 2025
"""

import os
import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from retriever import (
    AdvancedRetriever,
    RetrievalConfig,
    MetadataFilter,
    HybridSearcher,
    DiversitySelector
)
from utils.utils import ConfigManager, setup_logger
logger = setup_logger('TestAdvancedRetrieval', level='DEBUG')


def test_metadata_filter():
    """Prueba del filtrado por metadatos."""
    logger.info("=" * 80)
    logger.info("🧪 TEST 1: MetadataFilter")
    logger.info("=" * 80)

    config = RetrievalConfig(use_metadata_filter=True)
    metadata_filter = MetadataFilter(config)

    # Claim de prueba
    claim = "El Real Madrid se fundó en 1902 por Juan Padrós"

    # Documentos simulados con metadatos
    docs = [
        Document(
            page_content="El Real Madrid Club de Fútbol fue fundado el 6 de marzo de 1902...",
            metadata={
                'dates': '6 de marzo de 1902, 1902',
                'persons': 'Juan Padrós, Carlos Padrós, Julián Palacios',
                'organizations': 'Real Madrid Club de Fútbol',
                'content_type': 'historical'
            }
        ),
        Document(
            page_content="El Barcelona fue fundado en 1899 por Joan Gamper...",
            metadata={
                'dates': '1899',
                'persons': 'Joan Gamper',
                'organizations': 'FC Barcelona',
                'content_type': 'historical'
            }
        ),
        Document(
            page_content="El equipo ganó la Champions League en 2022...",
            metadata={
                'dates': '2022',
                'keywords': 'champions, league, equipo, ganó',
                'content_type': 'statistical'
            }
        )
    ]

    # Extraer entidades del claim
    entities = metadata_filter.extract_entities_from_claim(claim)
    logger.debug(f"📝 Claim: {claim}")
    logger.debug(f"🔍 Entidades extraídas: {entities}")

    # Puntuar documentos
    scored_docs = metadata_filter.filter_and_score(claim, docs)

    logger.debug(f"📊 Documentos puntuados por metadatos:")
    for i, (doc, score) in enumerate(scored_docs, 1):
        logger.debug(f"{i}. Score: {score:.3f}")
        logger.debug(f"   Contenido: {doc.page_content[:80]}...")
        if 'dates' in doc.metadata:
            logger.debug(f"   Fechas: {doc.metadata['dates']}")
        if 'persons' in doc.metadata:
            logger.debug(f"   Personas: {doc.metadata['persons']}")

    # Verificar que el doc. más relevante tiene score alto
    best_score = max(s for _, s in scored_docs)
    logger.info(f"✅ Mejor score: {best_score:.3f}")
    assert best_score > 0.3, "El documento más relevante debería tener score > 0.3"
    logger.info("✅ Test de MetadataFilter PASADO")


def test_hybrid_searcher():
    """Prueba de búsqueda híbrida."""
    logger.info("=" * 80)
    logger.info("🧪 TEST 2: HybridSearcher")
    logger.info("=" * 80)

    config = RetrievalConfig(use_hybrid_search=True, keyword_weight=0.3)
    hybrid_searcher = HybridSearcher(config)

    claim = "¿Cuándo ganó el Real Madrid su primera Copa de Europa?"

    docs = [
        Document(
            page_content="El Real Madrid ganó su primera Copa de Europa en 1956 frente al Stade de Reims...",
            metadata={'dates': '1956', 'keywords': 'copa, europa, real, madrid'}
        ),
        Document(
            page_content="La Champions League es el torneo más prestigioso de clubes en Europa...",
            metadata={'keywords': 'champions, league, europa, torneo'}
        ),
        Document(
            page_content="El Barcelona jugó contra el Manchester United en una final emocionante...",
            metadata={'keywords': 'barcelona, manchester, final'}
        )
    ]

    logger.debug(f"📝 Claim: {claim}")

    # Calcular keyword scores
    hybrid_scores = hybrid_searcher.hybrid_score(claim, docs)

    logger.debug(f"📊 Scores híbridos (semántico + keywords):")
    for i, (doc, score) in enumerate(hybrid_scores, 1):
        logger.debug(f"{i}. Score: {score:.3f}")
        logger.debug(f"   Contenido: {doc.page_content[:80]}...")

    # El primer documento debería tener el mejor score
    best_doc, best_score = hybrid_scores[0]
    logger.debug(f"✅ Mejor documento: score {best_score:.3f}")
    logger.debug(f"   {best_doc.page_content[:100]}...")

    assert best_score > 0, "Debería haber algún match"
    logger.info("✅ Test de HybridSearcher PASADO")


def test_diversity_selector():
    """Prueba de diversificación de fuentes."""
    logger.info("=" * 80)
    logger.info("🧪 TEST 3: DiversitySelector")
    logger.info("=" * 80)

    config = RetrievalConfig(
        use_diversity=True,
        max_chunks_per_source=2,
        diversity_penalty=0.15
    )
    diversity_selector = DiversitySelector(config)

    # Documentos de la misma fuente
    docs = [
        (Document(page_content="Chunk 1 de documento A", metadata={'source': 'doc_A.txt'}), 0.9),
        (Document(page_content="Chunk 2 de documento A", metadata={'source': 'doc_A.txt'}), 0.85),
        (Document(page_content="Chunk 3 de documento A", metadata={'source': 'doc_A.txt'}), 0.8),
        (Document(page_content="Chunk 4 de documento A", metadata={'source': 'doc_A.txt'}), 0.75),
        (Document(page_content="Chunk 1 de documento B", metadata={'source': 'doc_B.txt'}), 0.7),
        (Document(page_content="Chunk 1 de documento C", metadata={'source': 'doc_C.txt'}), 0.65),
    ]

    logger.debug(f"📚 Documentos originales (6 docs, 4 del mismo source):")
    for doc, score in docs:
        source = doc.metadata['source']
        logger.debug(f"   {source}: score {score:.2f}")

    # Aplicar diversificación
    diversified = diversity_selector.diversify(docs)

    logger.debug(f"🎯 Después de diversificación:")
    for doc, score in diversified[:6]:
        source = doc.metadata['source']
        logger.debug(f"   {source}: score ajustado {score:.2f}")

    # Verificar que hay más diversidad en el top
    top_3_sources = [doc.metadata['source'] for doc, _ in diversified[:3]]
    unique_sources = len(set(top_3_sources))

    logger.info(f"✅ Fuentes únicas en top-3: {unique_sources}/3")
    assert unique_sources >= 2, "Debería haber al menos 2 fuentes diferentes en top-3"
    logger.info("✅ Test de DiversitySelector PASADO")


def test_advanced_retriever():
    """Prueba del pipeline completo con base de datos real."""
    logger.info("=" * 80)
    logger.info("🧪 TEST 4: AdvancedRetriever (Pipeline Completo)")
    logger.info("=" * 80)

    # Cargar configuración
    config_manager = ConfigManager("config.yaml")

    # Inicializar embeddings
    model_name = config_manager.get('models.embeddings.name')
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Conectar a vector DB
    db_path = config_manager.get_path('vector_store')

    if not os.path.exists(db_path):
        logger.warning(f"⚠️  Base de datos no encontrada en: {db_path}")
        logger.warning("   Ejecuta primero: python ingest_data.py --clear")
        return

    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    # Cargar reranker
    try:
        reranker_name = config_manager.get('models.reranker.name')
        reranker = CrossEncoder(reranker_name)
        logger.info(f"✅ Reranker cargado: {reranker_name}")
    except Exception as e:
        logger.error(f"⚠️  No se pudo cargar reranker: {e}")
        reranker = None

    # Configurar retriever
    retrieval_config = RetrievalConfig(
        k_initial=30,
        use_metadata_filter=True,
        metadata_boost=0.2,
        use_hybrid_search=True,
        keyword_weight=0.3,
        use_reranker=reranker is not None,
        rerank_top_k=15,
        use_diversity=True,
        max_chunks_per_source=3,
        min_relevance_score=0.2,
        final_top_k=5
    )

    # Crear retriever
    retriever = AdvancedRetriever(
        vector_db=vector_db,
        reranker=reranker,
        config=retrieval_config
    )

    # Queries de prueba
    test_queries = [
        "¿Cuándo se fundó el Real Madrid?",
        "¿Quiénes fueron los fundadores del Real Madrid?",
        "¿Cuántas Copas de Europa ha ganado el Real Madrid?",
        "¿En qué estadio juega el Real Madrid?",
    ]

    logger.info(f"🔍 Ejecutando {len(test_queries)} queries de prueba...")

    for i, query in enumerate(test_queries, 1):
        logger.debug(f"{'─' * 80}")
        logger.debug(f"Query {i}: {query}")
        logger.debug('─' * 80)

        # Recuperar con scores
        docs, scores = retriever.retrieve(query, return_scores=True)

        logger.debug(f"📊 Recuperados {len(docs)} documentos:")

        for j, (doc, score) in enumerate(zip(docs, scores), 1):
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            chunk_id = doc.metadata.get('chunk_id', '?')
            content_type = doc.metadata.get('content_type', 'N/A')

            logger.debug(f"     {j}. [Score: {score:.3f}] {source} (chunk {chunk_id})")
            logger.debug(f"     Tipo: {content_type}")
            logger.debug(f"     Contenido: {doc.page_content[:150]}...")

            # Mostrar metadatos relevantes
            if 'dates' in doc.metadata:
                logger.debug(f"     Fechas: {doc.metadata['dates']}")
            if 'persons' in doc.metadata:
                persons = str(doc.metadata['persons'])
                if len(persons) > 80:
                    persons = persons[:80] + "..."
                logger.debug(f"     Personas: {persons}")

        # Verificar diversidad
        sources = [doc.metadata.get('source', 'unknown') for doc in docs]
        unique_sources = len(set(sources))
        logger.debug(f"  📚 Fuentes únicas: {unique_sources}/{len(docs)}")

    logger.info("=" * 80)
    logger.info("✅ Test de AdvancedRetriever COMPLETADO")
    logger.info("=" * 80)


def main():
    """Ejecuta todos los tests."""
    logger.info("=" * 80)
    logger.info("🧪 SUITE DE TESTS: ADVANCED RETRIEVAL")
    logger.info("=" * 80)

    try:
        # Tests unitarios (sin BD)
        test_metadata_filter()
        test_hybrid_searcher()
        test_diversity_selector()

        # Test de integración (con BD)
        test_advanced_retriever()

        logger.info("=" * 80)
        logger.info("✅✅✅ TODOS LOS TESTS PASADOS ✅✅✅")
        logger.info("=" * 80)

    except AssertionError as e:
        logger.error(f"❌ TEST FALLIDO: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Script de Prueba Rápida del Sistema Mejorado.

Verifica que todos los módulos nuevos funcionen correctamente
sin necesidad de ingestar datos completos.

Autor: Proyecto Final NLP - UC3M
Fecha: Diciembre 2025
"""

from langchain_core.documents import Document
from utils.utils import setup_logger
from preprocessor import DocumentPreprocessor
from chunker import SemanticChunker
from extractor import MetadataExtractor, ChunkMetadataEnricher
from hyde import HyDEGenerator, SimpleHyDEGenerator, HyDEAugmenter

logger = setup_logger(name="TEST_RAG", level="DEBUG")


def test_preprocessor():
    """Prueba el preprocesador de documentos."""
    logger.info("=" * 70)
    logger.info("🧹 PRUEBA 1: Document Preprocessor")
    logger.info("=" * 70)

    try:

        # Texto de prueba con problemas
        test_text = """
        El Real Madrid Club de Fútbol fue fundado en 1902.


        Visita https://www.realmadrid.com para más info.
        Contacto: info@realmadrid.com

        Referencias
        ========
        1. Wikipedia
        2. Sitio oficial
        """

        doc = Document(page_content=test_text, metadata={'source': 'test.txt'})

        # Probar preprocesador estándar
        preprocessor = DocumentPreprocessor()
        cleaned_doc = preprocessor.preprocess_document(doc)

        logger.debug("📄 Texto original:")
        logger.debug(test_text[:100] + "...")

        logger.debug("✨ Texto limpio:")
        logger.debug(cleaned_doc.page_content[:200] + "...")

        logger.info("📊 Metadatos extraídos:")
        logger.info(f"  - Tiene título: {cleaned_doc.metadata.get('has_title')}")
        logger.info(f"  - Número de secciones: {cleaned_doc.metadata.get('num_sections')}")
        logger.info(f"  - Número de párrafos: {cleaned_doc.metadata.get('num_paragraphs')}")

        logger.info("✅ Preprocesador funcionando correctamente")
        return True

    except Exception as e:
        logger.error(f"❌ Error en preprocesador: {e}")
        return False


def test_semantic_chunker():
    """Prueba el chunker semántico."""
    logger.info("=" * 70)
    logger.info("✂️ PRUEBA 2: Semantic Chunker")
    logger.info("=" * 70)

    try:

        test_text = """
        El Real Madrid Club de Fútbol es un club deportivo español con sede en Madrid.
        Fue fundado el 6 de marzo de 1902 como Madrid Football Club.
        Es uno de los clubes más exitosos del mundo.
        Ha ganado 14 títulos de la Liga de Campeones de la UEFA.
        Su estadio es el Santiago Bernabéu, inaugurado en 1947.
        El club cuenta con más de 100 millones de seguidores en todo el mundo.
        """

        doc = Document(page_content=test_text, metadata={'source': 'test.txt'})

        # Probar chunker semántico
        chunker = SemanticChunker(chunk_size=150, chunk_overlap=50)
        chunks = chunker.split_documents([doc])

        logger.debug(f"📊 Chunks generados: {len(chunks)}")

        for i, chunk in enumerate(chunks, 1):
            logger.debug(f"--- Chunk {i} ({len(chunk.page_content)} chars) ---")
            logger.debug(chunk.page_content)

        logger.info("✅ Chunker semántico funcionando correctamente")
        return True

    except Exception as e:
        logger.error(f"❌ Error en chunker: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_extractor():
    """Prueba el extractor de metadatos."""
    logger.info("=" * 70)
    logger.info("🏷️ PRUEBA 3: Metadata Extractor")
    logger.info("=" * 70)

    try:

        test_text = """
        Real Madrid Club de Fútbol

        El Real Madrid fue fundado el 6 de marzo de 1902 por Juan Padrós.
        El club está ubicado en Madrid, España.
        Florentino Pérez es el presidente actual.
        El equipo ha ganado 35 títulos de La Liga.
        """

        doc = Document(page_content=test_text, metadata={'source': 'test.txt'})

        extractor = MetadataExtractor(
            extract_dates=True,
            extract_entities=True,
            classify_content=True
        )

        enriched_doc = extractor.extract_metadata(doc)

        logger.info("📊 Metadatos extraídos:")
        logger.info(f"  - Título: {enriched_doc.metadata.get('title')}")
        logger.info(f"  - Fechas: {enriched_doc.metadata.get('dates', [])}")
        logger.info(f"  - Personas: {enriched_doc.metadata.get('persons', [])}")
        logger.info(f"  - Organizaciones: {enriched_doc.metadata.get('organizations', [])}")
        logger.info(f"  - Lugares: {enriched_doc.metadata.get('locations', [])}")
        logger.info(f"  - Tipo de contenido: {enriched_doc.metadata.get('content_type')}")
        logger.info(f"  - Densidad de info: {enriched_doc.metadata.get('info_density')}")
        logger.info(f"  - Keywords: {enriched_doc.metadata.get('keywords', [])[:5]}")

        logger.info("✅ Extractor de metadatos funcionando correctamente")
        return True

    except Exception as e:
        logger.error(f"❌ Error en extractor: {e}")
        logger.error("   (Nota: Si spaCy no está instalado, algunas funciones no estarán disponibles)")
        import traceback
        traceback.print_exc()
        return False


def test_hyde_generator():
    """Prueba el generador HyDE."""
    logger.info("=" * 70)
    logger.info("💡 PRUEBA 4: HyDE Generator")
    logger.info("=" * 70)

    try:

        test_text = """
        El Real Madrid Club de Fútbol fue fundado el 6 de marzo de 1902.
        Es uno de los clubes más exitosos del mundo, habiendo ganado 14 títulos
        de la Liga de Campeones de la UEFA. Su estadio es el Santiago Bernabéu,
        ubicado en Madrid, España.
        """

        doc = Document(
            page_content=test_text,
            metadata={
                'source': 'test.txt',
                'dates': ['6 de marzo de 1902'],
                'organizations': ['Real Madrid', 'UEFA'],
                'locations': ['Madrid', 'España']
            }
        )

        # Intentar generador completo, fallback a simple
        try:
            generator = HyDEGenerator(num_questions=3)
            logger.debug("📝 Usando HyDEGenerator completo")
        except Exception as e:
            generator = SimpleHyDEGenerator(num_questions=3)
            logger.warning("📝 Usando SimpleHyDEGenerator (sin spaCy)")
            logger.warning(f"Error SimpleHyDEGenerator: {e}")

        enriched_doc = generator.generate_questions(doc)
        questions = enriched_doc.metadata.get('hypothetical_questions', [])

        logger.debug(f"❓ Preguntas generadas ({len(questions)}):")
        for i, question in enumerate(questions, 1):
            logger.debug(f"  {i}. {question}")

        logger.info("✅ Generador HyDE funcionando correctamente")
        return True

    except Exception as e:
        logger.error(f"❌ Error en HyDE: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Prueba integración completa del pipeline."""
    logger.info("=" * 70)
    logger.info("🔗 PRUEBA 5: Pipeline Integrado")
    logger.info("=" * 70)

    try:
        # Documento de prueba
        test_text = """
        Real Madrid Club de Fútbol

        El Real Madrid fue fundado el 6 de marzo de 1902 como Madrid Football Club.
        Es uno de los clubes más exitosos del mundo. Ha ganado 14 títulos de la
        Liga de Campeones de la UEFA, más que cualquier otro club. Su estadio es
        el Santiago Bernabéu, inaugurado en 1947 y ubicado en Madrid, España.
        El presidente actual es Florentino Pérez.

        Referencias
        ==========
        Wikipedia, Sitio oficial
        """

        doc = Document(page_content=test_text, metadata={'source': 'test.txt'})

        logger.info("1️⃣ Preprocesamiento...")
        preprocessor = DocumentPreprocessor()
        doc = preprocessor.preprocess_document(doc)

        logger.info("2️⃣ Extracción de metadatos del documento...")
        metadata_extractor = MetadataExtractor()
        doc = metadata_extractor.extract_metadata(doc)

        logger.info("3️⃣ Chunking semántico...")
        chunker = SemanticChunker(chunk_size=150, chunk_overlap=50)
        chunks = chunker.split_documents([doc])
        logger.info(f"   Chunks creados: {len(chunks)}")

        logger.info("4️⃣ Enriquecimiento de chunks...")
        enricher = ChunkMetadataEnricher()
        for chunk in chunks:
            parent_meta = {'title': doc.metadata.get('title'), 'source': 'test.txt'}
            enricher.enrich_chunk(chunk, parent_meta)

        logger.info("5️⃣ Generación HyDE...")
        try:
            generator = HyDEGenerator(num_questions=2)
        except Exception as e:
            generator = SimpleHyDEGenerator(num_questions=2)
            logger.warning(f"Error SimpleHyDEGenerator {e}")

        augmenter = HyDEAugmenter(generator=generator, create_question_docs=False)
        final_chunks = augmenter.augment_chunks(chunks)

        logger.info(f"✨ Pipeline completado:")
        logger.info(f"   - Chunks finales: {len(final_chunks)}")
        logger.info(f"   - Primer chunk:")
        logger.info(f"     * Texto: {final_chunks[0].page_content[:80]}...")
        logger.info(f"     * Metadatos: {len(final_chunks[0].metadata)} campos")
        logger.info(f"     * Preguntas HyDE: {len(final_chunks[0].metadata.get('hypothetical_questions', []))}")

        logger.info("✅ Pipeline integrado funcionando correctamente")
        return True

    except Exception as e:
        logger.error(f"❌ Error en pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ejecuta todas las pruebas."""
    logger.info("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           PRUEBA DEL SISTEMA RAG MEJORADO                    ║
    ║              Testing de Módulos Implementados                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    tests = [
        ("Preprocesador", test_preprocessor),
        ("Chunker Semántico", test_semantic_chunker),
        ("Extractor de Metadatos", test_metadata_extractor),
        ("Generador HyDE", test_hyde_generator),
        ("Pipeline Integrado", test_integration),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Error crítico en {test_name}: {e}")
            results.append((test_name, False))

    # Resumen final
    logger.info("=" * 70)
    logger.info("📊 RESUMEN DE PRUEBAS")
    logger.info("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅" if result else "❌"
        logger.debug(f"{status} {test_name}")

    logger.info(f"🎯 Resultado: {passed}/{total} pruebas exitosas")

    if passed == total:
        logger.info("🎉 ¡Todos los módulos funcionan correctamente!")
        logger.info("   Puedes proceder a ingestar datos con el sistema mejorado.")
        logger.info("💡 Siguiente paso:")
        logger.info("   python ingest_data.py --clear")
    elif passed >= 3:
        logger.warning("⚠️  Algunos módulos tienen problemas menores.")
        logger.warning("   El sistema funcionará en modo degradado.")
        logger.warning("   Instala spaCy para funcionalidad completa:")
        logger.warning("   python -m spacy download es_core_news_sm")
    else:
        logger.error("❌ Hay problemas críticos en el sistema.")
        logger.error("   Revisa los errores anteriores y las dependencias.")


if __name__ == "__main__":
    main()

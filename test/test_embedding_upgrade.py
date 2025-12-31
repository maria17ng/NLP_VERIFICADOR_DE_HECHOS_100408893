"""
Test para validar la mejora de embeddings y reranker.

Compara:
- ANTES: paraphrase-multilingual-MiniLM-L12-v2 + mmarco-mMiniLMv2
- DESPUÉS: Alibaba-NLP/gte-multilingual-base + BAAI/bge-reranker-v2-m3

Métricas:
- Hit Rate: ¿Recupera el documento correcto en top-K?
- MRR (Mean Reciprocal Rank): ¿En qué posición está el doc correcto?
- Retrieval time: ¿Qué tan rápido es?
"""

import sys
from pathlib import Path
import time
from typing import List, Tuple
import yaml

# Agregar ruta del proyecto
sys.path.insert(0, str(Path(__file__).parent))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


def load_config():
    """Cargar configuración."""
    with open("settings/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_metrics(retrieved_docs: List[Document], ground_truth_section: int) -> dict:
    """
    Calcula métricas de recuperación.

    Args:
        retrieved_docs: Documentos recuperados (ordenados por relevancia)
        ground_truth_section: Sección correcta esperada

    Returns:
        Dict con métricas
    """
    # Extraer secciones de los docs recuperados
    retrieved_sections = []
    for doc in retrieved_docs:
        section = doc.metadata.get('section_number', -1)
        retrieved_sections.append(section)

    # Hit Rate: ¿Está el doc correcto en el top-K?
    hit = ground_truth_section in retrieved_sections

    # MRR: Posición del primer doc correcto
    try:
        first_correct_pos = retrieved_sections.index(ground_truth_section) + 1  # 1-indexed
        mrr = 1.0 / first_correct_pos
    except ValueError:
        mrr = 0.0  # No encontrado
        first_correct_pos = None

    return {
        'hit': hit,
        'mrr': mrr,
        'position': first_correct_pos,
        'retrieved_sections': retrieved_sections[:5]  # Mostrar top-5
    }


def test_retrieval_with_models(
        embedding_model: str,
        reranker_model: str,
        test_queries: List[Tuple[str, int]]
) -> dict:
    """
    Prueba recuperación con modelos específicos.

    Args:
        embedding_model: Nombre del modelo de embeddings
        reranker_model: Nombre del modelo de reranker
        test_queries: Lista de (query, ground_truth_section)

    Returns:
        Dict con resultados agregados
    """
    print(f"\n{'=' * 80}")
    print(f"TESTING: {embedding_model}")
    print(f"RERANKER: {reranker_model}")
    print(f"{'=' * 80}\n")

    # Inicializar embeddings
    print("🔧 Cargando modelo de embeddings...")
    start = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"   ✅ Embeddings cargados en {time.time() - start:.2f}s")

    # Cargar vector store
    print("🔧 Conectando a ChromaDB...")
    vector_db = Chroma(
        persist_directory="data/vector_store",
        embedding_function=embeddings
    )
    print(f"   ✅ {vector_db._collection.count()} documentos en BD")

    # Inicializar reranker
    print(f"🔧 Cargando reranker...")
    start = time.time()
    try:
        reranker = CrossEncoder(reranker_model, max_length=512)
        print(f"   ✅ Reranker cargado en {time.time() - start:.2f}s")
    except Exception as e:
        print(f"   ⚠️ Error cargando reranker: {e}")
        print(f"   ⚠️ Continuando sin reranking...")
        reranker = None

    # Ejecutar queries
    results = []
    total_retrieval_time = 0

    for i, (query, ground_truth_section) in enumerate(test_queries, 1):
        print(f"\n{'─' * 80}")
        print(f"TEST {i}/{len(test_queries)}: {query}")
        print(f"Ground Truth: Sección {ground_truth_section}")
        print(f"{'─' * 80}")

        # Búsqueda vectorial
        start = time.time()
        docs = vector_db.similarity_search(query, k=10)
        retrieval_time = time.time() - start
        total_retrieval_time += retrieval_time

        print(f"⏱️  Búsqueda vectorial: {retrieval_time:.3f}s")

        # DEBUG: Mostrar primeros docs recuperados
        if docs:
            print(f"\n   📄 Primeros 3 docs recuperados:")
            for i, doc in enumerate(docs[:3], 1):
                preview = doc.page_content[:80].replace('\n', ' ')
                section = doc.metadata.get('section_number', 'N/A')
                source = doc.metadata.get('source', 'N/A')
                print(f"      {i}. [Sec: {section}] {preview}...")
        else:
            print(f"   ⚠️ No se recuperaron documentos")

        # Reranking (si disponible)
        if reranker:
            start = time.time()
            # Crear pares (query, doc_text)
            pairs = [[query, doc.page_content] for doc in docs]
            scores = reranker.predict(pairs)

            # Ordenar por score
            reranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            docs = [docs[i] for i in reranked_indices[:5]]  # Top-5 después de reranking

            rerank_time = time.time() - start
            total_retrieval_time += rerank_time
            print(f"⏱️  Reranking: {rerank_time:.3f}s")

        # Calcular métricas
        metrics = calculate_metrics(docs, ground_truth_section)
        results.append(metrics)

        # Mostrar resultados
        print(f"\n📊 Resultados:")
        print(f"   Hit: {'✅' if metrics['hit'] else '❌'}")
        print(f"   MRR: {metrics['mrr']:.3f}")
        if metrics['position']:
            print(f"   Posición correcta: {metrics['position']}")
        else:
            print(f"   ❌ Documento correcto NO encontrado en top-10")
        print(f"   Secciones recuperadas (top-5): {metrics['retrieved_sections']}")

    # Métricas agregadas
    avg_hit_rate = sum(r['hit'] for r in results) / len(results)
    avg_mrr = sum(r['mrr'] for r in results) / len(results)
    avg_retrieval_time = total_retrieval_time / len(test_queries)

    print(f"\n{'=' * 80}")
    print(f"📈 MÉTRICAS AGREGADAS")
    print(f"{'=' * 80}")
    print(f"Hit Rate@5:     {avg_hit_rate:.2%} ({sum(r['hit'] for r in results)}/{len(results)} queries)")
    print(f"Mean MRR:       {avg_mrr:.3f}")
    print(f"Avg Time/Query: {avg_retrieval_time:.3f}s")
    print(f"{'=' * 80}\n")

    return {
        'embedding_model': embedding_model,
        'reranker_model': reranker_model,
        'hit_rate': avg_hit_rate,
        'mrr': avg_mrr,
        'avg_time': avg_retrieval_time,
        'individual_results': results
    }


def main():
    """Ejecutar tests comparativos."""
    print("=" * 80)
    print("TEST: COMPARACIÓN DE EMBEDDINGS Y RERANKERS")
    print("=" * 80)
    print("\nObjetivo: Validar mejora con nuevos modelos")
    print("Dataset: Real Madrid corpus (732 chunks)")
    print("\n" + "=" * 80)

    # Test queries con ground truth (sección donde está la respuesta correcta)
    # NOTA: Los section_number son ajustados según el chunking real del documento
    test_queries = [
        # Query sobre fundación (Sección 1 tiene "registrado 6 de marzo de 1902")
        ("El Real Madrid fue fundado en 1902", 1),
        ("El Real Madrid fue fundado en 1903", 1),  # ¡Fecha INCORRECTA! Debería encontrar 1902
        ("El Real Madrid fue fundado en 1900", 1),  # Fecha de orígenes

        # Query sobre estadio (Sección 105 tiene info del estadio inaugurado en 1947)
        ("El estadio Santiago Bernabéu fue inaugurado en 1947", 105),

        # Query sobre Champions League (Sección 90 tiene info de palmarés reciente)
        ("El Real Madrid ha ganado 14 Champions League", 90),

        # Query sobre jugadores históricos (Sección 153-154 tiene info de Di Stéfano)
        ("Alfredo Di Stéfano jugó en el Real Madrid", 153),
    ]

    # Cargar configuración para ver modelos actuales
    config = load_config()

    print("\n📋 CONFIGURACIÓN ACTUAL:")
    print(f"   Embedding: {config['models']['embeddings']['name']}")
    print(f"   Reranker:  {config['models']['reranker']['name']}")
    print(f"\n🧪 Ejecutando {len(test_queries)} test queries...\n")

    # Test con modelos actuales (de config.yaml)
    results_new = test_retrieval_with_models(
        embedding_model=config['models']['embeddings']['name'],
        reranker_model=config['models']['reranker']['name'],
        test_queries=test_queries
    )

    # Si quieres comparar con modelos antiguos, descomenta:
    # print("\n\n" + "="*80)
    # print("COMPARACIÓN CON MODELOS ANTERIORES")
    # print("="*80)
    #
    # results_old = test_retrieval_with_models(
    #     embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    #     reranker_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    #     test_queries=test_queries
    # )
    #
    # # Comparación
    # print("\n" + "="*80)
    # print("📊 COMPARACIÓN DE RESULTADOS")
    # print("="*80)
    # print(f"\nHit Rate:")
    # print(f"   ANTES: {results_old['hit_rate']:.2%}")
    # print(f"   AHORA: {results_new['hit_rate']:.2%}")
    # print(f"   MEJORA: {(results_new['hit_rate'] - results_old['hit_rate'])*100:+.1f} puntos porcentuales")
    #
    # print(f"\nMean Reciprocal Rank (MRR):")
    # print(f"   ANTES: {results_old['mrr']:.3f}")
    # print(f"   AHORA: {results_new['mrr']:.3f}")
    # print(f"   MEJORA: {(results_new['mrr'] - results_old['mrr']):+.3f}")
    #
    # print(f"\nTiempo promedio:")
    # print(f"   ANTES: {results_old['avg_time']:.3f}s")
    # print(f"   AHORA: {results_new['avg_time']:.3f}s")
    #
    # # Análisis por query
    # print("\n" + "="*80)
    # print("📋 ANÁLISIS DETALLADO POR QUERY")
    # print("="*80)
    # for i, (query, gt) in enumerate(test_queries):
    #     old_hit = results_old['individual_results'][i]['hit']
    #     new_hit = results_new['individual_results'][i]['hit']
    #
    #     status = "✅ MEJORÓ" if (not old_hit and new_hit) else "🔄 IGUAL" if old_hit == new_hit else "⚠️ EMPEORÓ"
    #     print(f"\n{i+1}. {query[:60]}...")
    #     print(f"   Ground Truth: Sec. {gt}")
    #     print(f"   ANTES: {'✅' if old_hit else '❌'} (pos: {results_old['individual_results'][i]['position']})")
    #     print(f"   AHORA: {'✅' if new_hit else '❌'} (pos: {results_new['individual_results'][i]['position']})")
    #     print(f"   {status}")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETADO")
    print("=" * 80)
    print("\n💡 Próximo paso: Si los resultados son buenos, ejecutar test_mejoras.py")
    print("   para validar la mejora en el fact-checking completo")


if __name__ == "__main__":
    main()

"""
Script para validar que los nuevos modelos se pueden cargar correctamente.
Ejecutar ANTES de re-ingestar datos.
"""

import sys
from pathlib import Path
import time
import yaml

sys.path.insert(0, str(Path(__file__).parent))


def load_config():
    """Cargar configuración."""
    with open("settings/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_embedding_model(model_name: str) -> bool:
    """
    Prueba cargar el modelo de embeddings.

    Returns:
        True si se carga correctamente
    """
    print(f"\n{'=' * 80}")
    print(f"TESTING EMBEDDING MODEL: {model_name}")
    print(f"{'=' * 80}\n")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        print("🔧 Descargando/cargando modelo...")
        start = time.time()

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )

        load_time = time.time() - start
        print(f"   ✅ Modelo cargado en {load_time:.2f}s")

        # Test embedding
        print("\n🧪 Probando embedding de texto...")
        test_text = "El Real Madrid fue fundado en 1902"
        start = time.time()
        embedding = embeddings.embed_query(test_text)
        embed_time = time.time() - start

        print(f"   ✅ Embedding generado en {embed_time:.3f}s")
        print(f"   Dimensión del vector: {len(embedding)}")
        print(f"   Primeros 5 valores: {embedding[:5]}")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 SOLUCIÓN:")
        print("   Instalar dependencia faltante:")
        print("   pip install FlagEmbedding")
        return False


def test_reranker_model(model_name: str) -> bool:
    """
    Prueba cargar el modelo de reranker.

    Returns:
        True si se carga correctamente
    """
    print(f"\n{'=' * 80}")
    print(f"TESTING RERANKER MODEL: {model_name}")
    print(f"{'=' * 80}\n")

    try:
        from sentence_transformers import CrossEncoder

        print("🔧 Descargando/cargando modelo...")
        start = time.time()

        reranker = CrossEncoder(model_name, max_length=512)

        load_time = time.time() - start
        print(f"   ✅ Modelo cargado en {load_time:.2f}s")

        # Test reranking
        print("\n🧪 Probando reranking...")
        query = "Real Madrid fundación 1902"
        docs = [
            "El Real Madrid fue registrado oficialmente en 1902",
            "El Barcelona fue fundado en 1899",
            "El estadio tiene capacidad para 80,000 personas"
        ]

        start = time.time()
        pairs = [[query, doc] for doc in docs]
        scores = reranker.predict(pairs)
        rerank_time = time.time() - start

        print(f"   ✅ Reranking completado en {rerank_time:.3f}s")
        print(f"\n   📊 Scores:")
        for i, (doc, score) in enumerate(zip(docs, scores), 1):
            print(f"      {i}. [{score:+.3f}] {doc[:60]}...")

        # Verificar que el doc correcto tiene mejor score
        best_idx = scores.argmax()
        if best_idx == 0:
            print(f"\n   ✅ Correcto: Doc más relevante tiene mejor score")
        else:
            print(f"\n   ⚠️ Ranking inesperado (puede ser normal con este test simple)")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 SOLUCIÓN:")
        print("   El modelo puede no estar disponible o requiere dependencias adicionales")
        print("   Intentar modelo alternativo: BAAI/bge-reranker-large")
        return False


def main():
    """Validar modelos configurados."""
    print("=" * 80)
    print("VALIDACIÓN DE MODELOS ACTUALIZADOS")
    print("=" * 80)

    # Cargar config
    config = load_config()

    embedding_model = config['models']['embeddings']['name']
    reranker_model = config['models']['reranker']['name']

    print(f"\n📋 Modelos a validar:")
    print(f"   Embedding: {embedding_model}")
    print(f"   Reranker:  {reranker_model}")

    # Test embedding
    embedding_ok = test_embedding_model(embedding_model)

    # Test reranker
    reranker_ok = test_reranker_model(reranker_model)

    # Resumen
    print(f"\n{'=' * 80}")
    print("RESUMEN DE VALIDACIÓN")
    print(f"{'=' * 80}\n")
    print(f"   Embedding: {'✅ OK' if embedding_ok else '❌ FALLO'}")
    print(f"   Reranker:  {'✅ OK' if reranker_ok else '❌ FALLO'}")

    if embedding_ok and reranker_ok:
        print(f"\n✅ TODOS LOS MODELOS FUNCIONAN CORRECTAMENTE")
        print(f"\n📋 Próximos pasos:")
        print(f"   1. Re-ingestar datos con nuevos embeddings:")
        print(f"      python ingest/ingest_data.py --clear")
        print(f"   2. Ejecutar test de validación:")
        print(f"      python test_embedding_upgrade.py")
        print(f"   3. Ejecutar test completo de fact-checking:")
        print(f"      python test_mejoras.py")
    else:
        print(f"\n❌ ALGUNOS MODELOS FALLARON")
        print(f"\n💡 Opciones:")
        print(f"   A) Instalar dependencias faltantes (ver errores arriba)")
        print(f"   B) Usar modelos alternativos en config.yaml:")
        print(f"      - Embedding: intfloat/multilingual-e5-large")
        print(f"      - Reranker: BAAI/bge-reranker-large")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()



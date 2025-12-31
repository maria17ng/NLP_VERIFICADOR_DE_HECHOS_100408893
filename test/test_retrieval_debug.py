"""
Script para debuggear el retrieval y ver qué documentos se recuperan.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from retriever import AdvancedRetriever, RetrievalConfig
from utils.utils import setup_logger

# Setup logger
logger = setup_logger('TestRetrieval', level='DEBUG')

# Cargar embeddings
print("Cargando embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Conectar a BD vectorial
print("Conectando a base de datos vectorial...")
vector_db = Chroma(
    persist_directory="data/vector_store",
    embedding_function=embeddings
)

# Cargar reranker
print("Cargando reranker...")
reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

# Configurar retriever
print("Configurando retriever avanzado...")
config = RetrievalConfig(
    k_initial=80,
    use_metadata_filter=True,
    use_hybrid_search=True,
    use_reranker=True,
    rerank_top_k=40,
    min_relevance_score=0.25,
    min_rerank_score=-5.0,
    keyword_weight=0.4
)

retriever = AdvancedRetriever(vector_db, reranker, config)

# Claims de prueba
test_claims = [
    "El Real Madrid fue fundado en 1902",
    "El Real Madrid fue fundado en 1903",
    "El Real Madrid fue fundado en 1950",
]

print("\n" + "=" * 80)
print("PRUEBAS DE RETRIEVAL")
print("=" * 80)

for claim in test_claims:
    print("\n" + "-" * 80)
    print(f"CLAIM: {claim}")
    print("-" * 80)

    # Recuperar documentos
    docs, scores = retriever.retrieve(claim, return_scores=True)

    print(f"\n✅ Recuperados {len(docs)} documentos\n")

    # Analizar qué años aparecen en los documentos
    found_1902 = False

    # Mostrar top 5 (aumentado para ver más contexto)
    for i, (doc, score) in enumerate(zip(docs[:5], scores[:5]), 1):
        print(f"--- DOCUMENTO {i} (Score: {score:.3f}) ---")
        preview = doc.page_content[:300].replace('\n', ' ')
        print(f"{preview}...\n")

        # Verificar si contiene información relevante sobre fundación
        content_lower = doc.page_content.lower()
        has_foundation = any(
            word in content_lower for word in ['fundado', 'fundación', 'registrado', 'creado', 'establecido'])
        has_1902 = '1902' in content_lower

        if has_1902:
            found_1902 = True

        if has_foundation and has_1902:
            print("✅✅ PERFECTO: Contiene fundación Y menciona 1902")
        elif has_foundation:
            print("✅ Contiene palabras relacionadas con fundación")
        elif has_1902:
            print("⚠️  Contiene 1902 pero sin contexto de fundación")
        else:
            print("❌ NO contiene información relevante sobre fundación")
        print()

    # Resumen
    if found_1902:
        print("🎯 ÉXITO: Se encontró el año correcto (1902) en los resultados")
    else:
        print("❌ FALLO: NO se encontró el año correcto (1902) en los resultados")
    print()

print("\n" + "=" * 80)
print("FIN DE PRUEBAS")
print("=" * 80)



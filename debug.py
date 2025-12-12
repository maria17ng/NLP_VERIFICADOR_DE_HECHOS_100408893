"""
Script para diagnosticar el contenido de ChromaDB.
Muestra los metadatos de los primeros documentos para entender qué está fallando.
"""

import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_config():
    """Cargar configuración."""
    with open("settings/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """Diagnosticar ChromaDB."""
    print("=" * 80)
    print("DIAGNÓSTICO DE CHROMADB")
    print("=" * 80)

    config = load_config()

    # Cargar embeddings
    print("\n🔧 Cargando embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config['models']['embeddings']['name'],
        model_kwargs={'device': 'cpu', 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("   ✅ Embeddings cargados")

    # Conectar a ChromaDB
    print("\n🔧 Conectando a ChromaDB...")
    vector_db = Chroma(
        persist_directory="data/vector_store",
        embedding_function=embeddings
    )

    total_docs = vector_db._collection.count()
    print(f"   ✅ {total_docs} documentos en BD")

    # Obtener primeros documentos
    print("\n📄 MUESTRA DE DOCUMENTOS (primeros 10):")
    print("=" * 80)

    # Query para obtener todos los docs
    all_docs = vector_db.similarity_search("Real Madrid", k=10)

    for i, doc in enumerate(all_docs, 1):
        print(f"\n--- Documento {i} ---")
        print(f"Contenido: {doc.page_content[:100]}...")
        print(f"Metadatos:")
        for key, value in doc.metadata.items():
            print(f"  - {key}: {value}")

    # Buscar docs que mencionen "1902" (fecha de fundación)
    print("\n" + "=" * 80)
    print("BÚSQUEDA: Documentos que mencionen '1902'")
    print("=" * 80)

    docs_1902 = vector_db.similarity_search("fundado 1902 registrado", k=5)

    for i, doc in enumerate(docs_1902, 1):
        print(f"\n--- Resultado {i} ---")
        # Buscar "1902" en el contenido
        if "1902" in doc.page_content:
            print("✅ CONTIENE '1902'")
            # Mostrar contexto
            idx = doc.page_content.find("1902")
            start = max(0, idx - 50)
            end = min(len(doc.page_content), idx + 100)
            print(f"Contexto: ...{doc.page_content[start:end]}...")
        else:
            print("❌ NO CONTIENE '1902'")
            print(f"Preview: {doc.page_content[:150]}...")

        print(f"Metadatos:")
        for key, value in doc.metadata.items():
            print(f"  - {key}: {value}")

    # Verificar si hay docs con section_number válido
    print("\n" + "=" * 80)
    print("VERIFICACIÓN: ¿Hay documentos con section_number válido?")
    print("=" * 80)

    # Obtener una muestra más grande
    large_sample = vector_db.similarity_search("Real Madrid historia", k=50)

    section_numbers = [doc.metadata.get('section_number', -1) for doc in large_sample]
    valid_sections = [s for s in section_numbers if s != -1 and s is not None]

    print(f"\nTotal docs muestreados: {len(large_sample)}")
    print(f"Docs con section_number válido: {len(valid_sections)}")
    print(f"Docs con section_number=-1 o None: {len(section_numbers) - len(valid_sections)}")

    if valid_sections:
        print(f"\nSection numbers encontrados: {sorted(set(valid_sections))[:10]}...")
    else:
        print("\n❌ NINGÚN DOCUMENTO TIENE section_number VÁLIDO")
        print("\n🔍 Metadatos disponibles en los documentos:")
        # Mostrar qué metadatos SÍ tienen
        all_keys = set()
        for doc in large_sample[:10]:
            all_keys.update(doc.metadata.keys())
        print(f"   Claves encontradas: {sorted(all_keys)}")

    print("\n" + "=" * 80)
    print("FIN DEL DIAGNÓSTICO")
    print("=" * 80)


if __name__ == "__main__":
    main()

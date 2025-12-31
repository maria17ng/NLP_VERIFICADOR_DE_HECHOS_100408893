"""
Script de Inspección de Base de Datos Vectorial.

Permite verificar el contenido de ChromaDB y explorar documentos indexados.

Autor: Proyecto Final NLP - UC3M
Fecha: Diciembre 2025
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from utils.utils import ConfigManager, setup_logger


class VectorDBInspector:
    """Inspector de base de datos vectorial."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el inspector.

        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = ConfigManager(config_path)
        self.logger = setup_logger('VectorDBInspector', level='INFO')

        # Cargar embeddings
        model_name = self.config.get('models.embeddings.name')
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # Conectar a BD
        db_path = self.config.get_path('vector_store')

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"❌ Base de datos no encontrada en: {db_path}")

        self.vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

        self.logger.info(f"✅ Conectado a BD: {db_path}")

    def get_stats(self) -> dict:
        """Obtiene estadísticas de la BD."""
        collection = self.vector_db._collection

        stats = {
            'total_documents': collection.count(),
            'collection_name': collection.name,
        }

        return stats

    def list_sample_documents(self, n: int = 5):
        """
        Lista documentos de muestra.

        Args:
            n: Número de documentos a mostrar
        """
        collection = self.vector_db._collection

        # Obtener muestra
        results = collection.get(
            limit=n,
            include=['documents', 'metadatas']
        )

        print(f"\n{'=' * 80}")
        print(f"📄 MUESTRA DE {n} DOCUMENTOS")
        print(f"{'=' * 80}\n")

        for i, (doc_id, document, metadata) in enumerate(zip(
                results['ids'],
                results['documents'],
                results['metadatas']
        ), 1):
            print(f"--- Documento {i} ---")
            print(f"ID: {doc_id}")
            print(f"\n📝 Contenido ({len(document)} chars):")
            print(document[:300] + "..." if len(document) > 300 else document)
            print(f"\n🏷️  Metadatos:")
            for key, value in metadata.items():
                print(f"  • {key}: {value}")
            print(f"\n{'-' * 80}\n")

    def search(self, query: str, k: int = 5):
        """
        Busca documentos similares a una consulta.

        Args:
            query: Texto de búsqueda
            k: Número de resultados
        """
        print(f"\n{'=' * 80}")
        print(f"🔍 BÚSQUEDA: '{query}'")
        print(f"{'=' * 80}\n")

        results = self.vector_db.similarity_search_with_score(query, k=k)

        for i, (doc, score) in enumerate(results, 1):
            print(f"--- Resultado {i} (score: {score:.4f}) ---")
            print(f"📝 Contenido:")
            print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
            print(f"\n🏷️  Metadatos:")
            for key, value in doc.metadata.items():
                print(f"  • {key}: {value}")
            print(f"\n{'-' * 80}\n")

    def get_by_source(self, source_name: str):
        """
        Obtiene todos los chunks de un archivo fuente.

        Args:
            source_name: Nombre del archivo fuente
        """
        collection = self.vector_db._collection

        # Obtener todos los documentos
        all_docs = collection.get(
            include=['documents', 'metadatas']
        )

        # Filtrar por fuente
        matching_docs = []
        for doc, metadata in zip(all_docs['documents'], all_docs['metadatas']):
            if source_name.lower() in metadata.get('source', '').lower():
                matching_docs.append((doc, metadata))

        print(f"\n{'=' * 80}")
        print(f"📂 DOCUMENTOS DE: {source_name}")
        print(f"{'=' * 80}")
        print(f"Total encontrados: {len(matching_docs)}\n")

        for i, (doc, metadata) in enumerate(matching_docs[:10], 1):  # Mostrar solo 10
            print(f"--- Chunk {i} ---")
            print(f"📝 Contenido ({len(doc)} chars):")
            print(doc[:200] + "..." if len(doc) > 200 else doc)
            print(f"\n🏷️  Metadatos clave:")
            print(f"  • source: {metadata.get('source')}")
            print(f"  • chunk_id: {metadata.get('chunk_id', 'N/A')}")
            if 'title' in metadata:
                print(f"  • title: {metadata.get('title')}")
            if 'dates' in metadata:
                print(f"  • dates: {metadata.get('dates')}")
            print(f"\n{'-' * 80}\n")

        if len(matching_docs) > 10:
            print(f"... y {len(matching_docs) - 10} chunks más\n")

    def list_all_sources(self):
        """Lista todos los archivos fuente en la BD."""
        collection = self.vector_db._collection

        all_docs = collection.get(include=['metadatas'])

        sources = set()
        for metadata in all_docs['metadatas']:
            source = metadata.get('source', 'unknown')
            sources.add(source)

        print(f"\n{'=' * 80}")
        print(f"📚 ARCHIVOS FUENTE EN LA BASE DE DATOS")
        print(f"{'=' * 80}\n")

        for i, source in enumerate(sorted(sources), 1):
            # Contar chunks por fuente
            count = sum(1 for m in all_docs['metadatas'] if m.get('source') == source)
            print(f"{i}. {source} ({count} chunks)")

        print(f"\n{'=' * 80}\n")

    def analyze_metadata(self):
        """Analiza los metadatos almacenados."""
        collection = self.vector_db._collection

        all_docs = collection.get(include=['metadatas'])

        # Analizar campos de metadatos
        metadata_fields = {}
        for metadata in all_docs['metadatas']:
            for key in metadata.keys():
                if key not in metadata_fields:
                    metadata_fields[key] = {
                        'count': 0,
                        'sample_values': set()
                    }
                metadata_fields[key]['count'] += 1

                # Guardar valor de ejemplo
                value = metadata[key]
                if isinstance(value, (str, int, float, bool)):
                    if len(metadata_fields[key]['sample_values']) < 3:
                        metadata_fields[key]['sample_values'].add(str(value)[:50])

        print(f"\n{'=' * 80}")
        print(f"📊 ANÁLISIS DE METADATOS")
        print(f"{'=' * 80}\n")

        for field, info in sorted(metadata_fields.items()):
            percentage = (info['count'] / len(all_docs['metadatas'])) * 100
            print(f"• {field}")
            print(f"  - Presente en: {info['count']}/{len(all_docs['metadatas'])} docs ({percentage:.1f}%)")
            if info['sample_values']:
                samples = list(info['sample_values'])[:3]
                print(f"  - Ejemplos: {', '.join(samples)}")
            print()

        print(f"{'=' * 80}\n")


def main():
    """Función principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspeccionar base de datos vectorial"
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Mostrar estadísticas generales'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=5,
        help='Número de documentos de muestra a mostrar'
    )
    parser.add_argument(
        '--search',
        type=str,
        help='Buscar documentos similares a esta consulta'
    )
    parser.add_argument(
        '--source',
        type=str,
        help='Listar chunks de un archivo fuente específico'
    )
    parser.add_argument(
        '--sources',
        action='store_true',
        help='Listar todos los archivos fuente'
    )
    parser.add_argument(
        '--metadata',
        action='store_true',
        help='Analizar metadatos almacenados'
    )
    parser.add_argument(
        '-k',
        type=int,
        default=5,
        help='Número de resultados en búsqueda'
    )

    args = parser.parse_args()

    # Inicializar inspector
    try:
        inspector = VectorDBInspector()
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("💡 Ejecuta primero: python ingest_data.py --clear\n")
        return

    # Ejecutar operaciones
    if args.stats:
        stats = inspector.get_stats()
        print(f"\n{'=' * 80}")
        print(f"📊 ESTADÍSTICAS DE LA BASE DE DATOS")
        print(f"{'=' * 80}\n")
        for key, value in stats.items():
            print(f"• {key}: {value}")
        print(f"\n{'=' * 80}\n")

    if args.sources:
        inspector.list_all_sources()

    if args.metadata:
        inspector.analyze_metadata()

    if args.source:
        inspector.get_by_source(args.source)

    if args.search:
        inspector.search(args.search, k=args.k)

    if not any([args.stats, args.sources, args.metadata, args.source, args.search]):
        # Modo por defecto: mostrar overview completo
        print("\n" + "=" * 80)
        print("🔍 INSPECCIÓN COMPLETA DE LA BASE DE DATOS")
        print("=" * 80)

        # Estadísticas
        stats = inspector.get_stats()
        print(f"\n📊 Total de documentos: {stats['total_documents']}")
        print(f"📦 Colección: {stats['collection_name']}")

        # Fuentes
        inspector.list_all_sources()

        # Análisis de metadatos
        inspector.analyze_metadata()

        # Muestra
        inspector.list_sample_documents(n=args.sample)

        print("\n💡 Comandos útiles:")
        print("  python inspect_db.py --search 'tu consulta'")
        print("  python inspect_db.py --source 'nombre_archivo.txt'")
        print("  python inspect_db.py --sources")
        print("  python inspect_db.py --metadata")
        print()


if __name__ == "__main__":
    main()

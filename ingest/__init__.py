"""
Sistema de Ingesta de Documentos para Base de Datos Vectorial.

Este módulo carga documentos (TXT y PDF), los divide en fragmentos (chunks)
y los almacena en una base de datos vectorial (ChromaDB) para su posterior
recuperación mediante búsqueda semántica.

Características:
- Soporte para archivos .txt y .pdf
- Fragmentación inteligente con solapamiento
- Metadatos detallados para citación precisa
- Logging completo del proceso
- Gestión de errores robusta

Autor: Proyecto Final NLP - UC3M
Fecha: Diciembre 2025
"""

from ingest.ingest_data import DocumentIngester

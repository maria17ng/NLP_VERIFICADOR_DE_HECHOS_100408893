"""
Módulo de Preprocesamiento de Documentos.

Este módulo proporciona funciones para limpiar y normalizar documentos
antes de su ingesta en la base de datos vectorial.

Características:
- Limpieza de caracteres especiales y espacios
- Normalización de texto (unicode, mayúsculas/minúsculas)
- Eliminación de contenido irrelevante (headers, footers, URLs)
- Detección de estructura (títulos, secciones, párrafos)
- Corrección de encoding

Autor: Proyecto Final NLP - UC3M
Fecha: Diciembre 2025
"""

from preprocessor.document_preprocessor import DocumentPreprocessor
from preprocessor.wikipedia_preprocessor import WikipediaPreprocessor
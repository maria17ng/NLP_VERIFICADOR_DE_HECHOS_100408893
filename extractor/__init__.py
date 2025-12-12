"""
Módulo de Extracción de Metadatos.

Este módulo extrae metadatos enriquecidos de documentos para mejorar
la recuperación y citación en el sistema RAG.

Características:
- Extracción de títulos y secciones
- Detección de fechas y entidades temporales
- Clasificación de tipo de información
- Detección de entidades nombradas
- Análisis de densidad de información

Autor: Proyecto Final NLP - UC3M
Fecha: Diciembre 2025
"""

from extractor.metadata_extractor import MetadataExtractor
from extractor.chunk_metadata import ChunkMetadataEnricher
from extractor.fact_metadata_extractor import FactMetadataExtractor
from extractor.topic_extractor import TopicExtractor

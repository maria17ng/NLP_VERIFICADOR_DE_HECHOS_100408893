"""
Módulo de Chunking Semántico.

Este módulo proporciona estrategias avanzadas de chunking que respetan
la estructura natural del texto (oraciones, párrafos, secciones).

Características:
- Chunking por oraciones usando spaCy
- Chunking por párrafos con contexto
- Chunking híbrido (múltiples tamaños)
- Respeto a límites semánticos
- Overlapping inteligente

Autor: Proyecto Final NLP - UC3M
Fecha: Diciembre 2025
"""

from chunker.hybrid_chunker import HybridChunker
from chunker.semantic_chunker import SemanticChunker
from chunker.section_aware import SectionAwareChunker

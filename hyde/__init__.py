"""
Módulo de Generación de Preguntas Hipotéticas (HyDE).

HyDE (Hypothetical Document Embeddings) mejora la recuperación generando
preguntas que cada chunk podría responder. Esto ayuda a encontrar chunks
relevantes incluso cuando la consulta del usuario no coincide exactamente
con el texto del documento.

Características:
- Generación automática de preguntas por chunk
- Múltiples perspectivas de consulta
- Almacenamiento de preguntas como metadatos
- Mejora del matching semántico

Autor: Proyecto Final NLP - UC3M
Fecha: Diciembre 2025
"""

from hyde.hyde_augmenter import HyDEAugmenter
from hyde.hyde_generator import HyDEGenerator
from hyde.simple_hyde import SimpleHyDEGenerator

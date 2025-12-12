"""
Sistema de Verificación de Hechos con RAG (Retrieval-Augmented Generation).

Este módulo implementa un verificador de hechos que:
- Utiliza una base de datos vectorial para recuperar evidencia
- Emplea un LLM para evaluar la veracidad de afirmaciones
- Soporta multilingüismo con traducción automática
- Implementa caché para optimizar consultas repetidas
- Proporciona citaciones precisas de las fuentes

Autor: Proyecto Final NLP - UC3M
Fecha: Diciembre 2025
"""

from verifier.verifier import FactChecker

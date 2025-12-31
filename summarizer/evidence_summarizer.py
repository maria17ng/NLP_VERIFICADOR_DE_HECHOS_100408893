"""
Generador de resúmenes de evidencia.

Este módulo proporciona capacidades de generación de resúmenes automáticos
del contenido recuperado, usando técnicas extractivas y abstractivas.
"""

from typing import List, Dict, Any, Optional
import re


class EvidenceSummarizer:
    """
    Genera resúmenes del contenido recuperado.

    Soporta dos métodos:
    1. Resumen extractivo: Selecciona las frases más relevantes
    2. Resumen abstractivo: Usa LLM para generar resumen condensado
    """

    def __init__(self, llm=None):
        """
        Inicializa el summarizer.

        Args:
            llm: Modelo de lenguaje opcional para resúmenes abstractivos
        """
        self.llm = llm

    def summarize_extractive(
            self,
            fragments: List[Dict[str, Any]],
            claim: str,
            max_sentences: int = 3
    ) -> str:
        """
        Genera un resumen extractivo seleccionando las frases más relevantes.

        Args:
            fragments: Lista de fragmentos de evidencia
            claim: Afirmación original
            max_sentences: Número máximo de frases a incluir

        Returns:
            Resumen extractivo
        """
        if not fragments:
            return "No hay evidencia disponible para resumir."

        # Extraer palabras clave del claim
        claim_words = set(re.findall(r'\b\w{4,}\b', claim.lower()))

        all_sentences = []

        # Extraer todas las frases de todos los fragmentos
        for frag in fragments:
            text = frag.get('fragmento', '')
            # Dividir en frases (simplificado)
            sentences = re.split(r'[.!?]+', text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 20:  # Ignorar frases muy cortas
                    all_sentences.append(sent)

        # Puntuar cada frase por relevancia
        scored_sentences = []
        for sent in all_sentences:
            sent_words = set(re.findall(r'\b\w{4,}\b', sent.lower()))
            # Score basado en overlap con palabras del claim
            score = len(sent_words & claim_words)
            # Bonus por números (fechas, cantidades)
            if re.search(r'\b\d{4}\b', sent):
                score += 2
            scored_sentences.append((score, sent))

        # Ordenar por score y tomar las top N
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [sent for score, sent in scored_sentences[:max_sentences]]

        return " ".join(top_sentences)

    def summarize_abstractive(
            self,
            fragments: List[Dict[str, Any]],
            claim: str,
            max_length: int = 200
    ) -> Optional[str]:
        """
        Genera un resumen abstractivo usando un LLM.

        Args:
            fragments: Lista de fragmentos de evidencia
            claim: Afirmación original
            max_length: Longitud máxima del resumen en palabras

        Returns:
            Resumen abstractivo o None si no hay LLM disponible
        """
        if not self.llm:
            return None

        if not fragments:
            return "No hay evidencia disponible para resumir."

        # Construir contexto concatenado
        context_parts = []
        for i, frag in enumerate(fragments, 1):
            doc = frag.get('documento', 'Desconocido')
            text = frag.get('fragmento', '')
            context_parts.append(f"[Fragmento {i} de {doc}]: {text}")

        full_context = "\n".join(context_parts)

        # Prompt para el LLM
        prompt = f"""Resume el siguiente contenido en relación con la afirmación: "{claim}"

CONTENIDO RECUPERADO:
{full_context}

Genera un resumen conciso (máximo {max_length} palabras) que:
1. Capture la información clave relacionada con la afirmación
2. Sea objetivo y basado en el contenido
3. Mantenga las fechas, nombres y datos específicos mencionados

RESUMEN:"""

        try:
            # Invocar LLM (asumiendo que tiene método similar)
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(prompt)
                # Si es un objeto con content
                if hasattr(response, 'content'):
                    return response.content.strip()
                # Si es string directo
                return str(response).strip()
            else:
                return None
        except Exception as e:
            return f"Error generando resumen: {e}"

    def generate_summary(
            self,
            fragments: List[Dict[str, Any]],
            claim: str,
            method: str = "extractive",
            **kwargs
    ) -> str:
        """
        Genera un resumen usando el método especificado.

        Args:
            fragments: Lista de fragmentos de evidencia
            claim: Afirmación original
            method: "extractive" o "abstractive"
            **kwargs: Argumentos adicionales para el método específico

        Returns:
            Resumen generado
        """
        if method == "extractive":
            return self.summarize_extractive(fragments, claim, **kwargs)
        elif method == "abstractive":
            result = self.summarize_abstractive(fragments, claim, **kwargs)
            # Fallback a extractivo si abstractivo falla
            if result is None:
                return self.summarize_extractive(fragments, claim)
            return result
        else:
            raise ValueError(f"Método '{method}' no soportado. Use 'extractive' o 'abstractive'")

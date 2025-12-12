import re
from langchain_core.documents import Document


class SimpleHyDEGenerator:
    """
    Generador HyDE simplificado sin dependencias de NLP.

    Usa solo expresiones regulares y heurísticas simples.
    Ideal para cuando no se tiene spaCy instalado.
    """

    def __init__(self, num_questions: int = 3):
        """
        Inicializa el generador simple.

        Args:
            num_questions: Número de preguntas a generar
        """
        self.num_questions = num_questions

    def generate_questions(self, chunk: Document) -> Document:
        """
        Genera preguntas simples basadas en heurísticas.

        Args:
            chunk: Chunk a procesar

        Returns:
            Chunk con preguntas
        """
        text = chunk.page_content
        questions = []

        # Extraer primer sustantivo propio como tema principal
        proper_nouns = re.findall(r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*\b', text)

        if proper_nouns:
            main_topic = proper_nouns[0]
            questions.append(f"¿Qué es {main_topic}?")
            questions.append(f"¿Cuál es la historia de {main_topic}?")
            questions.append(f"¿Qué información hay sobre {main_topic}?")

        # Preguntas genéricas si no hay sustantivos propios
        if not questions:
            questions = [
                "¿Qué información contiene este texto?",
                "¿De qué trata este documento?",
                "¿Qué datos importantes se mencionan?"
            ]

        chunk.metadata['hypothetical_questions'] = questions[:self.num_questions]
        return chunk

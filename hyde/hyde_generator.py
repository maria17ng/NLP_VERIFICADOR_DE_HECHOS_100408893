import re
from typing import List, Dict, Any
from langchain_core.documents import Document


class HyDEGenerator:
    """
    Generador de preguntas hipotéticas para chunks.

    Attributes:
        num_questions: Número de preguntas a generar por chunk
        min_chunk_length: Longitud mínima de chunk para generar preguntas
    """

    def __init__(self, num_questions: int = 3, min_chunk_length: int = 100):
        """
        Inicializa el generador HyDE.

        Args:
            num_questions: Preguntas a generar
            min_chunk_length: Longitud mínima de chunk
        """
        self.num_questions = num_questions
        self.min_chunk_length = min_chunk_length

    def generate_questions(self, chunk: Document) -> Document:
        """
        Genera preguntas hipotéticas para un chunk.

        Args:
            chunk: Chunk para el cual generar preguntas

        Returns:
            Chunk con preguntas añadidas a metadatos
        """
        text = chunk.page_content

        # Solo generar si el chunk es suficientemente largo
        if len(text) < self.min_chunk_length:
            chunk.metadata['hypothetical_questions'] = []
            return chunk

        # Generar preguntas basadas en análisis del contenido
        questions = self._generate_questions_from_text(text, chunk.metadata)

        # Añadir a metadatos
        chunk.metadata['hypothetical_questions'] = questions

        return chunk

    def _generate_questions_from_text(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Genera preguntas basadas en el contenido del texto.

        Usa heurísticas para identificar información factual y
        generar preguntas apropiadas.

        Args:
            text: Contenido del chunk
            metadata: Metadatos del chunk

        Returns:
            Lista de preguntas generadas
        """
        questions = []

        # 1. Preguntas basadas en entidades
        questions.extend(self._generate_entity_questions(metadata))

        # 2. Preguntas basadas en fechas/eventos
        questions.extend(self._generate_temporal_questions(text, metadata))

        # 3. Preguntas basadas en definiciones
        questions.extend(self._generate_definition_questions(text))

        # 4. Preguntas basadas en relaciones
        questions.extend(self._generate_relation_questions(text))

        # 5. Preguntas basadas en números/estadísticas
        questions.extend(self._generate_numerical_questions(text))

        # Limitar al número deseado
        return questions[:self.num_questions]

    @staticmethod
    def _generate_entity_questions(metadata: Dict[str, Any]) -> List[str]:
        """
        Genera preguntas sobre entidades mencionadas.

        Args:
            metadata: Metadatos con entidades

        Returns:
            Lista de preguntas sobre entidades
        """
        questions = []

        # Preguntas sobre personas
        if metadata.get('persons'):
            person = metadata['persons'][0]
            questions.append(f"¿Quién es {person}?")
            questions.append(f"¿Qué información hay sobre {person}?")

        # Preguntas sobre organizaciones
        if metadata.get('organizations'):
            org = metadata['organizations'][0]
            questions.append(f"¿Qué es {org}?")
            questions.append(f"¿Cuál es la historia de {org}?")

        # Preguntas sobre lugares
        if metadata.get('locations'):
            loc = metadata['locations'][0]
            questions.append(f"¿Dónde está {loc}?")
            questions.append(f"¿Qué características tiene {loc}?")

        return questions

    @staticmethod
    def _generate_temporal_questions(text: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Genera preguntas sobre eventos temporales.

        Args:
            text: Texto del chunk
            metadata: Metadatos con fechas

        Returns:
            Lista de preguntas temporales
        """
        questions = []

        # Si hay fechas
        if metadata.get('dates') or metadata.get('first_date'):
            # Patrones de eventos temporales
            if re.search(r'\bfundad[oa]s?\b|\bcread[oa]s?\b|\binaugur\w+\b', text, re.IGNORECASE):
                questions.append("¿Cuándo fue fundado?")
                questions.append("¿En qué año se creó?")

            if re.search(r'\bnaci[óo]\b|\bnacid[oa]\b', text, re.IGNORECASE):
                questions.append("¿Cuándo nació?")
                questions.append("¿En qué fecha nació?")

            if re.search(r'\bocurri[óo]\b|\bsucedi[óo]\b|\btuvo lugar\b', text, re.IGNORECASE):
                questions.append("¿Cuándo ocurrió?")
                questions.append("¿En qué fecha sucedió?")

        return questions

    @staticmethod
    def _generate_definition_questions(text: str) -> List[str]:
        """
        Genera preguntas de definición.

        Args:
            text: Texto del chunk

        Returns:
            Lista de preguntas de definición
        """
        questions = []

        # Detectar definiciones (patrones como "es un/una", "se define como")
        definition_patterns = [
            r'(\w+(?:\s+\w+)*)\s+es\s+un[a]?\s+(\w+)',
            r'(\w+(?:\s+\w+)*)\s+se\s+define\s+como',
            r'(\w+(?:\s+\w+)*)\s+consiste\s+en',
            r'(\w+(?:\s+\w+)*)\s+se\s+refiere\s+a'
        ]

        for pattern in definition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).strip()
                if len(subject.split()) <= 4:  # Evitar frases muy largas
                    questions.append(f"¿Qué es {subject}?")
                    questions.append(f"¿Cómo se define {subject}?")
                    break  # Solo uno por patrón

            if questions:
                break

        return questions

    @staticmethod
    def _generate_relation_questions(text: str) -> List[str]:
        """
        Genera preguntas sobre relaciones entre entidades.

        Args:
            text: Texto del chunk

        Returns:
            Lista de preguntas sobre relaciones
        """
        questions = []

        # Relaciones comunes
        if re.search(r'\bjuega\s+en\b|\bpertenece\s+a\b|\bes\s+parte\s+de\b', text, re.IGNORECASE):
            questions.append("¿Dónde juega?")
            questions.append("¿A qué pertenece?")

        if re.search(r'\bubicad[oa]\s+en\b|\bsitúad[oa]\s+en\b', text, re.IGNORECASE):
            questions.append("¿Dónde está ubicado?")
            questions.append("¿Dónde se encuentra?")

        if re.search(r'\bpreside\b|\bdirige\b|\blidera\b', text, re.IGNORECASE):
            questions.append("¿Quién lo dirige?")
            questions.append("¿Quién está al mando?")

        if re.search(r'\bgan[óo]\b|\bconsigui[óo]\b|\blogrón', text, re.IGNORECASE):
            questions.append("¿Qué logró?")
            questions.append("¿Qué ganó?")

        return questions

    @staticmethod
    def _generate_numerical_questions(text: str) -> List[str]:
        """
        Genera preguntas sobre datos numéricos.

        Args:
            text: Texto del chunk

        Returns:
            Lista de preguntas numéricas
        """
        questions = []

        # Detectar números y su contexto
        number_contexts = [
            (r'(\d+)\s+títulos?\b', "¿Cuántos títulos tiene?"),
            (r'(\d+)\s+veces\b', "¿Cuántas veces?"),
            (r'(\d+)\s+años?\b', "¿Cuántos años?"),
            (r'(\d+)\s+millones?\b', "¿Cuántos millones?"),
            (r'(\d+)\s+jugadores?\b', "¿Cuántos jugadores?"),
            (r'(\d+)\s+miembros?\b', "¿Cuántos miembros?"),
            (r'(\d+)%', "¿Qué porcentaje?"),
        ]

        for pattern, question in number_contexts:
            if re.search(pattern, text, re.IGNORECASE):
                questions.append(question)
                break  # Solo una pregunta numérica

        return questions

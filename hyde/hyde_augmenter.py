from typing import List, Optional
from langchain_core.documents import Document
from hyde.hyde_generator import HyDEGenerator


class HyDEAugmenter:
    """
    Aumenta chunks con preguntas hipotéticas para mejorar recuperación.

    Crea documentos adicionales con las preguntas que luego se indexan
    junto con el contenido original.
    """

    def __init__(self, generator: Optional[HyDEGenerator] = None, create_question_docs: bool = True):
        """
        Inicializa el aumentador HyDE.

        Args:
            generator: Generador de preguntas
            create_question_docs: Si crear documentos separados para preguntas
        """
        self.generator = generator or HyDEGenerator()
        self.create_question_docs = create_question_docs

    def augment_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Aumenta chunks con preguntas hipotéticas.

        Args:
            chunks: Lista de chunks a aumentar

        Returns:
            Lista aumentada con chunks originales y de preguntas
        """
        augmented_chunks = []

        for chunk in chunks:
            # Generar preguntas
            chunk_with_questions = self.generator.generate_questions(chunk)
            augmented_chunks.append(chunk_with_questions)

            # Crear documentos adicionales de preguntas si se solicita
            if self.create_question_docs:
                question_docs = self._create_question_documents(chunk_with_questions)
                augmented_chunks.extend(question_docs)

        return augmented_chunks

    @staticmethod
    def _create_question_documents(chunk: Document) -> List[Document]:
        """
        Crea documentos separados para cada pregunta.

        Estos documentos tienen como contenido la pregunta pero apuntan
        al chunk original, mejorando el matching cuando la consulta del
        usuario es similar a las preguntas generadas.

        Args:
            chunk: Chunk con preguntas generadas

        Returns:
            Lista de documentos de preguntas
        """
        questions = chunk.metadata.get('hypothetical_questions', [])
        if not questions:
            return []

        question_docs = []

        for i, question in enumerate(questions):
            # Crear documento de pregunta
            question_metadata = chunk.metadata.copy()
            question_metadata.update({
                'is_question_doc': True,
                'question_index': i,
                'original_chunk_content': chunk.page_content,
                'document_type': 'hypothetical_question'
            })

            # El contenido es la pregunta + preview del chunk
            preview = chunk.page_content[:200] + "..."
            content = f"{question}\n\n{preview}"

            question_doc = Document(
                page_content=content,
                metadata=question_metadata
            )

            question_docs.append(question_doc)

        return question_docs

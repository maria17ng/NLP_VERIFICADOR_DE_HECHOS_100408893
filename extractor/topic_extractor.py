"""
Topic Extractor usando Gensim LDA para detección automática de temas.
Basado en el temario de NLP - Topic Modeling con Gensim.
"""
import re
from typing import List, Dict, Tuple
from langchain_core.documents import Document
import warnings
warnings.filterwarnings('ignore')


class TopicExtractor:
    """
    Extrae temas de documentos usando LDA (Latent Dirichlet Allocation) de Gensim.

    Basado en el enfoque académico de la asignatura:
    - Usa Gensim para topic modeling
    - Contabiliza frecuencias de palabras (mejor que LLM)
    - Detecta temas latentes automáticamente
    """

    def __init__(self, num_topics: int = 10, passes: int = 10):
        """
        Inicializa el extractor de temas.

        Args:
            num_topics: Número de temas a detectar (default: 10)
            passes: Número de pasadas del algoritmo LDA (default: 10)
        """
        self.num_topics = num_topics
        self.passes = passes
        self.lda_model = None
        self.dictionary = None
        self.topic_labels = {}

        # Intentar importar gensim
        try:
            from gensim import corpora
            from gensim.models import LdaModel
            import spacy

            self.corpora = corpora
            self.LdaModel = LdaModel
            self.gensim_available = True

            # Cargar spaCy para preprocesamiento
            try:
                self.nlp = spacy.load("es_core_news_sm")
            except:
                print("⚠️ spaCy no disponible - usando preprocesamiento básico")
                self.nlp = None

        except ImportError:
            print("⚠️ Gensim no está instalado. Usando detección básica de temas.")
            self.gensim_available = False

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocesa texto para topic modeling.

        Args:
            text: Texto a preprocesar

        Returns:
            Lista de tokens limpios
        """
        if self.nlp:
            # Usar spaCy para lemmatización y limpieza
            doc = self.nlp(text.lower())

            # Filtrado MÁS PERMISIVO para mantener términos importantes
            tokens = [
                token.lemma_ for token in doc
                if not token.is_punct  # Solo eliminar puntuación
                and len(token.text) > 2
                and token.is_alpha
                # NO eliminar stopwords completamente - algunas son útiles para temas
                and token.text not in ['el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'y', 'o', 'en']
            ]

            # Si quedaron muy pocos tokens (< 3), usar versión sin stopwords
            if len(tokens) < 3:
                tokens = [
                    token.lemma_ for token in doc
                    if not token.is_punct
                    and len(token.text) > 2
                    and token.is_alpha
                ]
        else:
            # Fallback: limpieza básica con regex
            text_lower = text.lower()
            # Remover puntuación pero MANTENER números (importantes para contexto)
            text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
            tokens = [
                word for word in text_clean.split()
                if len(word) > 2
            ]

        return tokens

    def train(self, documents: List[Document]) -> None:
        """
        Entrena el modelo LDA con los documentos proporcionados.

        Args:
            documents: Lista de documentos de Langchain
        """
        if not self.gensim_available:
            print("⚠️ Gensim no disponible - no se puede entrenar modelo LDA")
            return

        print(f"🔍 Entrenando modelo LDA con {len(documents)} documentos...")

        # 1. Preprocesar todos los documentos
        corpus_tokenized = []
        for doc in documents:
            tokens = self._preprocess_text(doc.page_content)
            if tokens:  # Solo agregar si hay tokens
                corpus_tokenized.append(tokens)

        print(f"   Documentos tokenizados: {len(corpus_tokenized)}")

        # 2. Crear diccionario Gensim
        self.dictionary = self.corpora.Dictionary(corpus_tokenized)

        # Filtrar extremos ADAPTATIVAMENTE según tamaño del corpus
        num_docs = len(corpus_tokenized)

        if num_docs > 50:
            # Corpus grande: filtrado estricto
            no_below = 2
            no_above = 0.5
            keep_n = 1000
        elif num_docs > 10:
            # Corpus mediano: filtrado moderado
            no_below = 1  # Permitir palabras que aparecen solo 1 vez
            no_above = 0.7
            keep_n = 500
        else:
            # Corpus pequeño: NO filtrar (para tests)
            no_below = 1
            no_above = 1.0  # Permitir todas las frecuencias
            keep_n = None  # Sin límite

        if keep_n:
            self.dictionary.filter_extremes(
                no_below=no_below,
                no_above=no_above,
                keep_n=keep_n
            )
        else:
            # Solo filtrar mínimo para corpus muy pequeños
            self.dictionary.filter_extremes(
                no_below=no_below,
                no_above=no_above
            )

        print(f"   Vocabulario: {len(self.dictionary)} palabras únicas")

        # 3. Crear corpus Bag-of-Words
        corpus_bow = [self.dictionary.doc2bow(doc) for doc in corpus_tokenized]

        # 4. Entrenar modelo LDA
        print(f"   Entrenando LDA ({self.num_topics} temas, {self.passes} pasadas)...")
        self.lda_model = self.LdaModel(
            corpus=corpus_bow,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=42,
            passes=self.passes,
            alpha='auto',
            per_word_topics=True
        )

        # 5. Extraer y etiquetar temas
        self._generate_topic_labels()

        print("✅ Modelo LDA entrenado correctamente")
        print(f"\n📊 Temas detectados:")
        for topic_id, label in self.topic_labels.items():
            print(f"   • Tema {topic_id}: {label}")

    def _generate_topic_labels(self) -> None:
        """
        Genera etiquetas descriptivas para cada tema basadas en palabras clave.
        """
        if not self.lda_model:
            return

        for idx in range(self.num_topics):
            # Obtener top 5 palabras del tema
            top_words = self.lda_model.show_topic(idx, topn=5)
            # Concatenar palabras como etiqueta
            label = ", ".join([word for word, _ in top_words])
            self.topic_labels[idx] = label

    def get_document_topics(self, doc: Document) -> Dict[str, any]:
        """
        Obtiene los temas principales de un documento.

        Args:
            doc: Documento de Langchain

        Returns:
            Diccionario con temas detectados y sus probabilidades
        """
        if not self.gensim_available or not self.lda_model:
            return self._fallback_topic_detection(doc)

        # Preprocesar documento
        tokens = self._preprocess_text(doc.page_content)

        if not tokens:
            return {'topics': [], 'main_topic': None, 'topic_probabilities': {}}

        # Convertir a BOW
        bow = self.dictionary.doc2bow(tokens)

        # Obtener distribución de temas
        topic_dist = self.lda_model.get_document_topics(bow)

        # Ordenar por probabilidad
        topic_dist_sorted = sorted(topic_dist, key=lambda x: x[1], reverse=True)

        # Tema principal (el de mayor probabilidad)
        main_topic_id, main_topic_prob = topic_dist_sorted[0] if topic_dist_sorted else (None, 0.0)
        main_topic_label = self.topic_labels.get(main_topic_id, "unknown")

        # Todos los temas con prob > 0.1
        significant_topics = [
            (self.topic_labels.get(topic_id, f"topic_{topic_id}"), prob)
            for topic_id, prob in topic_dist_sorted
            if prob > 0.1
        ]

        return {
            'topics': [label for label, _ in significant_topics],
            'main_topic': main_topic_label,
            'main_topic_prob': float(main_topic_prob),
            'topic_probabilities': {label: float(prob) for label, prob in significant_topics}
        }

    def _fallback_topic_detection(self, doc: Document) -> Dict[str, any]:
        """
        Detección básica de temas si Gensim no está disponible.

        Usa patrones de palabras clave como fallback.
        """
        text_lower = doc.page_content.lower()

        topics = []

        # Categorías genéricas basadas en palabras clave
        keyword_topics = {
            'creación/fundación': ['fundado', 'fundación', 'creado', 'creación',
                                  'establecido', 'registrado', 'inicio', 'comenzó'],
            'eventos/logros': ['ganó', 'consiguió', 'logró', 'premio',
                              'reconocimiento', 'victoria'],
            'lugares': ['ubicado', 'situado', 'lugar', 'edificio', 'ciudad'],
            'personas': ['nacido', 'nació', 'persona', 'familia', 'vida'],
            'historia': ['historia', 'histórico', 'época', 'era', 'periodo'],
            'fechas': [r'\b\d{4}\b'],  # Años
        }

        for topic, keywords in keyword_topics.items():
            for keyword in keywords:
                if re.search(keyword, text_lower):
                    topics.append(topic)
                    break

        main_topic = topics[0] if topics else 'general'

        return {
            'topics': topics,
            'main_topic': main_topic,
            'main_topic_prob': 1.0 if topics else 0.0,
            'topic_probabilities': {t: 1.0 for t in topics}
        }

    def enrich_document_with_topics(self, doc: Document) -> Document:
        """
        Enriquece un documento con información de temas detectados.

        Args:
            doc: Documento a enriquecer

        Returns:
            Documento con metadata de temas
        """
        topic_info = self.get_document_topics(doc)

        # Agregar a metadata
        doc.metadata.update({
            'topics': ", ".join(topic_info['topics']) if topic_info['topics'] else "",
            'main_topic': topic_info['main_topic'],
            'main_topic_prob': topic_info['main_topic_prob'],
            'num_topics': len(topic_info['topics']),
            'has_topics': len(topic_info['topics']) > 0
        })

        return doc

    def enrich_documents(self, docs: List[Document]) -> List[Document]:
        """
        Enriquece múltiples documentos con información de temas.

        Args:
            docs: Lista de documentos

        Returns:
            Lista de documentos enriquecidos
        """
        enriched_docs = []

        for i, doc in enumerate(docs):
            enriched_doc = self.enrich_document_with_topics(doc)
            enriched_docs.append(enriched_doc)

            if (i + 1) % 100 == 0:
                print(f"   Procesados {i + 1}/{len(docs)} documentos")

        return enriched_docs

    def save_model(self, path: str) -> None:
        """Guarda el modelo LDA entrenado."""
        if self.lda_model:
            self.lda_model.save(path)
            self.dictionary.save(f"{path}.dict")
            print(f"✅ Modelo guardado en: {path}")

    def load_model(self, path: str) -> None:
        """Carga un modelo LDA previamente entrenado."""
        if self.gensim_available:
            self.lda_model = self.LdaModel.load(path)
            self.dictionary = self.corpora.Dictionary.load(f"{path}.dict")
            self._generate_topic_labels()
            print(f"✅ Modelo cargado desde: {path}")

import re
from typing import Dict, List, Optional
from langchain_core.documents import Document

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class MetadataExtractor:
    """
    Extractor de metadatos enriquecidos de documentos.

    Attributes:
        extract_dates: Si extraer fechas
        extract_entities: Si extraer entidades nombradas
        classify_content: Si clasificar tipo de contenido
    """

    def __init__(self, extract_dates: bool = True, extract_entities: bool = True, classify_content: bool = True):
        """
        Inicializa el extractor de metadatos.

        Args:
            extract_dates: Si extraer fechas
            extract_entities: Si extraer entidades
            classify_content: Si clasificar contenido
        """
        self.extract_dates = extract_dates
        self.extract_entities = extract_entities
        self.classify_content = classify_content

        # Cargar modelo de spaCy si está disponible
        self.nlp = None
        if SPACY_AVAILABLE and extract_entities:
            try:
                self.nlp = spacy.load("es_core_news_sm")
            except OSError:
                print("⚠️  Modelo spaCy 'es_core_news_sm' no encontrado.")

    def extract_metadata(self, document: Document) -> Document:
        """
        Extrae metadatos enriquecidos de un documento.

        Args:
            document: Documento a procesar

        Returns:
            Documento con metadatos enriquecidos
        """
        text = document.page_content
        metadata = document.metadata.copy()

        # Extraer título si no existe
        if 'title' not in metadata:
            title = self._extract_title(text)
            if title:
                metadata['title'] = title

        # Extraer fechas
        if self.extract_dates:
            dates = self._extract_dates(text)
            if dates:
                metadata['dates'] = dates
                metadata['first_date'] = dates[0]

        # Extraer entidades nombradas
        if self.extract_entities and self.nlp:
            entities = self._extract_entities(text)
            metadata.update(entities)

        # Clasificar tipo de contenido
        if self.classify_content:
            content_type = self._classify_content(text)
            metadata['content_type'] = content_type

        # Calcular densidad de información
        metadata['info_density'] = self._calculate_info_density(text)

        # Extraer palabras clave
        keywords = self._extract_keywords(text)
        if keywords:
            metadata['keywords'] = keywords

        # Actualizar documento
        document.metadata = metadata

        return document

    @staticmethod
    def _extract_title(text: str) -> Optional[str]:
        """
        Extrae el título del documento.

        Args:
            text: Texto del documento

        Returns:
            Título extraído o None
        """
        lines = text.strip().split('\n')

        # Buscar primera línea no vacía
        for line in lines[:5]:  # Solo primeras 5 líneas
            stripped = line.strip()
            if stripped and len(stripped) < 200:
                # Si parece un título (corto, sin punto final)
                if not stripped.endswith('.') or stripped.isupper():
                    return stripped

        return None

    @staticmethod
    def _extract_dates(text: str) -> List[str]:
        """
        Extrae fechas del texto.

        Args:
            text: Texto a analizar

        Returns:
            Lista de fechas encontradas
        """
        dates = []

        # Patrones de fechas comunes
        patterns = [
            # DD/MM/YYYY o DD-MM-YYYY
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',
            # YYYY/MM/DD
            r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
            # Mes YYYY
            r'\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?(\d{4})\b',
            # DD de Mes de YYYY
            r'\b(\d{1,2})\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+(\d{4})\b',
            # Solo año
            r'\b(19\d{2}|20\d{2})\b'
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(0)
                if date_str not in dates:
                    dates.append(date_str)

        return dates[:10]  # Limitar a 10 fechas

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extrae entidades nombradas del texto.

        Args:
            text: Texto a analizar

        Returns:
            Diccionario con entidades por tipo
        """
        # Limitar longitud para procesamiento
        text_sample = text[:5000] if len(text) > 5000 else text

        doc = self.nlp(text_sample)

        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'misc': []
        }

        for ent in doc.ents:
            if ent.label_ == 'PER':
                entities['persons'].append(ent.text)
            elif ent.label_ == 'ORG':
                entities['organizations'].append(ent.text)
            elif ent.label_ == 'LOC':
                entities['locations'].append(ent.text)
            else:
                entities['misc'].append(ent.text)

        # Eliminar duplicados y limitar cantidad
        for key in entities:
            entities[key] = list(set(entities[key]))[:10]

        return entities

    @staticmethod
    def _classify_content(text: str) -> str:
        """
        Clasifica el tipo de contenido del texto.

        Args:
            text: Texto a clasificar

        Returns:
            Tipo de contenido
        """
        text_lower = text.lower()
        text_sample = text_lower[:2000]  # Primeros 2000 caracteres

        # Indicadores por tipo
        biographical_indicators = [
            'nació', 'nacido', 'fundó', 'fundado', 'creó', 'vida',
            'biografía', 'trayectoria', 'carrera', 'formación'
        ]

        historical_indicators = [
            'historia', 'histórico', 'origen', 'evolución', 'siglo',
            'época', 'periodo', 'era', 'antiguo', 'pasado'
        ]

        statistical_indicators = [
            'estadística', 'datos', 'cifras', 'porcentaje', '%',
            'millones', 'miles', 'número', 'cantidad', 'total'
        ]

        descriptive_indicators = [
            'características', 'descripción', 'ubicado', 'situado',
            'compuesto', 'formado', 'consiste', 'contiene'
        ]

        # Contar indicadores
        scores = {
            'biographical': sum(1 for ind in biographical_indicators if ind in text_sample),
            'historical': sum(1 for ind in historical_indicators if ind in text_sample),
            'statistical': sum(1 for ind in statistical_indicators if ind in text_sample),
            'descriptive': sum(1 for ind in descriptive_indicators if ind in text_sample)
        }

        # Determinar tipo predominante
        if max(scores.values()) == 0:
            return 'general'

        return max(scores, key=scores.get)

    @staticmethod
    def _calculate_info_density(text: str) -> float:
        """
        Calcula la densidad de información del texto.

        Basado en:
        - Longitud promedio de palabras
        - Presencia de números y datos
        - Diversidad léxica

        Args:
            text: Texto a analizar

        Returns:
            Puntuación de densidad (0-1)
        """
        if not text:
            return 0.0

        # Tokenizar
        words = re.findall(r'\b\w+\b', text.lower())

        if not words:
            return 0.0

        # Métricas
        num_words = len(words)
        num_unique = len(set(words))
        lexical_diversity = num_unique / num_words if num_words > 0 else 0

        # Contar números (indicador de datos concretos)
        num_numbers = len(re.findall(r'\d+', text))
        number_density = min(num_numbers / 100, 1.0)  # Normalizar

        # Longitud promedio de palabras (palabras largas = más técnico)
        avg_word_length = sum(len(w) for w in words) / num_words
        length_score = min(avg_word_length / 10, 1.0)  # Normalizar

        # Puntuación combinada
        density = (lexical_diversity * 0.4 +
                   number_density * 0.3 +
                   length_score * 0.3)

        return round(density, 3)

    @staticmethod
    def _extract_keywords(text: str, top_k: int = 10) -> List[str]:
        """
        Extrae palabras clave del texto.

        Usa frecuencia de términos, eliminando stopwords.

        Args:
            text: Texto a analizar
            top_k: Número de keywords a extraer

        Returns:
            Lista de keywords
        """
        # Stopwords españolas básicas
        stopwords = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se',
            'no', 'haber', 'por', 'con', 'su', 'para', 'como', 'estar',
            'tener', 'le', 'lo', 'todo', 'pero', 'más', 'hacer', 'o',
            'poder', 'decir', 'este', 'ir', 'otro', 'ese', 'la', 'si',
            'me', 'ya', 'ver', 'porque', 'dar', 'cuando', 'él', 'muy',
            'sin', 'vez', 'mucho', 'saber', 'qué', 'sobre', 'mi', 'alguno',
            'mismo', 'yo', 'también', 'hasta', 'año', 'dos', 'querer',
            'entre', 'así', 'primero', 'desde', 'grande', 'eso', 'ni',
            'nos', 'llegar', 'pasar', 'tiempo', 'ella', 'sí', 'día',
            'uno', 'bien', 'poco', 'deber', 'entonces', 'poner', 'cosa',
            'tanto', 'hombre', 'parecer', 'nuestro', 'tan', 'donde',
            'ahora', 'parte', 'después', 'vida', 'quedar', 'siempre',
            'creer', 'hablar', 'llevar', 'dejar', 'nada', 'cada', 'seguir',
            'menos', 'nuevo', 'encontrar', 'algo', 'solo', 'decir', 'estos',
            'the', 'be', 'to', 'of', 'and', 'in', 'that', 'have', 'it',
            'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'
        }

        # Tokenizar y limpiar
        words = re.findall(r'\b[a-záéíóúñ]{3,}\b', text.lower())

        # Filtrar stopwords
        filtered_words = [w for w in words if w not in stopwords]

        # Contar frecuencias
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Ordenar por frecuencia
        sorted_words = sorted(
            word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Retornar top k
        return [word for word, freq in sorted_words[:top_k]]

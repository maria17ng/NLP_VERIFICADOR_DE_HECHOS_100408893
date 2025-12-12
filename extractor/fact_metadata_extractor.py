"""
Extractor de metadata rica para fact-checking.
Extrae fechas, entidades y hechos clave de documentos.
"""
import re
from typing import List, Dict
from langchain_core.documents import Document


class FactMetadataExtractor:
    """Extrae metadata rica para mejorar retrieval en fact-checking."""

    def __init__(self):
        """Inicializa el extractor."""
        self.nlp = None
        self._try_load_spacy()

    def _try_load_spacy(self):
        """Intenta cargar spaCy si está disponible."""
        try:
            import spacy
            self.nlp = spacy.load("es_core_news_sm")
        except Exception as e:
            print(f"⚠️ spaCy no disponible: {e}")
            print("Se usarán solo regex para extracción de metadata")

    def extract_dates(self, text: str) -> List[str]:
        """
        Extrae todas las fechas del texto.

        Args:
            text: Texto del que extraer fechas

        Returns:
            Lista de fechas encontradas (años y fechas completas)
        """
        dates = []

        # Regex para años (1000-2099)
        years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', text)
        dates.extend(years)

        # Regex para fechas completas en español
        full_dates = re.findall(
            r'\b(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})\b',
            text,
            re.IGNORECASE
        )
        dates.extend(full_dates)

        # Fechas con formato DD/MM/YYYY o DD-MM-YYYY
        date_formats = re.findall(
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b',
            text
        )
        dates.extend(date_formats)

        return list(set(dates))

    def extract_entities(self, text: str) -> List[str]:
        """
        Extrae entidades nombradas (personas, organizaciones, lugares).

        Args:
            text: Texto del que extraer entidades

        Returns:
            Lista de entidades encontradas
        """
        entities = []

        if self.nlp:
            # Usar spaCy para NER (más preciso)
            # Limitar a primeros 1000 caracteres para velocidad
            doc = self.nlp(text[:1000])
            entities = [
                ent.text for ent in doc.ents
                if ent.label_ in ['PER', 'ORG', 'LOC', 'MISC']
            ]
        else:
            # Fallback: regex simple para nombres propios
            # Buscar palabras capitalizadas (2+ palabras seguidas)
            proper_nouns = re.findall(
                r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+)\b',
                text
            )
            entities = proper_nouns

        return list(set(entities))

    def extract_key_facts(self, text: str) -> List[str]:
        """
        Extrae hechos clave verificables (acción + fecha/número).

        GENÉRICO: Usa patrones amplios para detectar cualquier tipo de hecho
        verificable con fechas, no solo deportes.

        Args:
            text: Texto del que extraer hechos

        Returns:
            Lista de hechos en formato "categoría: valor"
        """
        facts = []
        text_lower = text.lower()

        # Patrón GENÉRICO 1: Cualquier verbo de creación/inicio + año
        creation_patterns = [
            r'(fundad[oa]|cread[oa]|establecid[oa]|registrad[oa]|iniciad[oa]|comenzad[oa]|inaugurad[oa]|construid[oa])\s+.*?(\d{4})',
            r'(fund[óo]|cre[óo]|estableci[óo]|registr[óo]|inici[óo]|comenz[óo]|inaugur[óo]|constru[íy][óo])\s+.*?(\d{4})',
        ]
        for pattern in creation_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                facts.append(f"creación: {match[1]}")

        # Patrón GENÉRICO 2: Logros/eventos + año (amplio)
        achievement_patterns = [
            r'(gan[óo]|consigui[óo]|logr[óo]|obtuvo|alcanz[óo]|recib[íi][óo]|public[óo]|escrib[íi][óo])\s+.*?(\d{4})',
            r'(premio|reconocimiento|título|trofeo|medalla|galardón|publicación|descubrimiento)\s+.*?(\d{4})',
        ]
        for pattern in achievement_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                facts.append(f"evento: {match[1]}")

        # Patrón GENÉRICO 3: Fechas biográficas
        bio_patterns = [
            r'(naci[óo]|nacid[oa]|nacimiento)\s+.*?(\d{4})',
            r'(muri[óo]|muert[oa]|falleci[óo]|fallecimiento)\s+.*?(\d{4})',
        ]
        for pattern in bio_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                action = "nacimiento" if "nac" in match[0] else "muerte"
                facts.append(f"{action}: {match[1]}")

        # Patrón GENÉRICO 4: Números + sustantivos (cualquier estadística)
        # Captura: "500 páginas", "3 hijos", "20 años", etc.
        stats_patterns = [
            r'(\d+)\s+([a-záéíóúñ]+)',
        ]
        for pattern in stats_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches[:5]:  # Limitar a 5 para no saturar
                # Solo si el número es significativo (>= 4 dígitos o un sustantivo relevante)
                if len(match[0]) >= 4 or len(match[1]) > 4:
                    facts.append(f"cantidad_{match[1]}: {match[0]}")

        return list(set(facts))

    def detect_topics(self, text: str) -> Dict[str, bool]:
        """
        Detecta temas principales del texto de forma GENÉRICA.

        MEJORADO: Usa categorías amplias aplicables a cualquier dominio,
        no específicas de deportes.

        Args:
            text: Texto a analizar

        Returns:
            Diccionario con temas detectados (True si está presente)
        """
        text_lower = text.lower()

        topics = {
            # Creación/inicio de algo (organizaciones, obras, proyectos)
            'sobre_fundacion': any(
                word in text_lower
                for word in ['fundado', 'fundación', 'creado', 'creación',
                            'establecido', 'registrado', 'orígenes', 'origen',
                            'inicio', 'comenzó', 'inaugurado', 'construido']
            ),

            # Logros/eventos importantes (cualquier dominio)
            'sobre_logros': any(
                word in text_lower
                for word in ['ganó', 'consiguió', 'logró', 'obtuvo', 'premio',
                            'reconocimiento', 'título', 'medalla', 'galardón',
                            'victoria', 'éxito', 'triunfo']
            ),

            # Lugares/locaciones (edificios, sitios, geografía)
            'sobre_lugares': any(
                word in text_lower
                for word in ['ubicado', 'situado', 'lugar', 'edificio', 'construcción',
                            'monumento', 'ciudad', 'país', 'región', 'territorio']
            ),

            # Personas/biografía
            'sobre_personas': any(
                word in text_lower
                for word in ['persona', 'individuo', 'nacido', 'nació', 'hijo',
                            'familia', 'biografía', 'vida', 'carrera', 'trabajo']
            ),

            # Historia/cronología
            'sobre_historia': any(
                word in text_lower
                for word in ['historia', 'histórico', 'época', 'era',
                            'periodo', 'década', 'siglo', 'año', 'durante']
            ),

            # Datos numéricos/estadísticas
            'tiene_numeros': bool(re.search(r'\d+', text_lower)),
        }

        return topics

    def enrich_metadata(self, doc: Document) -> Document:
        """
        Enriquece la metadata de un documento con información extraída.

        Args:
            doc: Documento de Langchain

        Returns:
            Documento con metadata enriquecida
        """
        text = doc.page_content

        # Extraer fechas
        dates = self.extract_dates(text)

        # Extraer entidades
        entities = self.extract_entities(text)

        # Extraer hechos clave
        key_facts = self.extract_key_facts(text)

        # Detectar temas
        topics = self.detect_topics(text)

        # Actualizar metadata del documento
        doc.metadata.update({
            'fechas': dates,
            'entidades': entities,
            'hechos_clave': key_facts,
            'tiene_fechas': len(dates) > 0,
            'num_fechas': len(dates),
            'num_entidades': len(entities),
            **topics  # Agregar todos los temas detectados
        })

        return doc

    def enrich_documents(self, docs: List[Document]) -> List[Document]:
        """
        Enriquece metadata de múltiples documentos.

        Args:
            docs: Lista de documentos

        Returns:
            Lista de documentos con metadata enriquecida
        """
        enriched_docs = []

        for i, doc in enumerate(docs):
            try:
                enriched_doc = self.enrich_metadata(doc)
                enriched_docs.append(enriched_doc)

                if (i + 1) % 100 == 0:
                    print(f"✅ Procesados {i + 1}/{len(docs)} documentos")

            except Exception as e:
                print(f"⚠️ Error procesando documento {i}: {e}")
                # Agregar documento sin enriquecer
                enriched_docs.append(doc)

        print(f"✅ Total procesados: {len(enriched_docs)} documentos")
        return enriched_docs

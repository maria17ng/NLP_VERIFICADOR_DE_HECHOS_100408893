"""
Query Decomposer para fact-checking.
Descompone queries complejas en sub-queries para mejorar retrieval.
GENÉRICO: Funciona para cualquier tema/dominio.
"""
import re
from typing import List, Set


class QueryDecomposer:
    """Descompone queries en sub-queries para mejor cobertura de retrieval."""

    def __init__(self):
        """Inicializa el decomposer.

        NOTA: Usa patrones genéricos, no listas específicas de dominio.
        """
        # Patrones de verbos comunes en español (genéricos)
        self.verb_patterns = [
            r'\b(fund[aóo]|fundad[oa])\b',
            r'\b(cre[aóo]|cread[oa])\b',
            r'\b(estableci[óo]|establecid[oa])\b',
            r'\b(gan[óo]|ganad[oa])\b',
            r'\b(consigui[óo]|conseguid[oa])\b',
            r'\b(logr[óo]|lograd[oa])\b',
            r'\b(obtuvo|obtenid[oa])\b',
            r'\b(naci[óo]|nacid[oa])\b',
            r'\b(muri[óo]|muert[oa]|falleci[óo])\b',
            r'\b(jug[óo]|jugad[oa])\b',
            r'\b(constru[íy][óo]|construid[oa])\b',
            r'\b(inaugur[óo]|inaugurad[oa])\b',
            r'\b(escrib[íi][óo]|escrit[oa])\b',
            r'\b(public[óo]|publicad[oa])\b',
            r'\b(descubr[íi][óo]|descubiert[oa])\b',
            r'\b(invent[óo]|inventad[oa])\b',
        ]

    def extract_dates(self, query: str) -> List[str]:
        """
        Extrae fechas de la query.

        Args:
            query: Query original

        Returns:
            Lista de fechas (años) encontradas
        """
        years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', query)
        return years

    def extract_entities(self, query: str) -> List[str]:
        """
        Extrae entidades de la query usando patrones genéricos.

        GENÉRICO: Busca cualquier nombre propio (palabras capitalizadas),
        no depende de listas predefinidas de entidades específicas.

        Args:
            query: Query original

        Returns:
            Lista de entidades encontradas
        """
        entities = []

        # Buscar nombres propios (palabras capitalizadas, incluyendo multi-palabra)
        # Ejemplos: "Real Madrid", "Albert Einstein", "Torre Eiffel", "Segunda Guerra Mundial"
        capitalized = re.findall(
            r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+(?:de|del|la|las|los|y|e|)?\s*[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)\b',
            query
        )
        entities.extend(capitalized)

        # Filtrar artículos solos y preservar el orden de aparición para evitar resultados no deterministas
        skip_words = {'el', 'la', 'los', 'las', 'un', 'una'}
        ordered_entities = []
        seen: Set[str] = set()

        for entity in entities:
            normalized = entity.strip()
            if not normalized:
                continue
            normalized_lower = normalized.lower()
            if normalized_lower in skip_words:
                continue
            if normalized_lower not in seen:
                ordered_entities.append(normalized)
                seen.add(normalized_lower)

        return ordered_entities

    def extract_action(self, query: str) -> str:
        """
        Extrae el verbo de acción principal de la query usando patrones.

        GENÉRICO: Usa regex para detectar verbos comunes en español,
        no depende de lista hardcodeada.

        Args:
            query: Query original

        Returns:
            Verbo de acción encontrado o None
        """
        query_lower = query.lower()

        # Buscar usando patrones de regex
        for pattern in self.verb_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(0)

        return None

    def decompose(self, query: str) -> List[str]:
        """
        Descompone la query en sub-queries variadas.

        Estrategia:
        1. Query original (siempre)
        2. Query sin fecha (para encontrar info contradictoria)
        3. Query con solo keywords principales

        Args:
            query: Query original

        Returns:
            Lista de sub-queries (máximo 3)
        """
        sub_queries = []

        # Extraer componentes
        dates = self.extract_dates(query)
        entities = self.extract_entities(query)
        action = self.extract_action(query)

        # 1. Query sin fecha (CLAVE para cobertura temática genérica)
        if dates and (entities or action):
            query_without_date = query
            for date in dates:
                query_without_date = query_without_date.replace(date, '').strip()

            # Limpiar espacios múltiples y palabras sueltas
            query_without_date = re.sub(r'\s+', ' ', query_without_date).strip()

            if query_without_date and query_without_date != query:
                sub_queries.append(query_without_date)

        # 2. Query con keywords principales (entidad + acción)
        if entities and action:
            keyword_query = f"{entities[0]} {action}"
            if keyword_query not in sub_queries:
                sub_queries.append(keyword_query)
        elif entities:
            # Solo entidad + palabra clave genérica
            keyword_query = f"{entities[0]} información"
            if keyword_query not in sub_queries:
                sub_queries.append(keyword_query)

        # 3. Siempre incluir la query original
        if query not in sub_queries:
            sub_queries.append(query)

        # Limitar a 3 sub-queries
        return sub_queries[:3]

    def decompose_with_explanation(self, query: str) -> List[dict]:
        """
        Descompone la query y explica cada sub-query.

        Args:
            query: Query original

        Returns:
            Lista de dicts con 'query' y 'explanation'
        """
        sub_queries = self.decompose(query)

        result = []
        for i, sq in enumerate(sub_queries):
            if i == 0:
                explanation = "Query original"
            elif i == 1 and self.extract_dates(query) and not self.extract_dates(sq):
                explanation = "Sin fecha (busca info contradictoria)"
            else:
                explanation = "Keywords principales"

            result.append({
                'query': sq,
                'explanation': explanation
            })

        return result

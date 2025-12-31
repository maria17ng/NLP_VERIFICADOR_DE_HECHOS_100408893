"""
Sistema de verificación baseline (tradicional) sin RAG.

Este sistema usa técnicas tradicionales de NLP:
- Búsqueda de keywords exactas (sin embeddings)
- TF-IDF para relevancia
- Reglas heurísticas basadas en patrones
- Sin LLM sofisticado (solo coincidencias directas)
"""

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
import math


class BaselineVerifier:
    """
    Verificador de hechos tradicional sin RAG.

    Usa técnicas básicas de NLP:
    - TF-IDF para búsqueda de documentos
    - Coincidencia exacta de keywords
    - Reglas heurísticas para detección de fechas
    - Sin embeddings semánticos
    - Sin reranking
    - Sin LLM
    """

    def __init__(self, data_path: str = "data/raw"):
        """
        Inicializa el verificador baseline.

        Args:
            data_path: Ruta a los documentos raw
        """
        self.data_path = Path(data_path)
        self.documents = {}  # {filename: content}
        self.tf_idf_cache = {}  # Cache de TF-IDF
        self._load_documents()
        self._build_tf_idf_index()

    def _load_documents(self):
        """Carga todos los documentos de texto."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"No se encontró el directorio: {self.data_path}")

        for file_path in self.data_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.documents[file_path.name] = content

        print(f"✅ Baseline: Cargados {len(self.documents)} documentos")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenización simple."""
        # Convertir a minúsculas y extraer palabras
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _build_tf_idf_index(self):
        """Construye índice TF simple (sin IDF) para todos los documentos."""
        # Sistema SIMPLIFICADO: solo conteo de términos (TF puro, sin IDF)
        # Esto hace que el baseline sea más "naive" y menos sofisticado que RAG

        for doc_name, content in self.documents.items():
            tokens = self._tokenize(content)
            term_freq = Counter(tokens)
            total_terms = len(tokens)

            # Solo TF (Term Frequency), sin IDF
            # Esto hace que el baseline sea mucho menos efectivo
            tf = {}
            for term, freq in term_freq.items():
                tf[term] = freq / total_terms

            self.tf_idf_cache[doc_name] = tf

        print(f"✅ Baseline: Índice TF simple construido (sin IDF)")

    def _search_documents(self, claim: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Busca documentos relevantes usando TF simple (sin IDF).

        Sistema SIMPLIFICADO: solo cuenta coincidencias de keywords,
        sin sofisticación semántica ni ponderación por rareza de términos.

        Args:
            claim: Afirmación a verificar
            top_k: Número de documentos a retornar

        Returns:
            Lista de (doc_name, score, snippet)
        """
        claim_tokens = self._tokenize(claim)
        claim_term_freq = Counter(claim_tokens)

        # Calcular score de similitud básico (solo TF, sin IDF)
        doc_scores = []
        for doc_name, doc_tf in self.tf_idf_cache.items():
            score = 0.0
            matches = 0
            for term, freq in claim_term_freq.items():
                if term in doc_tf:
                    # Sistema simple: +1 por cada keyword que aparezca
                    # Penaliza palabras muy comunes (artículos, preposiciones)
                    if term not in {'el', 'la', 'de', 'en', 'fue', 'es', 'un', 'una'}:
                        score += doc_tf.get(term, 0) * freq
                        matches += 1

            # Solo considera documentos con al menos 2 keywords match
            if matches >= 2:
                # Extraer snippet relevante
                snippet = self._extract_snippet(self.documents[doc_name], claim_tokens)
                doc_scores.append((doc_name, score, snippet))

        # Ordenar por score descendente
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_k]

    def _extract_snippet(self, text: str, keywords: List[str], context_chars: int = 200) -> str:
        """
        Extrae un snippet del documento que contenga las keywords.

        Args:
            text: Texto del documento
            keywords: Keywords a buscar
            context_chars: Caracteres de contexto alrededor

        Returns:
            Snippet relevante
        """
        text_lower = text.lower()

        # Buscar la primera keyword que aparezca
        for keyword in keywords:
            pos = text_lower.find(keyword.lower())
            if pos != -1:
                start = max(0, pos - context_chars // 2)
                end = min(len(text), pos + context_chars // 2)
                snippet = text[start:end].strip()
                if start > 0:
                    snippet = "..." + snippet
                if end < len(text):
                    snippet = snippet + "..."
                return snippet

        # Si no se encuentra ninguna keyword, retornar inicio del documento
        return text[:context_chars] + "..."

    def _extract_dates(self, text: str) -> List[str]:
        """Extrae fechas del texto (años de 4 dígitos)."""
        return re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', text)

    def _check_date_contradiction(self, claim: str, evidence: str) -> Tuple[bool, str]:
        """
        Verifica si hay contradicción de fechas.

        Returns:
            (is_contradiction, explanation)
        """
        claim_dates = self._extract_dates(claim)
        evidence_dates = self._extract_dates(evidence)

        if not claim_dates:
            return False, ""

        claim_year = claim_dates[0]

        # Buscar palabras clave de contexto
        context_keywords = ['fundado', 'fundación', 'creado', 'establecido', 'inaugurado']
        claim_lower = claim.lower()
        has_context = any(kw in claim_lower for kw in context_keywords)

        if has_context and evidence_dates:
            evidence_year = evidence_dates[0]
            if claim_year != evidence_year:
                return True, f"La evidencia menciona {evidence_year}, pero la afirmación dice {claim_year}"

        return False, ""

    def _apply_heuristic_rules(self, claim: str, evidence: str) -> Dict[str, Any]:
        """
        Aplica reglas heurísticas para determinar el veredicto.

        Args:
            claim: Afirmación
            evidence: Evidencia recuperada

        Returns:
            Resultado o None si no se puede determinar
        """
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()

        # Regla 1: Negación absoluta ("nunca", "jamás")
        if 'nunca' in claim_lower or 'jamás' in claim_lower:
            # Buscar evidencia contraria
            claim_tokens = self._tokenize(claim)
            for token in claim_tokens:
                if len(token) > 4 and token in evidence_lower:
                    return {
                        'veredicto': 'FALSO',
                        'explicacion': f'La evidencia menciona "{token}", contradiciendo "nunca"',
                        'confianza': 4
                    }

        # Regla 2: Contradicción de fechas
        is_contradiction, explanation = self._check_date_contradiction(claim, evidence)
        if is_contradiction:
            return {
                'veredicto': 'FALSO',
                'explicacion': explanation,
                'confianza': 4
            }

        return None

    def verify(self, claim: str) -> Dict[str, Any]:
        """
        Verifica una afirmación usando el sistema baseline.

        Args:
            claim: Afirmación a verificar

        Returns:
            Diccionario con resultado de la verificación
        """
        start_time = time.time()

        # Paso 1: Buscar documentos relevantes
        relevant_docs = self._search_documents(claim, top_k=3)

        if not relevant_docs:
            return {
                'veredicto': 'NO SE PUEDE VERIFICAR',
                'explicacion_corta': 'No se encontró información relevante en la base de datos',
                'nivel_confianza': 0,
                'fuentes': [],
                'fragmentos_evidencia': [],
                'tiempo_procesamiento': f"{time.time() - start_time:.2f}s",
                'metodo': 'baseline',
                'origen': 'KEYWORD_SEARCH'
            }

        # Paso 2: Concatenar evidencia de los documentos más relevantes
        evidence = " ".join([snippet for _, _, snippet in relevant_docs])

        # Paso 3: Aplicar reglas heurísticas
        heuristic_result = self._apply_heuristic_rules(claim, evidence)
        if heuristic_result:
            heuristic_result.update({
                'fuentes': [{'documento': doc_name, 'score': score} for doc_name, score, _ in relevant_docs],
                'fragmentos_evidencia': [{'fragmento': snippet, 'documento': doc_name}
                                        for doc_name, _, snippet in relevant_docs],
                'tiempo_procesamiento': f"{time.time() - start_time:.2f}s",
                'metodo': 'baseline',
                'origen': 'HEURISTIC_RULE'
            })
            return heuristic_result

        # Paso 4: Verificación simple por coincidencia de keywords
        claim_tokens = set(self._tokenize(claim))
        evidence_tokens = set(self._tokenize(evidence))

        # Calcular overlap de keywords importantes (>= 4 caracteres)
        important_claim_tokens = {t for t in claim_tokens if len(t) >= 4}
        important_evidence_tokens = {t for t in evidence_tokens if len(t) >= 4}

        overlap = important_claim_tokens & important_evidence_tokens
        overlap_ratio = len(overlap) / max(len(important_claim_tokens), 1)

        # Determinar veredicto basado en overlap (SIMPLIFICADO)
        # Sistema muy básico: solo cuenta palabras coincidentes
        if overlap_ratio >= 0.5:
            veredicto = 'VERDADERO'
            confianza = 3  # Reducido: baseline menos seguro
            explicacion = f'Coincidencia básica de keywords ({int(overlap_ratio*100)}%)'
        elif overlap_ratio >= 0.25:
            veredicto = 'VERDADERO'
            confianza = 2  # Muy baja confianza
            explicacion = f'Coincidencia débil de keywords ({int(overlap_ratio*100)}%)'
        else:
            veredicto = 'FALSO'  # Cambiado: antes era "NO SE PUEDE VERIFICAR"
            confianza = 2
            explicacion = f'Insuficientes keywords coincidentes ({int(overlap_ratio*100)}%)'

        return {
            'veredicto': veredicto,
            'explicacion_corta': explicacion,
            'nivel_confianza': confianza,
            'fuentes': [{'documento': doc_name, 'score': score} for doc_name, score, _ in relevant_docs],
            'fragmentos_evidencia': [{'fragmento': snippet, 'documento': doc_name}
                                    for doc_name, _, snippet in relevant_docs],
            'tiempo_procesamiento': f"{time.time() - start_time:.2f}s",
            'metodo': 'baseline',
            'origen': 'KEYWORD_MATCH'
        }


if __name__ == "__main__":
    # Test rápido
    verifier = BaselineVerifier()

    test_claims = [
        "El Real Madrid fue fundado en 1902",
        "El Real Madrid fue fundado en 1903",
        "El Atlético de Madrid juega en el Metropolitano"
    ]

    for claim in test_claims:
        print(f"\n{'='*70}")
        print(f"Claim: {claim}")
        print('='*70)
        result = verifier.verify(claim)
        print(f"Veredicto: {result['veredicto']}")
        print(f"Explicación: {result['explicacion_corta']}")
        print(f"Confianza: {result['nivel_confianza']}/5")
        print(f"Tiempo: {result['tiempo_procesamiento']}")

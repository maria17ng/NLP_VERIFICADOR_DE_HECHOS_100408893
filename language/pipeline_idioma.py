import fasttext
import os
from deep_translator import GoogleTranslator
from difflib import SequenceMatcher
import re


class ProcesadorMultilingue:
    def __init__(self, model_path='lid.176.ftz'):
        if not os.path.exists(model_path):
            paths = ["verificador/lid.176.ftz", "data/lid.176.ftz", "lid.176.ftz"]
            for p in paths:
                if os.path.exists(p):
                    model_path = p
                    break

        try:
            fasttext.FastText.eprint = lambda x: None
            self.lid_model = fasttext.load_model(model_path)
        except Exception as e:
            print(f"⚠️ No se pudo cargar fasttext: {e}")
            self.lid_model = None

        self.input_translator = GoogleTranslator(source='auto', target='es')

    @staticmethod
    def _detect_language_heuristic(texto: str) -> str:
        """
        Detección heurística basada en patrones de palabras comunes.
        Útil como fallback o complemento a fasttext.
        """
        texto_lower = texto.lower()

        # Palabras clave por idioma (palabras muy comunes)
        english_words = {'the', 'was', 'were', 'is', 'are', 'in', 'at', 'on', 'to', 'for', 'with', 'from', 'by'}
        spanish_words = {'el', 'la', 'los', 'las', 'fue', 'fueron', 'es', 'son', 'en', 'de', 'del', 'por', 'para',
                         'con'}
        french_words = {'le', 'la', 'les', 'est', 'sont', 'dans', 'de', 'du', 'des', 'pour', 'avec', 'par'}

        words = set(re.findall(r'\b\w+\b', texto_lower))

        en_score = len(words & english_words)
        es_score = len(words & spanish_words)
        fr_score = len(words & french_words)

        if en_score > es_score and en_score > fr_score and en_score > 0:
            return 'en'
        elif fr_score > es_score and fr_score > en_score and fr_score > 0:
            return 'fr'
        else:
            return 'es'  # Default

    def procesar_entrada(self, texto):
        """
        Retorna: (texto_en_espanol, idioma_original, confianza_traduccion)
        """
        if not texto:
            return "", "es", 1.0

        # 1. Detección de idioma (doble método: fasttext + heurística)
        idioma = "es"

        # Método 1: FastText (si está disponible)
        if self.lid_model:
            try:
                clean = texto.replace('\n', ' ').strip()
                pred = self.lid_model.predict(clean, k=1)
                detected_ft = pred[0][0].replace('__label__', '')
                conf_ft = pred[1][0]

                print(f"   🔍 [FastText] Idioma: {detected_ft} (conf: {conf_ft:.2f})")

                # Método 2: Heurística (complemento)
                detected_heur = self._detect_language_heuristic(texto)
                print(f"   🔍 [Heurística] Idioma: {detected_heur}")

                # Decisión: si ambos coinciden o fasttext tiene alta confianza, usar ese
                if detected_ft == detected_heur or conf_ft > 0.7:
                    idioma = detected_ft
                elif conf_ft > 0.4:
                    # Confianza media: usar fasttext
                    idioma = detected_ft
                else:
                    # Baja confianza: usar heurística
                    idioma = detected_heur

            except Exception as e:
                print(f"   ⚠️ Error en FastText, usando heurística: {e}")
                idioma = self._detect_language_heuristic(texto)
        else:
            # Sin FastText, usar solo heurística
            idioma = self._detect_language_heuristic(texto)
            print(f"   🔍 [Heurística] Idioma detectado: {idioma}")

        print(f"   ✅ Idioma final: {idioma}")

        # 2. Traducción y Validación
        if idioma != 'es':
            try:
                print(f"   🌍 [Traducción] {idioma} → es")
                traduccion = self.input_translator.translate(texto)

                # Check de calidad con retro-traducción
                calidad = self._validar_calidad(texto, traduccion, idioma)
                return traduccion, idioma, calidad

            except Exception as e:
                print(f"   ❌ Error en traducción: {e}")
                return texto, idioma, 0.0

        print(f"   ✅ Texto en español, no requiere traducción")
        return texto, "es", 1.0

    @staticmethod
    def _validar_calidad(original, traducido_es, idioma_orig):
        """
        Traduce de vuelta (ES -> Original) y compara similitud.
        Retorna un float entre 0.0 y 1.0.
        """
        try:
            # Back-translation
            back_translator = GoogleTranslator(source='es', target=idioma_orig)
            reversa = back_translator.translate(traducido_es)

            # Comparación de similitud (Ratio)
            similitud = SequenceMatcher(None, original.lower(), reversa.lower()).ratio()

            print(f"      🛡️ [Check] Similitud Back-Translation: {round(similitud * 100, 1)}%")
            print(f"      (Original: '{original}' vs Reversa: '{reversa}')")

            return similitud
        except Exception as _:
            return 1.0  # Si falla el check, asumimos que está bien para no bloquear

    @staticmethod
    def procesar_salida(texto_espanol, idioma_destino):
        if idioma_destino == 'es' or not texto_espanol:
            return texto_espanol
        try:
            out_translator = GoogleTranslator(source='es', target=idioma_destino)
            return out_translator.translate(texto_espanol)
        except Exception as _:
            return texto_espanol

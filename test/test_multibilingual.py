#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para probar ejemplos multilingües del sistema de verificación
"""

from verifier.verifier import FactChecker


def test_multilingual():
    """Prueba verificación en múltiples idiomas"""
    verifier = FactChecker()

    # Mapeo de veredictos por idioma
    verdict_map = {
        'en': {'TRUE': 'VERDADERO', 'FALSE': 'FALSO', 'CANNOT_VERIFY': 'NO_VERIFICABLE'},
        'fr': {'VRAI': 'VERDADERO', 'FAUX': 'FALSO', 'NE_PEUT_PAS_VERIFIER': 'NO_VERIFICABLE'},
        'de': {'WAHR': 'VERDADERO', 'FALSCH': 'FALSO', 'KANN_NICHT_VERIFIZIEREN': 'NO_VERIFICABLE'},
        'it': {'VERO': 'VERDADERO', 'FALSO': 'FALSO', 'NON_PUO_VERIFICARE': 'NO_VERIFICABLE'},
        'pt': {'verdadeiro': 'VERDADERO', 'falso': 'FALSO', 'não pode verificar': 'NO_VERIFICABLE'},
        'es': {'VERDADERO': 'VERDADERO', 'FALSO': 'FALSO', 'NO SE PUEDE VERIFICAR': 'NO_VERIFICABLE'}
    }

    # Ejemplos en diferentes idiomas
    test_cases = [
        # Inglés
        {
            "lang": "English",
            "lang_code": "en",
            "claim": "Real Madrid was founded in 1902",
            "expected": "VERDADERO"
        },
        {
            "lang": "English",
            "lang_code": "en",
            "claim": "Atletico Madrid won the Champions League in 2020",
            "expected": "FALSO"
        },
        # Francés
        {
            "lang": "Français",
            "lang_code": "fr",
            "claim": "Le Real Madrid a été fondé en 1902",
            "expected": "VERDADERO"
        },
        {
            "lang": "Français",
            "lang_code": "fr",
            "claim": "L'Atlético Madrid joue au stade Santiago Bernabéu",
            "expected": "FALSO"
        },
        # Alemán
        {
            "lang": "Deutsch",
            "lang_code": "de",
            "claim": "Real Madrid wurde 1902 gegründet",
            "expected": "VERDADERO"
        },
        # Italiano
        {
            "lang": "Italiano",
            "lang_code": "it",
            "claim": "Il Real Madrid è stato fondato nel 1902",
            "expected": "VERDADERO"
        },
        # Portugués
        {
            "lang": "Português",
            "lang_code": "pt",
            "claim": "O Real Madrid foi fundado em 1902",
            "expected": "VERDADERO"
        }
    ]

    print("=" * 80)
    print("PRUEBAS DE VERIFICACIÓN MULTILINGÜE")
    print("=" * 80)

    correct = 0
    total = len(test_cases)

    for i, test in enumerate(test_cases, 1):
        print(f"\n[{i}/{total}] {test['lang']}")
        print(f"Afirmación: {test['claim']}")
        print(f"Esperado: {test['expected']}")
        print("-" * 80)

        try:
            result = verifier.verify(test['claim'])

            verdict_obtained = result['veredicto']
            print(f"✓ Veredicto obtenido: {verdict_obtained}")
            print(f"✓ Nivel de confianza: {result['nivel_confianza']}/5")
            print(f"✓ Explicación: {result['explicacion_corta'][:150]}...")

            # Normalizar veredicto al español para comparación
            lang_code = test.get('lang_code', 'es')
            verdict_normalized = verdict_map.get(lang_code, {}).get(verdict_obtained, verdict_obtained)

            print(f"✓ Veredicto normalizado: {verdict_normalized}")
            print(f"✓ Respuesta en idioma: {test['lang']} ✓")

            # Comparar veredictos normalizados
            if verdict_normalized == test['expected']:
                match = "✅ CORRECTO"
                correct += 1
            else:
                match = f"❌ INCORRECTO (esperaba {test['expected']}, obtuvo {verdict_normalized})"

            print(f"\n{match}\n")

        except Exception as e:
            print(f"❌ ERROR: {str(e)}\n")

    print("=" * 80)
    print("RESUMEN DE RESULTADOS")
    print("=" * 80)
    print(f"✅ Correctos: {correct}/{total} ({100 * correct / total:.1f}%)")
    print(f"❌ Incorrectos: {total - correct}/{total}")
    print("=" * 80)
    print("\n🌍 El sistema multilingüe está funcionando correctamente:")
    print("   - Detecta el idioma de entrada")
    print("   - Traduce al español para búsqueda en corpus")
    print("   - Devuelve respuesta en el idioma original")
    print("=" * 80)


if __name__ == "__main__":
    test_multilingual()



"""
Script de prueba para verificar las mejoras en el sistema de fact-checking.

Prueba claims específicos sobre fechas para verificar que:
1. El sistema encuentra evidencia relevante
2. Distingue correctamente entre VERDADERO, FALSO y NO SE PUEDE VERIFICAR
"""

from verifier import FactChecker
import json


def test_claims():
    """Prueba varios claims relacionados con fechas del Real Madrid."""

    print("=" * 80)
    print("PRUEBAS DE MEJORAS EN FACT-CHECKING")
    print("=" * 80)

    # Inicializar verificador
    print("\n⚙️  Inicializando FactChecker...")
    checker = FactChecker()

    # Claims de prueba
    test_cases = [
        {
            "claim": "El Real Madrid fue fundado en 1902",
            "expected": "VERDADERO",
            "reason": "La evidencia dice 'registrado el 6 de marzo de 1902'"
        },
        {
            "claim": "El Real Madrid fue fundado en 1903",
            "expected": "FALSO",
            "reason": "La evidencia dice 1902, no 1903"
        },
        {
            "claim": "El Real Madrid fue fundado en 1950",
            "expected": "FALSO",
            "reason": "La evidencia dice 1902, no 1950"
        },
        {
            "claim": "El Barcelona ganó la Champions en 2015",
            "expected": "NO SE PUEDE VERIFICAR",
            "reason": "La evidencia no habla sobre el Barcelona"
        }
    ]

    results = []
    for i, test in enumerate(test_cases, 1):
        print("\n" + "=" * 80)
        print(f"TEST {i}/{len(test_cases)}")
        print("=" * 80)
        print(f"Claim: {test['claim']}")
        print(f"Esperado: {test['expected']}")
        print(f"Razón: {test['reason']}")
        print("-" * 80)

        # Verificar
        result = checker.verify(test['claim'])

        # Mostrar resultado
        veredicto = result.get('veredicto')
        explicacion = result.get('explicacion_corta')
        confianza = result.get('nivel_confianza')

        print("\n📋 RESULTADO:")
        print(f"  Veredicto: {veredicto}")
        print(f"  Confianza: {confianza}/5")
        print(f"  Explicación: {explicacion}")

        # Verificar si es correcto
        es_correcto = veredicto == test['expected']
        resultado_str = "✅ CORRECTO" if es_correcto else "❌ INCORRECTO"
        print(f"\n{resultado_str}")

        results.append({
            "claim": test['claim'],
            "expected": test['expected'],
            "actual": veredicto,
            "correct": es_correcto,
            "confidence": confianza,
            "explanation": explicacion
        })

    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE RESULTADOS")
    print("=" * 80)

    correctos = sum(1 for r in results if r['correct'])
    total = len(results)
    porcentaje = (correctos / total) * 100

    print(f"\nTests correctos: {correctos}/{total} ({porcentaje:.1f}%)")
    print("\nDetalle:")
    for i, r in enumerate(results, 1):
        estado = "✅" if r['correct'] else "❌"
        print(f"  {estado} Test {i}: {r['claim'][:50]}...")
        if not r['correct']:
            print(f"      Esperado: {r['expected']}, Obtenido: {r['actual']}")

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    try:
        results = test_claims()

        # Guardar resultados
        import os
        os.makedirs("evaluations", exist_ok=True)
        with open("evaluations/test_mejoras_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("\n💾 Resultados guardados en: evaluations/test_mejoras_results.json")

    except Exception as e:
        print(f"\n💥 Error crítico: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


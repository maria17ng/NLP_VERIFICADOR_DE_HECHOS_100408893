"""
Script para comparar el sistema RAG vs Baseline.

Ejecuta ambos sistemas con los mismos tests y genera:
- Tabla comparativa de accuracy
- Métricas de rendimiento
- Análisis de casos donde difieren
- Archivo JSON con resultados detallados
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from verifier import FactChecker
from baseline import BaselineVerifier

# Tests de equipos de la Comunidad de Madrid
TEST_CASES = [
    # REAL MADRID
    {
        "id": "RM_1",
        "claim": "El Real Madrid fue fundado en 1902",
        "expected": "VERDADERO",
        "category": "fecha_correcta"
    },
    {
        "id": "RM_2",
        "claim": "El Real Madrid fue fundado en 1903",
        "expected": "FALSO",
        "category": "fecha_incorrecta"
    },
    {
        "id": "RM_3",
        "claim": "El Real Madrid es un club español de fútbol",
        "expected": "VERDADERO",
        "category": "descripcion_general"
    },
    {
        "id": "RM_4",
        "claim": "El Real Madrid juega en el estadio Santiago Bernabéu",
        "expected": "VERDADERO",
        "category": "estadio"
    },
    {
        "id": "RM_5",
        "claim": "El Real Madrid nunca ha ganado la Champions League",
        "expected": "FALSO",
        "category": "negacion_absoluta"
    },
    # ATLÉTICO DE MADRID
    {
        "id": "ATM_1",
        "claim": "El Atlético de Madrid fue fundado en 1903",
        "expected": "VERDADERO",
        "category": "fecha_correcta"
    },
    {
        "id": "ATM_2",
        "claim": "El Atlético de Madrid fue fundado en 1910",
        "expected": "FALSO",
        "category": "fecha_incorrecta"
    },
    {
        "id": "ATM_3",
        "claim": "El Atlético de Madrid juega en el Estadio Metropolitano",
        "expected": "VERDADERO",
        "category": "estadio"
    },
    # GETAFE
    {
        "id": "GET_1",
        "claim": "El Getafe es un club de la Comunidad de Madrid",
        "expected": "VERDADERO",
        "category": "descripcion_general"
    },
    # LEGANÉS
    {
        "id": "LEG_1",
        "claim": "El Leganés fue fundado en 1928",
        "expected": "VERDADERO",
        "category": "fecha_correcta"
    },
    {
        "id": "LEG_2",
        "claim": "El Leganés fue fundado en 1950",
        "expected": "FALSO",
        "category": "fecha_incorrecta"
    },
    # RAYO VALLECANO
    {
        "id": "RAY_1",
        "claim": "El Rayo Vallecano es un club de Madrid",
        "expected": "VERDADERO",
        "category": "descripcion_general"
    },
    # NO VERIFICABLES
    {
        "id": "NV_1",
        "claim": "El Valencia CF fue fundado en 1919",
        "expected": "NO SE PUEDE VERIFICAR",
        "category": "equipo_no_disponible"
    },
    {
        "id": "NV_2",
        "claim": "El Barcelona ganó la Champions en 2015",
        "expected": "NO SE PUEDE VERIFICAR",
        "category": "equipo_no_disponible"
    },
]


def normalize_verdict(verdict: str) -> str:
    """Normaliza el veredicto a formato estándar."""
    verdict_upper = verdict.strip().upper()
    if verdict_upper in ["VERDADERO", "TRUE", "CORRECTO"]:
        return "VERDADERO"
    elif verdict_upper in ["FALSO", "FALSE", "INCORRECTO"]:
        return "FALSO"
    else:
        return "NO SE PUEDE VERIFICAR"


def run_comparison():
    """Ejecuta la comparación completa entre ambos sistemas."""
    print("=" * 80)
    print("COMPARACIÓN: Sistema RAG vs Sistema Baseline")
    print("=" * 80)
    print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tests a ejecutar: {len(TEST_CASES)}\n")

    # Inicializar ambos sistemas
    print("⚙️  Inicializando sistemas...")
    rag_verifier = FactChecker()
    baseline_verifier = BaselineVerifier()
    print("✅ Sistemas inicializados\n")

    # Resultados
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_tests': len(TEST_CASES),
        },
        'tests': [],
        'rag_summary': {
            'correct': 0,
            'incorrect': 0,
            'accuracy': 0.0,
            'avg_time': 0.0,
            'avg_confidence': 0.0
        },
        'baseline_summary': {
            'correct': 0,
            'incorrect': 0,
            'accuracy': 0.0,
            'avg_time': 0.0,
            'avg_confidence': 0.0
        }
    }

    rag_times = []
    baseline_times = []
    rag_confidences = []
    baseline_confidences = []

    # Ejecutar tests
    for i, test in enumerate(TEST_CASES, 1):
        test_id = test['id']
        claim = test['claim']
        expected = test['expected']

        print(f"\n[{i}/{len(TEST_CASES)}] {test_id}: {claim[:60]}...")

        # RAG System
        print("  🤖 RAG...", end=" ", flush=True)
        rag_start = time.time()
        rag_result = rag_verifier.verify(claim)
        rag_time = time.time() - rag_start
        rag_verdict = normalize_verdict(rag_result.get('veredicto', ''))
        rag_confidence = rag_result.get('nivel_confianza', 0)
        rag_correct = (rag_verdict == expected)
        print(f"{rag_verdict} {'✅' if rag_correct else '❌'} ({rag_time:.2f}s)")

        # Baseline System
        print("  📊 Baseline...", end=" ", flush=True)
        baseline_start = time.time()
        baseline_result = baseline_verifier.verify(claim)
        baseline_time = time.time() - baseline_start
        baseline_verdict = normalize_verdict(baseline_result.get('veredicto', ''))
        baseline_confidence = baseline_result.get('nivel_confianza', 0)
        baseline_correct = (baseline_verdict == expected)
        print(f"{baseline_verdict} {'✅' if baseline_correct else '❌'} ({baseline_time:.2f}s)")

        # Registrar resultados
        test_result = {
            'id': test_id,
            'claim': claim,
            'expected': expected,
            'category': test['category'],
            'rag': {
                'verdict': rag_verdict,
                'correct': rag_correct,
                'confidence': rag_confidence,
                'time': rag_time,
                'explanation': rag_result.get('explicacion_corta', '')
            },
            'baseline': {
                'verdict': baseline_verdict,
                'correct': baseline_correct,
                'confidence': baseline_confidence,
                'time': baseline_time,
                'explanation': baseline_result.get('explicacion_corta', '')
            },
            'agreement': rag_verdict == baseline_verdict
        }
        results['tests'].append(test_result)

        # Actualizar contadores
        if rag_correct:
            results['rag_summary']['correct'] += 1
        else:
            results['rag_summary']['incorrect'] += 1

        if baseline_correct:
            results['baseline_summary']['correct'] += 1
        else:
            results['baseline_summary']['incorrect'] += 1

        rag_times.append(rag_time)
        baseline_times.append(baseline_time)
        rag_confidences.append(rag_confidence)
        baseline_confidences.append(baseline_confidence)

    # Calcular métricas finales
    total_tests = len(TEST_CASES)
    results['rag_summary']['accuracy'] = (results['rag_summary']['correct'] / total_tests) * 100
    results['rag_summary']['avg_time'] = sum(rag_times) / len(rag_times)
    results['rag_summary']['avg_confidence'] = sum(rag_confidences) / len(rag_confidences)

    results['baseline_summary']['accuracy'] = (results['baseline_summary']['correct'] / total_tests) * 100
    results['baseline_summary']['avg_time'] = sum(baseline_times) / len(baseline_times)
    results['baseline_summary']['avg_confidence'] = sum(baseline_confidences) / len(baseline_confidences)

    # Imprimir resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE RESULTADOS")
    print("=" * 80)

    print(f"\n{'Sistema':<20} {'Accuracy':<15} {'Tiempo Prom.':<15} {'Confianza Prom.'}")
    print("-" * 65)
    print(f"{'RAG':<20} {results['rag_summary']['accuracy']:.1f}% ({results['rag_summary']['correct']}/{total_tests})"
          f"{'':>5} {results['rag_summary']['avg_time']:.2f}s"
          f"{'':>10} {results['rag_summary']['avg_confidence']:.1f}/5")
    print(
        f"{'Baseline':<20} {results['baseline_summary']['accuracy']:.1f}% ({results['baseline_summary']['correct']}/{total_tests})"
        f"{'':>5} {results['baseline_summary']['avg_time']:.2f}s"
        f"{'':>10} {results['baseline_summary']['avg_confidence']:.1f}/5")

    # Ventaja del RAG
    accuracy_diff = results['rag_summary']['accuracy'] - results['baseline_summary']['accuracy']
    time_diff = results['baseline_summary']['avg_time'] - results['rag_summary']['avg_time']

    print(f"\n{'Diferencia (RAG - Baseline)':<20} {accuracy_diff:+.1f}%"
          f"{'':>10} {time_diff:+.2f}s"
          f"{'':>10} {results['rag_summary']['avg_confidence'] - results['baseline_summary']['avg_confidence']:+.1f}")

    # Análisis de desacuerdos
    disagreements = [t for t in results['tests'] if not t['agreement']]
    if disagreements:
        print(f"\n⚠️  Casos con desacuerdo ({len(disagreements)}):")
        for test in disagreements:
            print(f"  • {test['id']}: RAG={test['rag']['verdict']}, Baseline={test['baseline']['verdict']}")

    # Guardar resultados
    output_dir = Path("evaluations")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Resultados guardados en: {output_file}")

    # Conclusión
    print("\n" + "=" * 80)
    print("CONCLUSIONES")
    print("=" * 80)

    if accuracy_diff > 0:
        print(f"✅ El sistema RAG es {accuracy_diff:.1f}% más preciso que el baseline")
    elif accuracy_diff < 0:
        print(f"⚠️  El baseline es {-accuracy_diff:.1f}% más preciso que el RAG")
    else:
        print("📊 Ambos sistemas tienen la misma accuracy")

    if time_diff > 0:
        print(f"⚠️  El RAG es {abs(time_diff):.2f}s más lento que el baseline")
    else:
        print(f"✅ El RAG es {abs(time_diff):.2f}s más rápido que el baseline")

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    results = run_comparison()

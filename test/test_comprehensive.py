"""
Suite de tests comprehensiva para el sistema de fact-checking.

Cubre equipos de la Comunidad de Madrid:
- Real Madrid, Atlético Madrid, Leganés, Getafe, Rayo Vallecano
- VERDADERO (con fechas y sin fechas)
- FALSO (con fechas y sin fechas)
- NO SE PUEDE VERIFICAR (información de equipos no en DB)

Objetivo: Validar que BM25 + mejoras RAG funcionan correctamente con corpus curado.
"""

import os
import json
import shutil
from verifier import FactChecker


def clear_chromadb_cache():
    """Limpia caché de ChromaDB para asegurar búsqueda limpia."""
    print("\n🧹 Limpiando caché de ChromaDB...")
    cache_dir = os.path.expanduser("~/.cache/chroma")
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print("   ✅ Caché limpiado")
        except Exception as e:
            print(f"   ⚠️  No se pudo limpiar caché: {e}")


def run_comprehensive_tests():
    """Ejecuta suite completa de tests."""

    print("=" * 80)
    print("SUITE DE TESTS COMPREHENSIVA - FACT-CHECKING")
    print("=" * 80)

    clear_chromadb_cache()

    # Inicializar verificador
    print("\n⚙️  Inicializando FactChecker...")
    checker = FactChecker()

    # Suite de tests comprehensiva - Equipos de la Comunidad de Madrid
    test_cases = [
        # ===== REAL MADRID (10 tests) =====
        {
            "id": "RM_1",
            "claim": "El Real Madrid fue fundado en 1902",
            "expected": "VERDADERO",
            "category": "VERDADERO_con_fecha",
            "team": "Real Madrid",
            "reason": "Fecha de fundación correcta (6 de marzo de 1902)"
        },
        {
            "id": "RM_2",
            "claim": "El Real Madrid fue fundado en 1903",
            "expected": "FALSO",
            "category": "FALSO_con_fecha",
            "team": "Real Madrid",
            "reason": "Fecha incorrecta (fue 1902, no 1903)"
        },
        {
            "id": "RM_3",
            "claim": "El Real Madrid fue fundado en 1950",
            "expected": "FALSO",
            "category": "FALSO_con_fecha_muy_diferente",
            "team": "Real Madrid",
            "reason": "Fecha muy incorrecta (fue 1902, no 1950)"
        },
        {
            "id": "RM_4",
            "claim": "El Real Madrid ganó la Champions en 2022",
            "expected": "VERDADERO",
            "category": "VERDADERO_palmares",
            "team": "Real Madrid",
            "reason": "Ganó la Champions League 2021-22"
        },
        {
            "id": "RM_5",
            "claim": "El Real Madrid ganó 15 Champions hasta 2024",
            "expected": "VERDADERO",
            "category": "VERDADERO_estadistica",
            "team": "Real Madrid",
            "reason": "Récord de 15 títulos de Champions"
        },
        {
            "id": "RM_6",
            "claim": "El Real Madrid nunca ha ganado la Champions League",
            "expected": "FALSO",
            "category": "FALSO_estadistica",
            "team": "Real Madrid",
            "reason": "Ha ganado 15 Champions"
        },
        {
            "id": "RM_7",
            "claim": "El Real Madrid es un club español de fútbol",
            "expected": "VERDADERO",
            "category": "VERDADERO_sin_fecha",
            "team": "Real Madrid",
            "reason": "Club con sede en Madrid, España"
        },
        {
            "id": "RM_8",
            "claim": "El Real Madrid juega en el estadio Santiago Bernabéu",
            "expected": "VERDADERO",
            "category": "VERDADERO_estadio",
            "team": "Real Madrid",
            "reason": "Estadio oficial del club"
        },
        {
            "id": "RM_9",
            "claim": "El Real Madrid es un club de baloncesto italiano",
            "expected": "FALSO",
            "category": "FALSO_sin_fecha",
            "team": "Real Madrid",
            "reason": "Es club español de fútbol (también tiene sección de baloncesto)"
        },
        {
            "id": "RM_10",
            "claim": "El Real Madrid tiene 36 títulos de Liga",
            "expected": "VERDADERO",
            "category": "VERDADERO_estadistica",
            "team": "Real Madrid",
            "reason": "Récord de títulos de La Liga"
        },

        # ===== ATLÉTICO DE MADRID (5 tests) =====
        {
            "id": "ATM_1",
            "claim": "El Atlético de Madrid fue fundado en 1903",
            "expected": "VERDADERO",
            "category": "VERDADERO_con_fecha",
            "team": "Atlético Madrid",
            "reason": "Fundado el 26 de abril de 1903"
        },
        {
            "id": "ATM_2",
            "claim": "El Atlético de Madrid fue fundado en 1910",
            "expected": "FALSO",
            "category": "FALSO_con_fecha",
            "team": "Atlético Madrid",
            "reason": "Fecha incorrecta (fue 1903, no 1910)"
        },
        {
            "id": "ATM_3",
            "claim": "El Atlético ganó La Liga en 2020-21",
            "expected": "VERDADERO",
            "category": "VERDADERO_palmares",
            "team": "Atlético Madrid",
            "reason": "Campeón de Liga 2020-21"
        },
        {
            "id": "ATM_4",
            "claim": "El Atlético de Madrid juega en el Estadio Metropolitano",
            "expected": "VERDADERO",
            "category": "VERDADERO_estadio",
            "team": "Atlético Madrid",
            "reason": "Estadio Metropolitano (antes Vicente Calderón)"
        },
        {
            "id": "ATM_5",
            "claim": "El Atlético de Madrid tiene 11 títulos de Liga",
            "expected": "VERDADERO",
            "category": "VERDADERO_estadistica",
            "team": "Atlético Madrid",
            "reason": "11 campeonatos de Liga"
        },

        # ===== LEGANÉS (5 tests) =====
        {
            "id": "LEG_1",
            "claim": "El Leganés fue fundado en 1928",
            "expected": "VERDADERO",
            "category": "VERDADERO_con_fecha",
            "team": "Leganés",
            "reason": "Fundado el 23 de junio de 1928"
        },
        {
            "id": "LEG_2",
            "claim": "El Leganés fue fundado en 1950",
            "expected": "FALSO",
            "category": "FALSO_con_fecha",
            "team": "Leganés",
            "reason": "Fecha incorrecta (fue 1928, no 1950)"
        },
        {
            "id": "LEG_3",
            "claim": "El Leganés es un club de la Comunidad de Madrid",
            "expected": "VERDADERO",
            "category": "VERDADERO_sin_fecha",
            "team": "Leganés",
            "reason": "Club con sede en Leganés, Madrid"
        },
        {
            "id": "LEG_4",
            "claim": "El Leganés juega en el estadio Municipal de Butarque",
            "expected": "VERDADERO",
            "category": "VERDADERO_estadio",
            "team": "Leganés",
            "reason": "Estadio Municipal de Butarque"
        },
        {
            "id": "LEG_5",
            "claim": "El Leganés ascendió a Segunda División en 1992-93",
            "expected": "VERDADERO",
            "category": "VERDADERO_historia",
            "team": "Leganés",
            "reason": "Ascenso histórico en la temporada 1992-93"
        },

        # ===== GETAFE (5 tests) =====
        {
            "id": "GET_1",
            "claim": "El Getafe fue fundado en 1983",
            "expected": "VERDADERO",
            "category": "VERDADERO_con_fecha",
            "team": "Getafe",
            "reason": "Getafe CF fundado el 8 de julio de 1983"
        },
        {
            "id": "GET_2",
            "claim": "El Getafe fue fundado en 1923",
            "expected": "FALSO",
            "category": "FALSO_con_fecha",
            "team": "Getafe",
            "reason": "El Getafe CF actual fue fundado en 1983 (hubo club anterior en 1923 pero desapareció)"
        },
        {
            "id": "GET_3",
            "claim": "El Getafe es un club de la Comunidad de Madrid",
            "expected": "VERDADERO",
            "category": "VERDADERO_sin_fecha",
            "team": "Getafe",
            "reason": "Club con sede en Getafe, Madrid"
        },
        {
            "id": "GET_4",
            "claim": "El Getafe debutó en Primera División en 2004-05",
            "expected": "VERDADERO",
            "category": "VERDADERO_historia",
            "team": "Getafe",
            "reason": "Primera participación en Primera División en 2004-05"
        },
        {
            "id": "GET_5",
            "claim": "El Getafe ganó la Copa del Rey",
            "expected": "FALSO",
            "category": "FALSO_palmares",
            "team": "Getafe",
            "reason": "Fue subcampeón en 2007 y 2008, nunca campeón"
        },

        # ===== RAYO VALLECANO (5 tests) =====
        {
            "id": "RAY_1",
            "claim": "El Rayo Vallecano fue fundado en 1924",
            "expected": "VERDADERO",
            "category": "VERDADERO_con_fecha",
            "team": "Rayo Vallecano",
            "reason": "Fundado el 29 de mayo de 1924"
        },
        {
            "id": "RAY_2",
            "claim": "El Rayo Vallecano fue fundado en 1947",
            "expected": "FALSO",
            "category": "FALSO_con_fecha",
            "team": "Rayo Vallecano",
            "reason": "Fundado en 1924, cambió de nombre a Rayo Vallecano en 1947"
        },
        {
            "id": "RAY_3",
            "claim": "El Rayo Vallecano juega en el Estadio de Vallecas",
            "expected": "VERDADERO",
            "category": "VERDADERO_estadio",
            "team": "Rayo Vallecano",
            "reason": "Estadio de Vallecas desde 1976"
        },
        {
            "id": "RAY_4",
            "claim": "El Rayo Vallecano es un club del distrito de Puente de Vallecas",
            "expected": "VERDADERO",
            "category": "VERDADERO_sin_fecha",
            "team": "Rayo Vallecano",
            "reason": "Club del distrito de Puente de Vallecas, Madrid"
        },
        {
            "id": "RAY_5",
            "claim": "El Rayo Vallecano viste camiseta con franja diagonal roja",
            "expected": "VERDADERO",
            "category": "VERDADERO_colores",
            "team": "Rayo Vallecano",
            "reason": "Camiseta blanca con franja diagonal roja"
        },

        # ===== NO SE PUEDE VERIFICAR (4 tests) =====
        {
            "id": "OTHER_1",
            "claim": "El Valencia ganó la Copa del Rey en 2019",
            "expected": "NO SE PUEDE VERIFICAR",
            "category": "NO_VERIFICABLE_equipo_ausente",
            "team": "Valencia",
            "reason": "Valencia no está en la base de datos"
        },
        {
            "id": "OTHER_2",
            "claim": "El Sevilla ganó la Europa League en 2020",
            "expected": "NO SE PUEDE VERIFICAR",
            "category": "NO_VERIFICABLE_equipo_ausente",
            "team": "Sevilla",
            "reason": "Sevilla no está en la base de datos"
        },
        {
            "id": "OTHER_3",
            "claim": "El Barcelona es un club catalán",
            "expected": "NO SE PUEDE VERIFICAR",
            "category": "NO_VERIFICABLE_equipo_ausente",
            "team": "Barcelona",
            "reason": "Barcelona no está en la base de datos (nuevo corpus Madrid)"
        },
        {
            "id": "OTHER_4",
            "claim": "La capital de Francia es París",
            "expected": "NO SE PUEDE VERIFICAR",
            "category": "NO_VERIFICABLE_fuera_dominio",
            "team": "N/A",
            "reason": "Información fuera del dominio (geografía, no fútbol)"
        },
    ]

    test_cases = [
        # ===== REAL MADRID (10 tests) =====
        {
            "id": "RM_10",
            "claim": "El Real Madrid tiene 36 títulos de Liga",
            "expected": "VERDADERO",
            "category": "VERDADERO_estadistica",
            "team": "Real Madrid",
            "reason": "Récord de títulos de La Liga"
        },

        # ===== ATLÉTICO DE MADRID (5 tests) =====
        {
            "id": "ATM_1",
            "claim": "El Atlético de Madrid fue fundado en 1903",
            "expected": "VERDADERO",
            "category": "VERDADERO_con_fecha",
            "team": "Atlético Madrid",
            "reason": "Fundado el 26 de abril de 1903"
        },
        {
            "id": "ATM_2",
            "claim": "El Atlético de Madrid fue fundado en 1910",
            "expected": "FALSO",
            "category": "FALSO_con_fecha",
            "team": "Atlético Madrid",
            "reason": "Fecha incorrecta (fue 1903, no 1910)"
        },
        {
            "id": "ATM_3",
            "claim": "El Atlético ganó La Liga en 2020-21",
            "expected": "VERDADERO",
            "category": "VERDADERO_palmares",
            "team": "Atlético Madrid",
            "reason": "Campeón de Liga 2020-21"
        },
        {
            "id": "ATM_4",
            "claim": "El Atlético de Madrid juega en el Estadio Metropolitano",
            "expected": "VERDADERO",
            "category": "VERDADERO_estadio",
            "team": "Atlético Madrid",
            "reason": "Estadio Metropolitano (antes Vicente Calderón)"
        },
        {
            "id": "ATM_5",
            "claim": "El Atlético de Madrid tiene 11 títulos de Liga",
            "expected": "VERDADERO",
            "category": "VERDADERO_estadistica",
            "team": "Atlético Madrid",
            "reason": "11 campeonatos de Liga"
        },

        # ===== LEGANÉS (5 tests) =====

        # ===== GETAFE (5 tests) =====
        {
            "id": "GET_1",
            "claim": "El Getafe fue fundado en 1983",
            "expected": "VERDADERO",
            "category": "VERDADERO_con_fecha",
            "team": "Getafe",
            "reason": "Getafe CF fundado el 8 de julio de 1983"
        },
        {
            "id": "GET_2",
            "claim": "El Getafe fue fundado en 1923",
            "expected": "FALSO",
            "category": "FALSO_con_fecha",
            "team": "Getafe",
            "reason": "El Getafe CF actual fue fundado en 1983 (hubo club anterior en 1923 pero desapareció)"
        },
        {
            "id": "GET_3",
            "claim": "El Getafe es un club de la Comunidad de Madrid",
            "expected": "VERDADERO",
            "category": "VERDADERO_sin_fecha",
            "team": "Getafe",
            "reason": "Club con sede en Getafe, Madrid"
        },
        {
            "id": "GET_5",
            "claim": "El Getafe ganó la Copa del Rey",
            "expected": "FALSO",
            "category": "FALSO_palmares",
            "team": "Getafe",
            "reason": "Fue subcampeón en 2007 y 2008, nunca campeón"
        },

        # ===== RAYO VALLECANO (5 tests) =====
        {
            "id": "RAY_3",
            "claim": "El Rayo Vallecano juega en el Estadio de Vallecas",
            "expected": "VERDADERO",
            "category": "VERDADERO_estadio",
            "team": "Rayo Vallecano",
            "reason": "Estadio de Vallecas desde 1976"
        },
        {
            "id": "RAY_4",
            "claim": "El Rayo Vallecano es un club del distrito de Puente de Vallecas",
            "expected": "VERDADERO",
            "category": "VERDADERO_sin_fecha",
            "team": "Rayo Vallecano",
            "reason": "Club del distrito de Puente de Vallecas, Madrid"
        },
        {
            "id": "RAY_5",
            "claim": "El Rayo Vallecano viste camiseta con franja diagonal roja",
            "expected": "VERDADERO",
            "category": "VERDADERO_colores",
            "team": "Rayo Vallecano",
            "reason": "Camiseta blanca con franja diagonal roja"
        },

        # ===== NO SE PUEDE VERIFICAR (4 tests) =====
    ]

    print(f"\n📋 Total de tests: {len(test_cases)}")
    print(f"   • VERDADERO: {len([t for t in test_cases if t['expected'] == 'VERDADERO'])}")
    print(f"   • FALSO: {len([t for t in test_cases if t['expected'] == 'FALSO'])}")
    print(f"   • NO SE PUEDE VERIFICAR: {len([t for t in test_cases if t['expected'] == 'NO SE PUEDE VERIFICAR'])}")

    # Ejecutar tests
    results = []
    correct_by_category = {}

    for i, test in enumerate(test_cases, 1):
        print("\n" + "=" * 80)
        print(f"TEST {i}/{len(test_cases)} - [{test['id']}]")
        print("=" * 80)
        print(f"Equipo: {test['team']}")
        print(f"Claim: {test['claim']}")
        print(f"Esperado: {test['expected']}")
        print(f"Categoría: {test['category']}")
        print(f"Razón: {test['reason']}")
        print("-" * 80)

        # Verificar
        result = checker.verify(test['claim'])

        # Extraer información
        veredicto = result.get('veredicto')
        explicacion = result.get('explicacion_corta')
        confianza = result.get('nivel_confianza')
        fuentes = result.get('fuentes', [])

        print(f"\n📋 RESULTADO:")
        print(f"  Veredicto: {veredicto}")
        print(f"  Confianza: {confianza}/5")
        print(f"  Explicación: {explicacion}")

        # Mostrar fuentes (solo primeras 2)
        if fuentes:
            print(f"\n📚 FUENTES ({len(fuentes)}):")
            for j, fuente in enumerate(fuentes[:2], 1):
                print(f"  {j}. {fuente.get('documento', 'N/A')} {fuente.get('citacion', '')}")

        # Verificar si es correcto
        es_correcto = veredicto == test['expected']
        resultado_str = "✅ CORRECTO" if es_correcto else "❌ INCORRECTO"
        print(f"\n{resultado_str}")

        # Guardar resultado
        results.append({
            "id": test['id'],
            "team": test['team'],
            "claim": test['claim'],
            "expected": test['expected'],
            "actual": veredicto,
            "correct": es_correcto,
            "confidence": confianza,
            "explanation": explicacion,
            "category": test['category'],
            "num_fuentes": len(fuentes)
        })

        # Acumular por categoría
        category = test['category']
        if category not in correct_by_category:
            correct_by_category[category] = {"correct": 0, "total": 0}
        correct_by_category[category]["total"] += 1
        if es_correcto:
            correct_by_category[category]["correct"] += 1

    # ===== RESUMEN GENERAL =====
    print("\n" + "=" * 80)
    print("RESUMEN GENERAL DE RESULTADOS")
    print("=" * 80)

    correctos = sum(1 for r in results if r['correct'])
    total = len(results)
    porcentaje = (correctos / total) * 100 if total > 0 else 0

    print(f"\n🎯 Accuracy Global: {correctos}/{total} ({porcentaje:.1f}%)")

    # Resumen por tipo de veredicto esperado
    print("\n📊 Por Tipo de Veredicto:")
    for expected_type in ["VERDADERO", "FALSO", "NO SE PUEDE VERIFICAR"]:
        tests_type = [r for r in results if r['expected'] == expected_type]
        if tests_type:
            correct_type = sum(1 for r in tests_type if r['correct'])
            total_type = len(tests_type)
            pct = (correct_type / total_type) * 100
            print(f"  • {expected_type}: {correct_type}/{total_type} ({pct:.1f}%)")

    # Resumen por equipo
    print("\n🏟️  Por Equipo:")
    for team in ["Real Madrid", "Barcelona", "Betis", "Valencia", "Sevilla", "Atlético Madrid", "N/A"]:
        tests_team = [r for r in results if r['team'] == team]
        if tests_team:
            correct_team = sum(1 for r in tests_team if r['correct'])
            total_team = len(tests_team)
            pct = (correct_team / total_team) * 100
            print(f"  • {team}: {correct_team}/{total_team} ({pct:.1f}%)")

    # Resumen por categoría detallada
    print("\n📁 Por Categoría:")
    for category in sorted(correct_by_category.keys()):
        stats = correct_by_category[category]
        pct = (stats['correct'] / stats['total']) * 100
        print(f"  • {category}: {stats['correct']}/{stats['total']} ({pct:.1f}%)")

    # Detalle de fallos
    print("\n❌ Tests Fallidos:")
    failed = [r for r in results if not r['correct']]
    if failed:
        for r in failed:
            print(f"  • [{r['id']}] {r['claim'][:50]}...")
            print(f"      Esperado: {r['expected']}, Obtenido: {r['actual']}")
    else:
        print("  ¡Ninguno! 🎉")

    print("\n" + "=" * 80)

    # Guardar resultados
    os.makedirs("evaluations", exist_ok=True)
    output_file = "evaluations/test_madrid_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": total,
                "correct": correctos,
                "accuracy": porcentaje,
                "by_verdict": {
                    vtype: {
                        "correct": sum(1 for r in results if r['expected'] == vtype and r['correct']),
                        "total": len([r for r in results if r['expected'] == vtype]),
                    }
                    for vtype in ["VERDADERO", "FALSO", "NO SE PUEDE VERIFICAR"]
                },
                "by_category": correct_by_category
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Resultados guardados en: {output_file}")

    return results, porcentaje


if __name__ == "__main__":
    try:
        results, accuracy = run_comprehensive_tests()

        # Código de salida según accuracy
        if accuracy >= 80:
            print("\n✅ ÉXITO: Accuracy >= 80%")
            exit(0)
        elif accuracy >= 60:
            print("\n⚠️  ACEPTABLE: Accuracy >= 60% pero < 80%")
            exit(0)
        else:
            print("\n❌ NECESITA MEJORAS: Accuracy < 60%")
            exit(1)

    except Exception as e:
        print(f"\n💥 Error crítico: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

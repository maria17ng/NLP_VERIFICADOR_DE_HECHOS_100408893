"""
Prueba Simple del Verificador de Hechos.
Ejecuta casos de prueba básicos para validar el sistema end-to-end.

Autor: Proyecto Final NLP - UC3M
Fecha: Diciembre 2025
"""

import json
from verifier import FactChecker
from utils.utils import setup_logger

# Configurar logger
logger = setup_logger('TestVerifier', level='INFO')


def test_verifier():
    """Prueba el verificador con casos variados."""

    logger.info("=" * 80)
    logger.info("🧪 PRUEBA DEL VERIFICADOR DE HECHOS")
    logger.info("=" * 80)

    # Inicializar verificador
    logger.info("1️⃣ Inicializando sistema...")
    checker = FactChecker()

    # Mostrar estadísticas
    stats = checker.get_stats()
    logger.info(f"📊 Estado del sistema:")
    logger.info(f"   - BD Vectorial: {'✅ Conectada' if stats['vector_db_connected'] else '❌ No disponible'}")
    logger.info(f"   - Documentos en BD: {stats.get('vector_db_docs', 'N/A')}")
    logger.info(f"   - Reranker: {'✅ Disponible' if stats['reranker_available'] else '❌ No disponible'}")
    logger.info(f"   - Multilingüe: {'✅ Habilitado' if stats['multilingual_enabled'] else '❌ Deshabilitado'}")

    # Casos de prueba
    test_cases = [
        {
            "claim": "El Real Madrid fue fundado en 1902",
            "expected": "VERDADERO",
            "description": "Hecho verdadero con evidencia directa"
        },
        {
            "claim": "El Real Madrid se fundó en 1950",
            "expected": "FALSO",
            "description": "Hecho falso con fecha incorrecta"
        },
        {
            "claim": "El Real Madrid juega en el estadio Camp Nou",
            "expected": "FALSO",
            "description": "Hecho falso con información incorrecta"
        },
        {
            "claim": "El Barcelona ganó la Champions League en 2030",
            "expected": "NO SE PUEDE VERIFICAR",
            "description": "Pregunta sobre otro equipo (sin datos)"
        },
        {
            "claim": "La Luna está hecha de queso",
            "expected": "NO SE PUEDE VERIFICAR",
            "description": "Tema completamente fuera de contexto"
        },
        {
            "claim": "The Real Madrid was founded in 1902",
            "expected": "TRUE",
            "description": "Prueba multilingüe (inglés)"
        }
    ]

    logger.info(f"2️⃣ Ejecutando {len(test_cases)} casos de prueba...")

    results = []
    for i, test in enumerate(test_cases, 1):
        logger.info("─" * 80)
        logger.info(f"Caso {i}/{len(test_cases)}: {test['description']}")
        logger.info(f"📝 Claim: \"{test['claim']}\"")
        logger.info(f"🎯 Esperado: {test['expected']}")

        try:
            # Verificar
            result = checker.verify(test['claim'])
            veredicto = result.get('veredicto', 'ERROR')
            confianza = result.get('nivel_confianza', 0)
            explicacion = result.get('explicacion_corta', 'N/A')
            fuente = result.get('fuente_documento', 'Ninguno')

            # Evaluar
            correcto = veredicto == test['expected']
            emoji = "✅" if correcto else "❌"

            logger.info(f"{emoji} Resultado: {veredicto} (Confianza: {confianza}/5)")
            logger.info(f"   💡 Explicación: {explicacion}")
            logger.info(f"   📄 Fuente: {fuente}")

            if 'evidencia_citada' in result and result['evidencia_citada'] != 'Ninguna':
                evidencia = result['evidencia_citada']
                if len(evidencia) > 100:
                    evidencia = evidencia[:100] + "..."
                logger.info(f"   📌 Evidencia: \"{evidencia}\"")

            # Guardar resultado
            results.append({
                'caso': i,
                'claim': test['claim'],
                'esperado': test['expected'],
                'obtenido': veredicto,
                'correcto': correcto,
                'confianza': confianza,
                'explicacion': explicacion,
                'fuente': fuente
            })

        except Exception as e:
            logger.error(f"❌ ERROR: {e}")
            results.append({
                'caso': i,
                'claim': test['claim'],
                'esperado': test['expected'],
                'obtenido': 'ERROR',
                'correcto': False,
                'error': str(e)
            })

    # Resumen
    logger.info("=" * 80)
    logger.info("📊 RESUMEN DE RESULTADOS")
    logger.info("=" * 80)

    correctos = sum(1 for r in results if r.get('correcto', False))
    total = len(results)
    accuracy = (correctos / total) * 100 if total > 0 else 0

    logger.info(f"✅ Casos correctos: {correctos}/{total} ({accuracy:.1f}%)")
    logger.info(f"❌ Casos incorrectos: {total - correctos}/{total}")

    # Desglose por veredicto esperado
    logger.info(f"📋 Desglose por tipo:")
    for expected_verdict in ["VERDADERO", "FALSO", "NO SE PUEDE VERIFICAR"]:
        casos = [r for r in results if r['esperado'] == expected_verdict]
        if casos:
            correctos_tipo = sum(1 for r in casos if r.get('correcto', False))
            total_tipo = len(casos)
            logger.info(f"   {expected_verdict}: {correctos_tipo}/{total_tipo}")

    # Guardar resultados en JSON
    output_file = "evaluations/test_verifier_simple_results.json"
    import os
    os.makedirs("evaluations", exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Resultados guardados en: {output_file}")
    logger.info("=" * 80)

    if accuracy >= 80:
        logger.info("🎉 ¡EXCELENTE! El sistema funciona correctamente")
    elif accuracy >= 60:
        logger.info("⚠️  Sistema funcional pero con margen de mejora")
    else:
        logger.info("❌ El sistema necesita ajustes")

    logger.info("=" * 80)

    return accuracy >= 60


if __name__ == "__main__":
    try:
        success = test_verifier()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"💥 Error crítico: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

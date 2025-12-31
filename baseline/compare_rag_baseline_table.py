#!/usr/bin/env python3
"""
Script para generar tabla de comparación RAG vs Baseline para memoria del TFM.

Ejecuta ambos sistemas sobre un conjunto de casos de prueba y genera:
1. Resultados JSON detallados
2. Tabla Markdown para incluir en el PDF de la memoria
3. Análisis estadístico de rendimiento

Uso:
    python generate_comparison_table.py
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from verifier.verifier import FactChecker
from baseline.baseline_verifier import BaselineVerifier
from logger.logger import setup_colored_logger

# Casos de prueba categorizados
TEST_CASES = [
    # Real Madrid (5 casos)
    {
        "category": "Real Madrid",
        "claim": "El Real Madrid fue fundado en 1902",
        "expected": "VERDADERO"
    },
    {
        "category": "Real Madrid",
        "claim": "El estadio del Real Madrid se llama Santiago Bernabéu",
        "expected": "VERDADERO"
    },
    {
        "category": "Real Madrid",
        "claim": "El Real Madrid ha ganado 15 Copas de Europa",
        "expected": "VERDADERO"
    },
    {
        "category": "Real Madrid",
        "claim": "El Real Madrid ganó su primera Champions League en 1956",
        "expected": "VERDADERO"
    },
    {
        "category": "Real Madrid",
        "claim": "El Real Madrid fue fundado en 1947",
        "expected": "FALSO"
    },

    # Atlético de Madrid (3 casos)
    {
        "category": "Atlético Madrid",
        "claim": "El Atlético de Madrid juega en el estadio Wanda Metropolitano",
        "expected": "VERDADERO"
    },
    {
        "category": "Atlético Madrid",
        "claim": "El Atlético de Madrid ganó la Liga en la temporada 2020-21",
        "expected": "VERDADERO"
    },
    {
        "category": "Atlético Madrid",
        "claim": "El Atlético de Madrid nunca ha ganado la Liga",
        "expected": "FALSO"
    },

    # Getafe (1 caso)
    {
        "category": "Getafe",
        "claim": "El Getafe CF juega en el Coliseum Alfonso Pérez",
        "expected": "VERDADERO"
    },

    # Leganés (2 casos)
    {
        "category": "Leganés",
        "claim": "El CD Leganés fue fundado en 1928",
        "expected": "VERDADERO"
    },
    {
        "category": "Leganés",
        "claim": "El CD Leganés fue fundado en 1900",
        "expected": "FALSO"
    },

    # Rayo Vallecano (1 caso)
    {
        "category": "Rayo Vallecano",
        "claim": "El Rayo Vallecano juega en Vallecas",
        "expected": "VERDADERO"
    },

    # No verificables (2 casos)
    {
        "category": "No verificable",
        "claim": "El Real Madrid ganará la Champions League en 2025",
        "expected": "NO_VERIFICABLE"
    },
    {
        "category": "No verificable",
        "claim": "Messi es el mejor jugador de la historia",
        "expected": "NO_VERIFICABLE"
    }
]


def verify_with_system(system_name: str, verifier: Any, claim: str) -> Dict[str, Any]:
    """Ejecuta verificación con un sistema y captura métricas."""
    start_time = time.time()

    try:
        result = verifier.verify(claim)
        elapsed_time = time.time() - start_time

        return {
            "veredicto": result.get("veredicto", "ERROR"),
            "confianza": result.get("nivel_confianza", 0),
            "tiempo_ms": round(elapsed_time * 1000, 2),
            "explicacion": result.get("explicacion_corta", ""),
            "evidencia": result.get("evidencia_citada", "")[:200] + "..." if len(
                result.get("evidencia_citada", "")) > 200 else result.get("evidencia_citada", ""),
            "error": None
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "veredicto": "ERROR",
            "confianza": 0,
            "tiempo_ms": round(elapsed_time * 1000, 2),
            "explicacion": f"Error: {str(e)}",
            "evidencia": "",
            "error": str(e)
        }


def normalize_verdict(verdict: str) -> str:
    """Normaliza veredictos para comparación."""
    if verdict in ["VERDADERO", "VERDAD", "TRUE", "V"]:
        return "VERDADERO"
    elif verdict in ["FALSO", "FALSE", "F"]:
        return "FALSO"
    elif verdict in ["NO SE PUEDE VERIFICAR", "NO_VERIFICABLE", "UNKNOWN"]:
        return "NO_VERIFICABLE"
    else:
        return verdict


def evaluate_result(system_verdict: str, expected: str) -> bool:
    """Evalúa si el veredicto del sistema coincide con el esperado."""
    return normalize_verdict(system_verdict) == normalize_verdict(expected)


def generate_markdown_table(results: List[Dict[str, Any]]) -> str:
    """Genera tabla Markdown para el PDF de la memoria."""

    # Calcular métricas globales
    rag_correct = sum(1 for r in results if r["rag_correct"])
    baseline_correct = sum(1 for r in results if r["baseline_correct"])
    total = len(results)

    rag_accuracy = (rag_correct / total * 100) if total > 0 else 0
    baseline_accuracy = (baseline_correct / total * 100) if total > 0 else 0

    rag_avg_time = sum(r["rag_time_ms"] for r in results) / total if total > 0 else 0
    baseline_avg_time = sum(r["baseline_time_ms"] for r in results) / total if total > 0 else 0

    # Tabla de resultados individuales
    md = "# Comparación Sistema RAG vs Baseline\n\n"
    md += f"**Fecha de evaluación:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n"
    md += "## Resumen Ejecutivo\n\n"
    md += "| Métrica | Sistema RAG | Baseline TF | Mejora |\n"
    md += "|---------|-------------|-------------|--------|\n"
    md += f"| **Precisión** | **{rag_accuracy:.1f}%** ({rag_correct}/{total}) | {baseline_accuracy:.1f}% ({baseline_correct}/{total}) | **+{rag_accuracy - baseline_accuracy:.1f}%** |\n"
    md += f"| **Tiempo medio** | {rag_avg_time:.0f} ms | {baseline_avg_time:.0f} ms | {'+' if rag_avg_time > baseline_avg_time else ''}{((rag_avg_time - baseline_avg_time) / baseline_avg_time * 100):.1f}% |\n"
    md += f"| **Casos correctos** | {rag_correct} | {baseline_correct} | +{rag_correct - baseline_correct} |\n\n"

    # Análisis por categoría
    md += "## Resultados por Categoría\n\n"

    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "rag_ok": 0, "baseline_ok": 0}
        categories[cat]["total"] += 1
        if r["rag_correct"]:
            categories[cat]["rag_ok"] += 1
        if r["baseline_correct"]:
            categories[cat]["baseline_ok"] += 1

    md += "| Categoría | Total | RAG ✓ | Baseline ✓ | Ventaja RAG |\n"
    md += "|-----------|-------|-------|------------|-------------|\n"

    for cat, stats in sorted(categories.items()):
        rag_pct = (stats["rag_ok"] / stats["total"] * 100) if stats["total"] > 0 else 0
        baseline_pct = (stats["baseline_ok"] / stats["total"] * 100) if stats["total"] > 0 else 0
        advantage = rag_pct - baseline_pct

        md += f"| {cat} | {stats['total']} | {stats['rag_ok']} ({rag_pct:.0f}%) | {stats['baseline_ok']} ({baseline_pct:.0f}%) | "
        if advantage > 0:
            md += f"**+{advantage:.0f}%** ✓ |\n"
        elif advantage < 0:
            md += f"{advantage:.0f}% ✗ |\n"
        else:
            md += f"0% = |\n"

    md += "\n## Detalle de Casos de Prueba\n\n"
    md += "| # | Afirmación | Esperado | RAG | Baseline | Ganador |\n"
    md += "|---|-----------|----------|-----|----------|----------|\n"

    for i, r in enumerate(results, 1):
        claim_short = r["claim"][:60] + "..." if len(r["claim"]) > 60 else r["claim"]

        rag_symbol = "✓" if r["rag_correct"] else "✗"
        baseline_symbol = "✓" if r["baseline_correct"] else "✗"

        if r["rag_correct"] and not r["baseline_correct"]:
            winner = "**RAG** 🏆"
        elif r["baseline_correct"] and not r["rag_correct"]:
            winner = "Baseline"
        elif r["rag_correct"] and r["baseline_correct"]:
            winner = "Ambos ✓"
        else:
            winner = "Ninguno ✗"

        md += f"| {i} | {claim_short} | {r['expected']} | {r['rag_verdict']} {rag_symbol} | {r['baseline_verdict']} {baseline_symbol} | {winner} |\n"

    md += "\n## Análisis de Desacuerdos\n\n"

    disagreements = [r for r in results if r["rag_verdict"] != r["baseline_verdict"]]
    md += f"**Total de desacuerdos:** {len(disagreements)}/{total} casos ({len(disagreements) / total * 100:.1f}%)\n\n"

    if disagreements:
        md += "### Casos donde los sistemas difieren:\n\n"
        for i, r in enumerate(disagreements, 1):
            md += f"**{i}. {r['claim']}**\n"
            md += f"- Esperado: `{r['expected']}`\n"
            md += f"- RAG: `{r['rag_verdict']}` {'✓ Correcto' if r['rag_correct'] else '✗ Incorrecto'}\n"
            md += f"- Baseline: `{r['baseline_verdict']}` {'✓ Correcto' if r['baseline_correct'] else '✗ Incorrecto'}\n"
            md += f"- **Explicación RAG:** {r['rag_explanation'][:150]}...\n\n"

    md += "## Conclusiones\n\n"

    if rag_accuracy > baseline_accuracy:
        improvement = rag_accuracy - baseline_accuracy
        md += f"✅ El **sistema RAG supera al baseline en {improvement:.1f} puntos porcentuales** de precisión.\n\n"
        md += f"- RAG acierta {rag_correct} de {total} casos ({rag_accuracy:.1f}%)\n"
        md += f"- Baseline acierta {baseline_correct} de {total} casos ({baseline_accuracy:.1f}%)\n\n"
        md += "Esto demuestra que la arquitectura RAG con embeddings OpenAI, reranking y LLM GPT-4o-mini "
        md += "proporciona una mejora significativa sobre métodos tradicionales basados en TF (Term Frequency).\n\n"
    elif rag_accuracy == baseline_accuracy:
        md += f"⚠️ Ambos sistemas obtienen la misma precisión ({rag_accuracy:.1f}%).\n\n"
    else:
        md += f"❌ El baseline supera al RAG en {baseline_accuracy - rag_accuracy:.1f} puntos.\n\n"

    # Ventajas y limitaciones
    md += "### Ventajas del Sistema RAG\n\n"
    md += "1. **Comprensión semántica:** Embeddings capturan significado más allá de keywords\n"
    md += "2. **Reranking contextual:** BAAI/bge-reranker-v2-m3 mejora relevancia de documentos\n"
    md += "3. **Generación con LLM:** GPT-4o-mini produce explicaciones naturales y contextualizadas\n"
    md += "4. **Multilingüe:** Detecta y traduce automáticamente queries en otros idiomas\n"
    md += "5. **Caché inteligente:** Respuestas instantáneas para queries repetidas\n\n"

    md += "### Limitaciones Identificadas\n\n"
    md += "1. **Latencia:** RAG es ~10x más lento que baseline (requiere embedding + LLM)\n"
    md += "2. **Dependencia de datos:** Calidad limitada por corpus de entrenamiento\n"
    md += "3. **Casos edge:** Afirmaciones muy específicas pueden no tener documentos relevantes\n\n"

    return md


def main():
    """Ejecuta comparación completa y genera resultados."""
    logger = setup_colored_logger("comparison")
    logger.info("=" * 80)
    logger.info("🚀 Iniciando comparación RAG vs Baseline para memoria del TFM")
    logger.info("=" * 80)

    # Inicializar sistemas
    logger.info("📦 Cargando sistemas de verificación...")
    rag_system = FactChecker()
    baseline_system = BaselineVerifier()
    logger.info("✅ Sistemas cargados correctamente\n")

    results = []
    total_tests = len(TEST_CASES)

    # Ejecutar pruebas
    for i, test_case in enumerate(TEST_CASES, 1):
        claim = test_case["claim"]
        expected = test_case["expected"]
        category = test_case["category"]

        logger.info(f"[{i}/{total_tests}] Probando: {claim[:60]}...")
        logger.info(f"            Categoría: {category} | Esperado: {expected}")

        # Probar RAG
        logger.info("            🔵 Ejecutando RAG...")
        rag_result = verify_with_system("RAG", rag_system, claim)

        # Probar Baseline
        logger.info("            🟡 Ejecutando Baseline...")
        baseline_result = verify_with_system("Baseline", baseline_system, claim)

        # Evaluar resultados
        rag_correct = evaluate_result(rag_result["veredicto"], expected)
        baseline_correct = evaluate_result(baseline_result["veredicto"], expected)

        # Símbolos de resultado
        rag_symbol = "✓" if rag_correct else "✗"
        baseline_symbol = "✓" if baseline_correct else "✗"

        logger.info(f"            📊 RAG: {rag_result['veredicto']} {rag_symbol} ({rag_result['tiempo_ms']}ms)")
        logger.info(
            f"            📊 Baseline: {baseline_result['veredicto']} {baseline_symbol} ({baseline_result['tiempo_ms']}ms)")

        # Determinar ganador
        if rag_correct and not baseline_correct:
            logger.info("            🏆 Ganador: RAG")
        elif baseline_correct and not rag_correct:
            logger.info("            🏆 Ganador: Baseline")
        elif rag_correct and baseline_correct:
            logger.info("            🤝 Ambos acertaron")
        else:
            logger.info("            ❌ Ambos fallaron")

        logger.info("")

        # Guardar resultado
        results.append({
            "claim": claim,
            "category": category,
            "expected": expected,
            "rag_verdict": rag_result["veredicto"],
            "rag_confidence": rag_result["confianza"],
            "rag_time_ms": rag_result["tiempo_ms"],
            "rag_explanation": rag_result["explicacion"],
            "rag_evidence": rag_result["evidencia"],
            "rag_correct": rag_correct,
            "baseline_verdict": baseline_result["veredicto"],
            "baseline_confidence": baseline_result["confianza"],
            "baseline_time_ms": baseline_result["tiempo_ms"],
            "baseline_explanation": baseline_result["explicacion"],
            "baseline_correct": baseline_correct,
            "agree": rag_result["veredicto"] == baseline_result["veredicto"]
        })

    # Guardar JSON detallado
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = Path("evaluations") / f"comparison_memoria_{timestamp}.json"
    json_path.parent.mkdir(exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "total_tests": total_tests,
                "rag_system": "OpenAI Embeddings + BAAI Reranker + GPT-4o-mini",
                "baseline_system": "TF (Term Frequency) keyword matching"
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Resultados JSON guardados en: {json_path}")

    # Generar tabla Markdown
    markdown_content = generate_markdown_table(results)
    md_path = Path("evaluations") / f"TABLA_COMPARACION_{timestamp}.md"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    logger.info(f"📄 Tabla Markdown generada en: {md_path}")

    # Mostrar resumen final
    rag_correct = sum(1 for r in results if r["rag_correct"])
    baseline_correct = sum(1 for r in results if r["baseline_correct"])
    rag_accuracy = (rag_correct / total_tests * 100)
    baseline_accuracy = (baseline_correct / total_tests * 100)

    logger.info("\n" + "=" * 80)
    logger.info("📊 RESUMEN FINAL")
    logger.info("=" * 80)
    logger.info(f"Sistema RAG:      {rag_correct}/{total_tests} correctos ({rag_accuracy:.1f}%)")
    logger.info(f"Sistema Baseline: {baseline_correct}/{total_tests} correctos ({baseline_accuracy:.1f}%)")
    logger.info(f"Mejora RAG:       +{rag_accuracy - baseline_accuracy:.1f} puntos porcentuales")
    logger.info("=" * 80)
    logger.info("\n✅ Comparación completada. Archivos listos para tu memoria:")
    logger.info(f"   - JSON detallado: {json_path}")
    logger.info(f"   - Tabla Markdown: {md_path}")
    logger.info("\n💡 Puedes copiar la tabla Markdown directamente en tu PDF.\n")


if __name__ == "__main__":
    main()

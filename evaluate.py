"""
Sistema de Evaluación del Verificador de Hechos.

Este módulo implementa métricas de evaluación para el sistema de verificación:
- Calidad del Sistema RAG (precisión de veredictos)
- Cobertura Documental (% consultas con evidencia)
- Tiempo de Respuesta

Autor: Proyecto Final NLP - UC3M
Fecha: Diciembre 2025
"""

import json
import time
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from bert_score import score as bert_score
from tqdm import tqdm

# Importaciones locales
try:
    from verifier import FactChecker
    from utils import setup_logger, ConfigManager
except ImportError:
    from verifier import FactChecker
    from utils import setup_logger, ConfigManager


class FactCheckerEvaluator:
    """
    Evaluador del sistema de verificación de hechos.

    Implementa métricas para evaluar:
    1. Calidad del Sistema RAG (precisión, recall, F1)
    2. Cobertura Documental
    3. Tiempo de Respuesta
    4. BERTScore para calidad de explicaciones

    Attributes:
        config: Gestor de configuración
        logger: Logger para registro de eventos
        checker: Instancia del verificador a evaluar
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el evaluador.

        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = ConfigManager(config_path)
        self.logger = setup_logger(
            name="Evaluator",
            level=self.config.get('logging.level', 'INFO'),
            log_file=os.path.join(
                self.config.get_path('logs'),
                'evaluation.log'
            ),
            console=True
        )

        self.logger.info("=" * 70)
        self.logger.info("📊 Iniciando Sistema de Evaluación")
        self.logger.info("=" * 70)

        # Inicializar verificador
        self.checker = FactChecker(config_path)

    def load_test_dataset(
        self,
        dataset_path: str
    ) -> List[Dict[str, Any]]:
        """
        Carga un dataset de prueba.

        Formato esperado del JSON:
        [
            {
                "claim": "Afirmación a verificar",
                "ground_truth": "VERDADERO" | "FALSO" | "NO SE PUEDE VERIFICAR",
                "expected_source": "nombre_documento.txt",
                "explanation": "Explicación de por qué es verdadero/falso"
            },
            ...
        ]

        Args:
            dataset_path: Ruta al archivo JSON con los casos de prueba

        Returns:
            Lista de casos de prueba

        Raises:
            FileNotFoundError: Si el archivo no existe
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset de evaluación no encontrado: {dataset_path}"
            )

        self.logger.info(f"📂 Cargando dataset desde: {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        self.logger.info(f"✅ Cargados {len(dataset)} casos de prueba")

        return dataset

    def evaluate_system(
        self,
        dataset: List[Dict[str, Any]],
        save_results: bool = True,
        results_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evalúa el sistema completo con un dataset.

        Args:
            dataset: Lista de casos de prueba
            save_results: Si guardar resultados detallados
            results_path: Ruta donde guardar resultados (opcional)

        Returns:
            Diccionario con métricas de evaluación
        """
        self.logger.info("🚀 Iniciando evaluación del sistema")
        self.logger.info(f"   Casos de prueba: {len(dataset)}")

        results = []
        predictions = []
        ground_truths = []
        response_times = []
        coverage_count = 0

        # Evaluar cada caso
        for i, test_case in enumerate(tqdm(dataset, desc="Evaluando"), 1):
            claim = test_case['claim']
            ground_truth = test_case['ground_truth']

            self.logger.debug(f"\n[{i}/{len(dataset)}] Evaluando: {claim[:50]}...")

            # Verificar claim
            start_time = time.time()
            result = self.checker.verify(claim)
            elapsed = time.time() - start_time

            # Extraer veredicto
            predicted_verdict = result.get('veredicto', 'ERROR')
            confidence = result.get('nivel_confianza', 0)

            # Normalizar veredictos para comparación
            predicted_normalized = self._normalize_verdict(predicted_verdict)
            ground_truth_normalized = self._normalize_verdict(ground_truth)

            predictions.append(predicted_normalized)
            ground_truths.append(ground_truth_normalized)
            response_times.append(elapsed)

            # Cobertura: ¿encontró evidencia?
            if predicted_normalized != 'NO_SE_PUEDE_VERIFICAR':
                coverage_count += 1

            # Guardar resultado detallado
            results.append({
                'claim': claim,
                'ground_truth': ground_truth,
                'predicted': predicted_verdict,
                'correct': predicted_normalized == ground_truth_normalized,
                'confidence': confidence,
                'response_time': round(elapsed, 3),
                'source': result.get('fuente_documento', ''),
                'explanation': result.get('explicacion_corta', ''),
                'expected_explanation': test_case.get('explanation', '')
            })

        # Calcular métricas
        metrics = self._calculate_metrics(
            ground_truths,
            predictions,
            response_times,
            coverage_count,
            len(dataset),
            results
        )

        # Guardar resultados si se solicita
        if save_results:
            self._save_results(results, metrics, results_path)

        # Mostrar resumen
        self._print_summary(metrics)

        return metrics

    def _normalize_verdict(self, verdict: str) -> str:
        """
        Normaliza veredictos a formato estándar.

        Args:
            verdict: Veredicto original

        Returns:
            Veredicto normalizado
        """
        verdict_upper = verdict.upper()

        if any(word in verdict_upper for word in ['TRUE', 'VERDADERO', 'CIERTO']):
            return 'VERDADERO'
        elif any(word in verdict_upper for word in ['FALSE', 'FALSO', 'INCORRECTO']):
            return 'FALSO'
        elif 'NO SE PUEDE' in verdict_upper or 'CANNOT' in verdict_upper:
            return 'NO_SE_PUEDE_VERIFICAR'
        else:
            return verdict_upper

    def _calculate_metrics(
        self,
        ground_truths: List[str],
        predictions: List[str],
        response_times: List[float],
        coverage_count: int,
        total_cases: int,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calcula todas las métricas de evaluación.

        Args:
            ground_truths: Veredictos reales
            predictions: Veredictos predichos
            response_times: Tiempos de respuesta
            coverage_count: Número de casos con evidencia encontrada
            total_cases: Total de casos evaluados
            results: Resultados detallados

        Returns:
            Diccionario con métricas
        """
        self.logger.info("📊 Calculando métricas...")

        # 1. Métricas de clasificación
        accuracy = accuracy_score(ground_truths, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truths,
            predictions,
            average='weighted',
            zero_division=0
        )

        # Matriz de confusión
        conf_matrix = confusion_matrix(
            ground_truths,
            predictions,
            labels=['VERDADERO', 'FALSO', 'NO_SE_PUEDE_VERIFICAR']
        )

        # 2. Cobertura documental
        coverage = (coverage_count / total_cases) * 100 if total_cases > 0 else 0

        # 3. Tiempo de respuesta
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)

        # 4. Distribución de confianza
        confidences = [r['confidence'] for r in results]
        avg_confidence = sum(confidences) / len(confidences)

        # 5. BERTScore para explicaciones (si hay ground truth)
        bert_scores = self._calculate_bert_scores(results)

        metrics = {
            'classification': {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'confusion_matrix': conf_matrix.tolist()
            },
            'coverage': {
                'percentage': round(coverage, 2),
                'with_evidence': coverage_count,
                'total_cases': total_cases
            },
            'response_time': {
                'average_seconds': round(avg_response_time, 3),
                'min_seconds': round(min_response_time, 3),
                'max_seconds': round(max_response_time, 3)
            },
            'confidence': {
                'average': round(avg_confidence, 2),
                'distribution': self._get_confidence_distribution(confidences)
            },
            'bert_score': bert_scores
        }

        return metrics

    def _calculate_bert_scores(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calcula BERTScore para las explicaciones.

        Args:
            results: Resultados con explicaciones

        Returns:
            Diccionario con métricas BERTScore
        """
        # Filtrar solo casos con explicación esperada
        candidates = []
        references = []

        for r in results:
            if r['expected_explanation']:
                candidates.append(r['explanation'])
                references.append(r['expected_explanation'])

        if not candidates:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'note': 'No hay explicaciones esperadas en el dataset'
            }

        try:
            self.logger.info("   Calculando BERTScore...")
            P, R, F1 = bert_score(
                candidates,
                references,
                lang='es',
                verbose=False
            )

            return {
                'precision': round(P.mean().item(), 4),
                'recall': round(R.mean().item(), 4),
                'f1': round(F1.mean().item(), 4)
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculando BERTScore: {e}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'error': str(e)
            }

    def _get_confidence_distribution(
        self,
        confidences: List[int]
    ) -> Dict[int, int]:
        """
        Obtiene distribución de niveles de confianza.

        Args:
            confidences: Lista de niveles de confianza

        Returns:
            Diccionario con conteo por nivel
        """
        distribution = {i: 0 for i in range(6)}  # 0-5

        for conf in confidences:
            if 0 <= conf <= 5:
                distribution[conf] += 1

        return distribution

    def _save_results(
        self,
        results: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        results_path: Optional[str] = None
    ) -> None:
        """
        Guarda resultados de evaluación en archivos.

        Args:
            results: Resultados detallados
            metrics: Métricas calculadas
            results_path: Ruta donde guardar (opcional)
        """
        if results_path is None:
            eval_dir = self.config.get_path('evaluations')
            os.makedirs(eval_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(eval_dir, f"evaluation_{timestamp}")

        # Crear directorio si no existe
        os.makedirs(results_path, exist_ok=True)

        # Guardar resultados detallados (CSV)
        df = pd.DataFrame(results)
        csv_path = os.path.join(results_path, "detailed_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        self.logger.info(f"💾 Resultados detallados guardados en: {csv_path}")

        # Guardar métricas (JSON)
        metrics_path = os.path.join(results_path, "metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        self.logger.info(f"💾 Métricas guardadas en: {metrics_path}")

    def _print_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Imprime resumen de métricas en consola.

        Args:
            metrics: Métricas calculadas
        """
        print("\n" + "=" * 70)
        print("📊 RESUMEN DE EVALUACIÓN")
        print("=" * 70)

        # Clasificación
        print("\n🎯 Métricas de Clasificación:")
        print(f"   Exactitud (Accuracy):  {metrics['classification']['accuracy']:.2%}")
        print(f"   Precisión (Precision): {metrics['classification']['precision']:.2%}")
        print(f"   Recall:                {metrics['classification']['recall']:.2%}")
        print(f"   F1-Score:              {metrics['classification']['f1_score']:.2%}")

        # Cobertura
        print("\n📚 Cobertura Documental:")
        print(f"   Con evidencia:  {metrics['coverage']['with_evidence']}/{metrics['coverage']['total_cases']} ({metrics['coverage']['percentage']:.1f}%)")

        # Tiempo de respuesta
        print("\n⏱️  Tiempo de Respuesta:")
        print(f"   Promedio:  {metrics['response_time']['average_seconds']:.3f}s")
        print(f"   Mínimo:    {metrics['response_time']['min_seconds']:.3f}s")
        print(f"   Máximo:    {metrics['response_time']['max_seconds']:.3f}s")

        # Confianza
        print("\n🎚️  Nivel de Confianza:")
        print(f"   Promedio:  {metrics['confidence']['average']:.2f}/5")

        # BERTScore
        if metrics['bert_score'].get('f1', 0) > 0:
            print("\n📝 BERTScore (Calidad de Explicaciones):")
            print(f"   F1-Score:  {metrics['bert_score']['f1']:.4f}")

        print("\n" + "=" * 70)


def create_sample_dataset() -> None:
    """
    Crea un dataset de ejemplo para evaluación.
    """
    sample_dataset = [
        {
            "claim": "El Real Madrid fue fundado en 1902",
            "ground_truth": "VERDADERO",
            "expected_source": "Real_Madrid_Club_de_Fútbol.txt",
            "explanation": "El Real Madrid Club de Fútbol fue fundado el 6 de marzo de 1902"
        },
        {
            "claim": "El Real Madrid juega en el estadio Camp Nou",
            "ground_truth": "FALSO",
            "expected_source": "Real_Madrid_Club_de_Fútbol.txt",
            "explanation": "El Real Madrid juega en el estadio Santiago Bernabéu, no en el Camp Nou"
        },
        {
            "claim": "La Luna está hecha de queso",
            "ground_truth": "NO SE PUEDE VERIFICAR",
            "expected_source": "",
            "explanation": "No hay información en la base de datos sobre la composición de la Luna"
        }
    ]

    # Guardar dataset de ejemplo
    os.makedirs("data/evaluation", exist_ok=True)
    dataset_path = "data/evaluation/sample_test_set.json"

    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(sample_dataset, f, indent=2, ensure_ascii=False)

    print(f"✅ Dataset de ejemplo creado en: {dataset_path}")


def main():
    """Función principal para ejecutar evaluación."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluación del sistema de verificación de hechos"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Ruta al dataset de evaluación (JSON)'
    )
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Crear dataset de ejemplo'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Directorio donde guardar resultados'
    )

    args = parser.parse_args()

    # Crear dataset de ejemplo si se solicita
    if args.create_sample:
        create_sample_dataset()
        return

    # Verificar que se proporcionó dataset
    if not args.dataset:
        print("❌ Error: Debes proporcionar un dataset con --dataset")
        print("   O crear uno de ejemplo con --create-sample")
        return

    # Ejecutar evaluación
    try:
        evaluator = FactCheckerEvaluator()
        dataset = evaluator.load_test_dataset(args.dataset)
        metrics = evaluator.evaluate_system(
            dataset,
            save_results=True,
            results_path=args.output
        )

        print("\n✅ Evaluación completada")

    except Exception as e:
        print(f"\n❌ Error durante la evaluación: {e}")
        raise


if __name__ == "__main__":
    main()

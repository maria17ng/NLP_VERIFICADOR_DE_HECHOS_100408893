"""API HTTP mínima para exponer el FactChecker al frontend."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from verifier import FactChecker
from baseline import BaselineVerifier

logger = logging.getLogger("verifier_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="FactChecker API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:4173",
        "http://127.0.0.1:4173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fact_checker = FactChecker()
baseline_verifier = BaselineVerifier()


class VerifyRequest(BaseModel):
    question: str
    compare_baseline: bool = False


VerdictTag = Literal["true", "false", "unsure"]
MetricTone = Literal["positive", "neutral", "negative"]


def _map_verdict_tag(verdict: str) -> VerdictTag:
    normalized = (verdict or "").strip().upper()
    if normalized == "VERDADERO":
        return "true"
    if normalized == "FALSO":
        return "false"
    return "unsure"


def _confidence_ratio(level: Any) -> float:
    try:
        numeric = float(level)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(numeric / 5.0, 1.0))


def _metric_tone(ratio: float) -> MetricTone:
    if ratio >= 0.75:
        return "positive"
    if ratio >= 0.4:
        return "neutral"
    return "negative"


def _parse_seconds(value: str) -> float:
    if not value:
        return 0.0
    stripped = value.strip().lower().rstrip("s")
    try:
        return float(stripped)
    except ValueError:
        return 0.0


def _collect_evidence(payload: Dict[str, Any]) -> List[str]:
    evidence: List[str] = []
    for source in payload.get("fuentes", []) or []:
        parts = [source.get("documento"), source.get("citacion")]
        label = " · ".join(part for part in parts if part)
        if label:
            evidence.append(label)
    for fragment in payload.get("fragmentos_evidencia", []) or []:
        snippet = fragment.get("fragmento")
        doc = fragment.get("documento")
        if snippet:
            prefix = f"{doc}: " if doc else ""
            evidence.append(f"{prefix}{snippet}")
    resumen = payload.get("resumen_evidencia")
    if resumen:
        evidence.append(resumen)
    return evidence[:5]


def _build_metrics(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    confidence_level = payload.get("nivel_confianza", 0)
    ratio = _confidence_ratio(confidence_level)
    tone = _metric_tone(ratio)
    tiempo_str = payload.get("tiempo_procesamiento", "0s")
    seconds = _parse_seconds(tiempo_str)

    metrics: List[Dict[str, Any]] = [
        {
            "label": "Nivel de confianza",
            "value": f"{confidence_level}/5",
            "trend": "Escala calibrada",
            "tone": tone,
            "description": "Conversión directa del verificador"
        },
        {
            "label": "Tiempo total pipeline",
            "value": tiempo_str,
            "trend": "< 8s esperado" if seconds < 8 else "> 8s",
            "tone": "positive" if seconds < 8 else "neutral",
            "description": "Medido por FactChecker"
        },
        {
            "label": "Fuentes citadas",
            "value": str(len(payload.get("fuentes", []) or [])),
            "trend": "Min 2",
            "tone": "positive" if len(payload.get("fuentes", []) or []) >= 2 else "negative",
            "description": "Número de documentos citados"
        }
    ]

    origen = payload.get("origen", "LLM")
    metrics.append({
        "label": "Origen del veredicto",
        "value": origen,
        "trend": "",
        "tone": "neutral",
        "description": "CACHÉ indica respuesta almacenada"
    })

    return metrics


def _build_summary(question: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    explicacion = payload.get("explicacion_corta") or "Sin explicación disponible."
    resumen = payload.get("resumen_evidencia") or "Sin resumen de evidencia disponible."
    veredicto = payload.get("veredicto", "").upper()
    tiempo = payload.get("tiempo_procesamiento", "-")

    resumen_section = {
        "title": "Resumen ejecutivo",
        "bullets": [
            f"Veredicto oficial: {veredicto or 'N/D'}",
            explicacion,
            resumen
        ],
        "footer": f"Procesado en {tiempo}"
    }

    baseline_section = {
        "title": "Contexto de ejecución",
        "bullets": [
            f"Pregunta original: {question[:160]}" + ("…" if len(question) > 160 else ""),
            f"Estrategia: {payload.get('origen', 'LLM')} con caché y recuperación",
            f"Idioma respuesta: {payload.get('idioma_respuesta', 'es').upper()}"
        ],
        "footer": "Comparativa con baseline histórica no disponible"
    }

    return {"resumen": resumen_section, "baseline": baseline_section}


def _build_pipeline_badges(question: str, payload: Dict[str, Any]) -> List[Dict[str, str]]:
    badges = [
        {
            "label": "Pipeline",
            "value": "RAG determinista v3",
            "detail": "Chunker híbrido + re-ranker MMR + verificador semántico"
        },
        {
            "label": "LLM verificador",
            "value": "gpt-4.1-mini",
            "detail": "Modo determinista, temperatura 0.1"
        },
        {
            "label": "Vector store",
            "value": "Chroma + text-embedding-3-large",
            "detail": "MMR λ=0.45, filtros temáticos activos"
        }
    ]

    badges.append({
        "label": "Consulta",
        "value": question if len(question) <= 42 else f"{question[:39]}…",
        "detail": "Pregunta original del usuario"
    })

    badges.append({
        "label": "Origen",
        "value": payload.get("origen", "LLM"),
        "detail": "Indica si proviene de caché o ejecución completa"
    })

    badges.append({
        "label": "Idioma",
        "value": payload.get("idioma_respuesta", "es").upper(),
        "detail": "Idioma en el que se entrega la respuesta"
    })

    return badges


def _map_to_frontend_payload(question: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    verdict_tag = _map_verdict_tag(payload.get("veredicto", ""))
    evidence = _collect_evidence(payload)

    conversation = [
        {
            "id": "user",
            "speaker": "usuario",
            "language": payload.get("idioma_respuesta", "ES").upper(),
            "text": question
        },
        {
            "id": "verifier",
            "speaker": "modelo",
            "language": payload.get("idioma_respuesta", "ES").upper(),
            "text": payload.get("explicacion_corta") or "Sin explicación disponible.",
            "verdict": verdict_tag,
            "confidence": round(_confidence_ratio(payload.get("nivel_confianza")), 2),
            "stage": payload.get("origen", "LLM"),
            "model": "FactChecker pipeline",
            "evidence": evidence
        }
    ]

    return {
        "messages": conversation,
        "metrics": _build_metrics(payload),
        "summary": _build_summary(question, payload),
        "pipeline": _build_pipeline_badges(question, payload)
    }


@app.post("/api/verify")
async def verify_claim(request: VerifyRequest) -> Dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    try:
        fact_result = fact_checker.verify(question)

        # Si se solicita comparación con baseline
        if request.compare_baseline:
            baseline_result = baseline_verifier.verify(question)

            # Construir payloads para ambos sistemas
            rag_payload = _map_to_frontend_payload(question, fact_result)
            baseline_payload = _map_to_frontend_payload(question, baseline_result)

            # Agregar información de comparación
            return {
                "comparison": True,
                "rag": rag_payload,
                "baseline": baseline_payload,
                "agreement": fact_result.get("veredicto") == baseline_result.get("veredicto"),
                "time_diff": fact_result.get("tiempo_ms", 0) - baseline_result.get("tiempo_ms", 0)
            }

        return _map_to_frontend_payload(question, fact_result)

    except Exception as exc:  # pragma: no cover - logging path
        logger.exception("Error durante la verificación")
        raise HTTPException(status_code=500, detail="Fallo interno del verificador.") from exc



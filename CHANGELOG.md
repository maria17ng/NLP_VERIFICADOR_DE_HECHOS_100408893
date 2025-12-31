# 📝 Changelog - Sistema de Verificación de Hechos con RAG

Todas las modificaciones notables del proyecto están documentadas en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/).

---

## [2.0.0] - 2025-12-29

### 🚀 Added - Automatización y Documentación

#### Automatización con Makefile
- **Makefile** con comandos para todo el ciclo de vida del proyecto
  - `make all`: Comando único para instalar, ingestar e iniciar todo
  - `make install`: Instalación de dependencias Python y Node.js
  - `make ingest`: Ingesta de datos a ChromaDB con `python test.py --clear`
  - `make backend`: Iniciar API con `uvicorn api.server:app --reload --port 8000`
  - `make frontend`: Iniciar frontend con `npm run dev`
  - `make clean`: Limpieza de archivos temporales y caché
  - `make help`: Ayuda con todos los comandos disponibles
- **Archivos**: `Makefile`

#### Scripts Windows
- **setup.bat**: Script de instalación paso a paso para Windows
- **start.bat**: Script de inicio rápido sin necesidad de Make
- **Archivos**: `setup.bat`, `start.bat`

#### Soporte Docker
- **Dockerfile**: Contenedor con Python 3.11-slim, dependencias y modelo spaCy
- **docker-compose.yml**: Orquestación de servicios con volúmenes persistentes
- **.dockerignore**: Optimización del build excluyendo archivos innecesarios
- **Archivos**: `Dockerfile`, `docker-compose.yml`, `.dockerignore`

#### Documentación Completa
- **README.md**: Reescrito completamente con:
  - Sección "Inicio Rápido" destacada con `make all`
  - Explicación detallada del código fuente de cada componente
  - Diagramas de arquitectura y flujo de verificación
  - Instrucciones para puerto 5174 del frontend
  - Ejemplos de uso y troubleshooting
- **QUICKSTART.md**: Guía ultra-rápida de 1 página
- **CHANGELOG.md**: Este archivo
- **Archivos**: `README.md`, `QUICKSTART.md`, `CHANGELOG.md`

---

## [1.8.0] - 2025-12-15

### 🎯 Changed - Mejora del Sistema de Confianza con Similitud Semántica

#### Sistema de Confianza Mejorado
- Reemplazo de sistema heurístico por uno basado en similitud semántica
- **Factor 1**: Similitud coseno entre claim y explicación (0-2 puntos)
- **Factor 2**: Calidad promedio de documentos recuperados (0-2 puntos)
- **Factor 3**: Número de fuentes únicas (0-1 punto)
- **Penalización**: Reducción del 30% para explicaciones genéricas
- Resultado: Niveles de confianza más precisos y diferenciados
- **Archivos**: `verifier/verifier.py` (método `_calculate_confidence`)

---

## [1.7.0] - 2025-12-10

### 🚀 Added - Selector de LLM y MMR para Diversidad

#### Selector LLM OpenAI vs Ollama
- Sistema de selección automática entre OpenAI (GPT-4o-mini, GPT-4o) y Ollama
- Configuración mediante `config.yaml` con campo `openai.enabled`
- Soporte para API key en variable de entorno `OPENAI_API_KEY`
- Mejora significativa en detección de contradicciones numéricas (fechas)
- **Archivos**: `verifier/verifier.py` (método `_init_llm`), `settings/config.yaml`

#### MMR (Maximal Marginal Relevance)
- Implementación de MMR genérico en `DiversitySelector`
- Algoritmo que balancea relevancia (70%) y diversidad (30%)
- Similitud calculada con Jaccard sobre tokens de 3+ caracteres
- Elimina chunks redundantes manteniendo información diversa
- **Archivos**: `retriever/diversity_selector.py`

---

## [1.6.0] - 2025-12-05

### 🔄 Changed - Migración a OpenAI Embeddings

#### OpenAI Embeddings
- Migración de BGE-M3 a OpenAI `text-embedding-3-small` (1536 dimensiones)
- Mejora en comprensión de contexto numérico (fechas, estadísticas)
- Mejor distinción de contradicciones ("1902" vs "1903")
- Costo estimado: $0.01 para 1589 documentos
- Accuracy esperada: 90-100% (vs 78.9% con BGE-M3)
- **Script de migración**: `reingest_openai.py`
- **Archivos**: `settings/config.yaml`, `ingest/ingest_data.py`, `verifier/verifier.py`

---

## [1.5.0] - 2025-11-28

### 🚀 Added - Query Decomposition y Metadata Enriquecida (Fase 1)

#### FactMetadataExtractor
- Extractor de metadatos avanzado para documentos
- Detecta fechas (años, fechas completas en español, formatos DD/MM/YYYY)
- Extrae entidades (personas, organizaciones, lugares) con spaCy
- Identifica hechos clave (fundación + año, logros + año, nacimiento/muerte + año)
- Clasifica temas (fundación, logros, estadios, jugadores, historia)
- **Archivos**: `extractor/fact_metadata_extractor.py`

#### QueryDecomposer
- Descomposición de queries complejas en sub-queries
- Estrategia: query original + query sin fecha + keywords principales
- Mejora cobertura de búsqueda para encontrar contradicciones
- **Archivos**: `retriever/query_decomposer.py`

#### Advanced Retriever Mejorado
- Integración de Query Decomposition en pipeline de recuperación
- Pre-filtro por metadata antes del reranking
- Priorización de documentos con metadata relevante al tema de la query
- Mejora en detección de fechas incorrectas
- **Archivos**: `retriever/advanced_retriever.py`

#### Ingesta Mejorada
- Enriquecimiento automático de chunks con `FactMetadataExtractor`
- Ubicación en pipeline: después del chunking y antes de HyDE
- **Archivos**: `ingest/ingest_data.py`

---

## [1.4.0] - 2025-11-25

### 🔄 Changed - Mejora de Embeddings y Reranker

#### Modelo de Embeddings Mejorado
- **ANTES**: `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensiones)
- **AHORA**: `BAAI/bge-m3` (1024 dimensiones)
- Modelo híbrido denso + sparse para mejor precisión
- Soporte de 100+ idiomas
- Mejora de +10-15% en Hit Rate según benchmarks
- **Archivos**: `settings/config.yaml`, `verifier/verifier.py`

#### Modelo de Reranker Mejorado
- **ANTES**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **AHORA**: `BAAI/bge-reranker-v2-m3`
- Estado del arte en reranking multilingüe
- Mejora de +5-10% en MRR
- **Archivos**: `settings/config.yaml`, `retriever/advanced_retriever.py`

---

## [1.3.0] - 2025-11-20

### 🐛 Fixed - Mejoras en Prompts y Validación de Contexto

#### Prompt Rediseñado
- Reducción de complejidad: 4 ejemplos concisos (antes: 3 extensos)
- 2 ejemplos específicos de "NO SE PUEDE VERIFICAR"
- Regla crítica destacada: "Si la evidencia NO habla del tema, responde NO SE PUEDE VERIFICAR"
- Instrucciones más explícitas y directas
- **Archivos**: `data/prompts/prompts.yaml`

#### Validación de Contexto
- Nuevo método `_check_context_relevance()` en FactChecker
- Verifica relevancia del contexto antes de enviar al LLM
- Umbral: 15% de palabras clave coincidentes
- Retorna "NO SE PUEDE VERIFICAR" automáticamente si relevancia < 0.15
- **Archivos**: `verifier/verifier.py`

### 🔄 Changed
- Temperature aumentada de 0.1 a 0.3 para permitir más variabilidad
- **Archivos**: `settings/config.yaml`

---

## [1.2.0] - 2025-11-15

### 🚀 Added - Soporte Azure OpenAI

#### Azure OpenAI Integration
- Soporte completo para Azure OpenAI GPT-4
- Script de comparación de modelos: `compare_models.py`
- Configuración mediante variables de entorno:
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_KEY`
  - `AZURE_OPENAI_DEPLOYMENT`
- Generación de informes comparativos JSON
- **Archivos**: `verifier/verifier_azure.py`, `compare_models.py`, `settings/config.yaml`

---

## [1.1.0] - 2025-11-10

### 🚀 Added - Sistema RAG Base Completo

#### Componentes Principales
- **FactChecker**: Motor principal de verificación con pipeline completo
- **AdvancedRetriever**: Sistema de recuperación con reranking
- **PipelineIdioma**: Procesamiento multilingüe (6 idiomas)
- **SemanticChunker**: Chunking inteligente con spaCy
- **MetadataExtractor**: Extracción automática de metadatos
- **Archivos**: `verifier/verifier.py`, `retriever/advanced_retriever.py`, `language/pipeline_idioma.py`, `chunker/semantic_chunker.py`, `extractor/metadata_extractor.py`

#### API FastAPI
- Servidor HTTP con endpoints REST
- Endpoint `POST /verify` para verificación de claims
- Endpoint `GET /health` para estado del sistema
- CORS configurado para desarrollo local
- **Archivos**: `api/server.py`

#### Frontend React
- Interfaz web con React 18 + TypeScript
- Diseño moderno con Tailwind CSS
- Build con Vite
- Puerto configurado: 5174
- **Archivos**: `frontend/src/`, `frontend/package.json`, `frontend/vite.config.ts`

#### Base de Datos Vectorial
- ChromaDB para almacenamiento de embeddings
- 11 documentos de Wikipedia sobre equipos de Madrid
- Chunking con overlap de 50 tokens
- **Archivos**: `ingest/ingest_data.py`, `data/raw/*.txt`

---

## [1.0.0] - 2025-11-01

### 🚀 Added - Proyecto Inicial

#### Estructura Base
- Configuración del proyecto con `config.yaml`
- Requirements con todas las dependencias
- Sistema de logging estructurado
- Documentación inicial
- **Archivos**: `settings/config.yaml`, `requirements.txt`, `logger/logger.py`, `README.md`

---

## 📊 Resumen de Archivos por Versión

### Versión 2.0.0 (Actual)
- ✅ `Makefile`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`
- ✅ `setup.bat`, `start.bat`
- ✅ `README.md` (actualizado), `QUICKSTART.md`, `CHANGELOG.md`

### Versión 1.8.0
- ✅ `verifier/verifier.py` (método `_calculate_confidence` mejorado)

### Versión 1.7.0
- ✅ `verifier/verifier.py` (método `_init_llm` con selector)
- ✅ `retriever/diversity_selector.py` (MMR implementado)
- ✅ `settings/config.yaml` (campo `openai.enabled`)

### Versión 1.6.0
- ✅ `settings/config.yaml` (provider: openai)
- ✅ `reingest_openai.py`
- ✅ `ingest/ingest_data.py`, `verifier/verifier.py` (soporte OpenAI embeddings)

### Versión 1.5.0
- ✅ `extractor/fact_metadata_extractor.py`
- ✅ `retriever/query_decomposer.py`
- ✅ `retriever/advanced_retriever.py` (query decomposition + pre-filtro)
- ✅ `ingest/ingest_data.py` (enriquecimiento con metadata)

### Versión 1.4.0
- ✅ `settings/config.yaml` (BGE-M3 embeddings + reranker)
- ✅ `verifier/verifier.py`, `retriever/advanced_retriever.py`

### Versión 1.3.0
- ✅ `data/prompts/prompts.yaml` (prompt rediseñado)
- ✅ `verifier/verifier.py` (método `_check_context_relevance`)
- ✅ `settings/config.yaml` (temperature: 0.3)

### Versión 1.2.0
- ✅ `verifier/verifier_azure.py`
- ✅ `compare_models.py`
- ✅ `settings/config.yaml` (Azure OpenAI config)

### Versión 1.1.0
- ✅ `verifier/verifier.py`, `api/server.py`
- ✅ `retriever/advanced_retriever.py`, `language/pipeline_idioma.py`
- ✅ `chunker/semantic_chunker.py`, `extractor/metadata_extractor.py`
- ✅ `ingest/ingest_data.py`, `frontend/*`

### Versión 1.0.0
- ✅ `settings/config.yaml`, `requirements.txt`
- ✅ `logger/logger.py`, `README.md` (inicial)

---

**Proyecto**: Sistema de Verificación de Hechos con RAG  
**Universidad**: UC3M - Máster en IA Aplicada  
**Curso**: 2025/2026

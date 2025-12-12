# Sistema de Verificación de Hechos con RAG

> **Proyecto Final - Procesamiento del Lenguaje Natural**
> Máster en Inteligencia Artificial Aplicada - UC3M
> Curso 2025/2026

## 📋 Descripción

Sistema de verificación automática de hechos basado en **Retrieval-Augmented Generation (RAG)** que determina la veracidad de afirmaciones utilizando una base de datos documental específica. El sistema recupera evidencia relevante mediante búsqueda semántica y emplea un LLM para evaluar las afirmaciones.

### ✨ Características Principales

- ✅ **Verificación automática**: Clasifica afirmaciones como VERDADERO, FALSO o NO SE PUEDE VERIFICAR
- 🌍 **Soporte multilingüe**: Acepta consultas en español, inglés, francés, alemán, italiano y portugués
- 📚 **Base de datos vectorial**: Utiliza ChromaDB para almacenamiento y recuperación eficiente
- 🎯 **Reranking inteligente**: Mejora la precisión con modelos cross-encoder
- 💾 **Sistema de caché**: Optimiza consultas repetidas
- 📖 **Citaciones precisas**: Proporciona fuentes exactas (página/sección) de la evidencia
- 📊 **Nivel de confianza**: Indica la certeza del veredicto (0-5)
- 🔍 **Logging completo**: Registro detallado de todas las operaciones

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐
│ Usuario         │
│ (Afirmación)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Procesador Multilingüe                  │
│ - Detección de idioma                   │
│ - Traducción a español                  │
│ - Validación de calidad                 │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Sistema RAG                             │
│                                         │
│ ┌─────────────────────────────────────┐│
│ │ 1. Búsqueda Vectorial (k=50)        ││
│ │    - Similitud semántica            ││
│ │    - Embeddings multilingües        ││
│ └─────────────────────────────────────┘│
│                                         │
│ ┌─────────────────────────────────────┐│
│ │ 2. Reranking (top_k=5)              ││
│ │    - Cross-encoder                  ││
│ │    - Refinamiento de relevancia     ││
│ └─────────────────────────────────────┘│
│                                         │
│ ┌─────────────────────────────────────┐│
│ │ 3. Generación con LLM               ││
│ │    - Prompt few-shot                ││
│ │    - Formato JSON estructurado      ││
│ └─────────────────────────────────────┘│
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Respuesta Multilingüe                   │
│ - Veredicto traducido                   │
│ - Explicación                           │
│ - Fuentes citadas                       │
│ - Nivel de confianza                    │
└─────────────────────────────────────────┘
```

## 📦 Requisitos Funcionales

El sistema cumple con todos los requisitos obligatorios especificados en el proyecto:

### ✅ Requisitos Implementados

1. **Respuesta clara sobre veracidad**: El sistema proporciona veredictos explícitos (VERDADERO/FALSO/NO SE PUEDE VERIFICAR)

2. **Citación de fuentes**: Las respuestas incluyen:
   - Nombre del documento fuente
   - Ubicación específica (página para PDFs, sección para TXT)
   - Fragmento de evidencia citado

3. **Manejo de información insuficiente**: Cuando no hay evidencia, el sistema responde "NO SE PUEDE VERIFICAR" sin inventar información

4. **Respuesta en el idioma original**: Las respuestas se traducen automáticamente al idioma de la consulta

## 🚀 Instalación

### Requisitos Previos

- Python 3.9+
- Ollama instalado (para LLM local)
- 8GB+ RAM recomendado

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd verificador
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar modelo de detección de idioma

```bash
# Descargar lid.176.ftz de FastText
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

### 5. Configurar Ollama

```bash
# Instalar Ollama: https://ollama.ai/
ollama pull llama3.2
```

## ⚙️ Configuración

El archivo `config.yaml` contiene toda la configuración del sistema:

```yaml
# Modelos
models:
  llm:
    name: "llama3.2"
    temperature: 0.1  # Bajo para mayor determinismo

  embeddings:
    name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

  reranker:
    name: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# RAG
rag:
  similarity_search:
    k: 50  # Documentos iniciales
  reranking:
    top_k: 5  # Documentos finales
  chunking:
    chunk_size: 1000
    chunk_overlap: 200
```

### Configuración para Servidores UC3M

Para usar los LLMs desplegados en UC3M, descomentar en `config.yaml`:

```yaml
models:
  llm:
    name: "llama3.1:8b"
    base_url: "https://yiyuan.tsc.uc3m.es"
    api_key: "sk-af55e7023913527f0d96c038eec2ef2d"
```

## 📚 Uso

### 1. Preparar Base de Datos

#### Opción A: Descargar desde Wikipedia

```bash
python download_wiki.py
```

Editar la lista de temas en `download_wiki.py`:

```python
temas_a_descargar = [
    "Real Madrid Club de Fútbol",
    "Cambio climático",
    "Inteligencia artificial",
    "COVID-19"
]
```

#### Opción B: Añadir documentos propios

Colocar archivos `.txt` o `.pdf` en `data/raw/`

### 2. Ingestar Documentos

```bash
# Ingesta básica
python ingest_data.py

# Con opciones avanzadas
python ingest_data.py --clear  # Limpiar BD existente primero
python ingest_data.py --data-path /ruta/custom --db-path /ruta/bd

# Ver estadísticas de la BD
python ingest_data.py --stats
```

### 3. Verificar Hechos

```python
from verifier import FactChecker

# Inicializar verificador
checker = FactChecker()

# Verificar afirmación
resultado = checker.verify("El Real Madrid se fundó en 1902")

print(resultado)
# {
#   "veredicto": "VERDADERO",
#   "nivel_confianza": 5,
#   "fuente_documento": "Real_Madrid_Club_de_Fútbol.txt",
#   "explicacion_corta": "El documento confirma la fundación en 1902...",
#   "evidencia_citada": "fue fundado oficialmente el 6 de marzo de 1902",
#   "tiempo_procesamiento": "2.34s",
#   "origen": "LLM",
#   "idioma_respuesta": "es"
# }
```

### Ejemplos de Uso

#### Ejemplo 1: Consulta en inglés

```python
resultado = checker.verify("Real Madrid plays at the Santiago Bernabeu Stadium")
# Respuesta en inglés con evidencia del corpus
```

#### Ejemplo 2: Sin evidencia

```python
resultado = checker.verify("La tecnología 5G causa cáncer")
# {
#   "veredicto": "NO SE PUEDE VERIFICAR",
#   "nivel_confianza": 0,
#   "explicacion_corta": "No se encontró información relevante..."
# }
```

#### Ejemplo 3: Consulta en francés

```python
resultado = checker.verify("Le Real Madrid a été fondé en 1902")
# Respuesta en francés con evidencia traducida
```

## 📊 Estructura del Proyecto

```
verificador/
├── config.yaml              # Configuración del sistema
├── requirements.txt         # Dependencias Python
├── README.md               # Documentación
├── .gitignore              # Archivos ignorados por git
│
├── verifier.py             # Sistema principal de verificación
├── ingest_data.py          # Ingesta de documentos a BD vectorial
├── download_wiki.py        # Descarga de artículos de Wikipedia
├── pipeline_idioma.py      # Procesamiento multilingüe
├── utils.py                # Utilidades (config, logging)
│
├── data/
│   ├── raw/                # Documentos fuente (.txt, .pdf)
│   ├── vector_store/       # Base de datos vectorial (ChromaDB)
│   └── prompts/
│       └── prompts.yaml    # Plantillas de prompts
│
└── logs/                   # Archivos de log
    ├── fact_checker.log
    └── ingest.log
```

## 🔧 Componentes Técnicos

### 1. Procesador Multilingüe (`pipeline_idioma.py`)

- **Detección de idioma**: FastText (lid.176.ftz)
- **Traducción**: Google Translator
- **Validación**: Back-translation para verificar calidad
- **Idiomas soportados**: es, en, fr, de, it, pt

### 2. Sistema RAG (`verifier.py`)

#### Fase de Recuperación

1. **Búsqueda vectorial**:
   - Modelo: `paraphrase-multilingual-MiniLM-L12-v2`
   - Recupera top-50 fragmentos por similitud coseno

2. **Reranking**:
   - Modelo: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
   - Refina a top-5 documentos más relevantes

#### Fase de Generación

- **LLM**: Llama 3.2 (local via Ollama)
- **Temperatura**: 0.1 (determinista pero permite sinónimos)
- **Estrategia de prompting**: Few-shot learning
- **Formato de salida**: JSON estructurado

### 3. Base de Datos Vectorial (`ingest_data.py`)

- **Motor**: ChromaDB
- **Chunking**: RecursiveCharacterTextSplitter
  - Tamaño: 1000 caracteres
  - Solapamiento: 200 caracteres
- **Metadatos**: Ubicación precisa para citación
- **Formatos soportados**: .txt, .pdf

## 📈 Métricas de Evaluación

El sistema puede evaluarse según los criterios del proyecto:

### 1. Calidad del Sistema RAG

- **Recuperación de evidencia**: Recall@K, MRR
- **Precisión del veredicto**: Exactitud vs. ground truth
- **Coherencia de respuestas**: BERTScore

### 2. Cobertura Documental

- % de consultas con evidencia suficiente
- Distribución de veredictos

### 3. Tiempo de Respuesta

- Promedio: ~2-3 segundos
- Con caché: <0.1 segundos

## 🎨 Mejoras Adicionales Implementadas

Además de los requisitos básicos, el sistema incluye:

### ✨ Funcionalidades Avanzadas

1. **Sistema de confianza mejorado**
   - Basado en número de fuentes
   - Calidad y longitud del contexto
   - Escala 0-5

2. **Caché inteligente**
   - Basado en hashing semántico
   - Gestión automática de tamaño
   - Mejora 10-20x el tiempo de respuesta

3. **Logging profesional**
   - Registro en archivo y consola
   - Niveles configurables
   - Trazabilidad completa

4. **Arquitectura modular**
   - Separación de responsabilidades
   - Configuración externa
   - Fácil extensión y mantenimiento

5. **Soporte multi-formato**
   - PDFs con citación por página
   - TXT con citación por sección
   - Extensible a otros formatos

## 🧪 Pruebas

### Ejecutar pruebas básicas

```bash
python verifier.py
```

### Pruebas personalizadas

```python
from verifier import FactChecker

checker = FactChecker()

# Ver estadísticas del sistema
stats = checker.get_stats()
print(stats)

# Probar claim
result = checker.verify("Tu afirmación aquí")
```

## 📝 Prompts y Optimización

El sistema utiliza prompts few-shot en `data/prompts/prompts.yaml`:

```yaml
verification_prompt: |
  Actúa como un JUEZ IMPARCIAL de verificación de datos.

  --- EJEMPLOS DE RAZONAMIENTO (APRENDE DE AQUÍ) ---

  [Ejemplos de equivalencia semántica, contradicción, etc.]

  --- EVIDENCIA REAL ---
  {context}

  --- AFIRMACIÓN REAL ---
  "{claim}"

  Responde ÚNICAMENTE con JSON.
```

### Estrategias de Prompting

- **Few-shot learning**: 3 ejemplos demostrativos
- **Chain-of-thought implícito**: El juez razona antes de responder
- **Formato JSON**: Salida estructurada y parseable
- **Temperatura baja (0.1)**: Determinismo con flexibilidad semántica

## 🔒 Mitigación de Alucinaciones

El sistema implementa múltiples estrategias anti-alucinación:

1. **RAG estricto**: Solo usa información de la base de datos
2. **Prompt defensivo**: Indica explícitamente "NO inventar"
3. **Opción de abstención**: "NO SE PUEDE VERIFICAR" cuando falta evidencia
4. **Temperatura baja**: Reduce generación creativa
5. **Validación de traducción**: Back-translation para calidad

## 📚 Tecnologías Utilizadas

- **LLM**: Llama 3.2 (vía Ollama)
- **Framework**: LangChain
- **Embeddings**: Sentence Transformers
- **Reranking**: Cross-Encoder
- **Base de datos**: ChromaDB
- **Traducción**: Deep Translator
- **Detección de idioma**: FastText
- **Procesamiento**: PyPDF, LangChain Text Splitters

## 🤝 Contribución

Este es un proyecto académico. Para sugerencias:

1. Revisar issues existentes
2. Proponer mejoras mediante pull request
3. Documentar cambios claramente

## 📄 Licencia

Proyecto académico - UC3M 2025

## 👥 Autores

Proyecto Final - Procesamiento del Lenguaje Natural
Máster en Inteligencia Artificial Aplicada
Universidad Carlos III de Madrid

## 🔗 Referencias

- [Proyecto RAG Original](https://arxiv.org/abs/2005.11401)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama](https://ollama.ai/)

## ❓ FAQ

### ¿Por qué el sistema dice "NO SE PUEDE VERIFICAR"?

- No hay documentos relevantes en la base de datos
- La afirmación es demasiado vaga o ambigua
- El tema no está cubierto en el corpus

**Solución**: Añade más documentos relacionados con el tema

### ¿Cómo mejoro la precisión?

1. Aumentar el corpus documental
2. Ajustar parámetros de chunking
3. Aumentar `k` en búsqueda vectorial
4. Usar un modelo de embeddings más potente
5. Optimizar los prompts

### ¿Funciona sin conexión a internet?

Sí, excepto para:
- Descargar modelos inicialmente
- Traducción (usa Google Translator)
- Descargar artículos de Wikipedia

Para uso offline completo, considera usar modelos de traducción locales.

### ¿Puedo usar otros LLMs?

Sí, edita `config.yaml` y cambia el modelo. Opciones:

- Ollama: llama3.1, qwen3, gemma3
- OpenAI: gpt-4, gpt-3.5-turbo (requiere API key)
- HuggingFace: cualquier modelo compatible

---

**¿Preguntas?** Consulta la documentación completa o abre un issue.

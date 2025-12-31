# 🔍 Sistema de Verificación de Hechos con RAG

> **Proyecto Final - Procesamiento del Lenguaje Natural**  
> Máster en Inteligencia Artificial Aplicada - UC3M  
> Curso 2025/2026

---

## 🚀 Inicio Rápido

### Opción 1: Docker (Recomendado - Todo en Contenedores)

```bash
make all
```

**🐳 Docker se instalará automáticamente en Ubuntu 24.04** si no lo tienes.

Ese único comando:
- ✅ Verifica e instala Docker si es necesario
- ✅ Ingiere los datos a ChromaDB
- ✅ Construye las imágenes Docker (backend + frontend)
- ✅ Inicia ambos servicios en contenedores
- ✅ Backend: http://localhost:8000
- ✅ Frontend: http://localhost:5174

**Detener los contenedores:**
```bash
make docker-down
```

**Ver logs:**
```bash
docker logs -f factchecker-backend
docker logs -f factchecker-frontend
```

### Opción 2: Desarrollo Local (Sin Docker)

```bash
make dev
```

**IMPORTANTE - Dependencias del Sistema (solo para desarrollo local)**

En Ubuntu/Linux:
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip nodejs npm -y
```

En Windows:
- Python 3.12: https://www.python.org/downloads/
- Node.js 18+: https://nodejs.org/

Ese único comando:
- ✅ Instala todas las dependencias (Python + Node.js)
- ✅ Descarga modelos necesarios (spaCy)
- ✅ Ingiere los datos a la base vectorial ChromaDB
- ✅ Inicia el backend con hot-reload en http://localhost:8000
- ✅ Inicia el frontend con hot-reload en http://localhost:5174

**📌 Importante**: El frontend se ejecuta en el puerto **5174** (configurado en Vite).

**🐧 Ubuntu 24.04**: Ver [UBUNTU_SETUP.md](UBUNTU_SETUP.md) para troubleshooting detallado.

### Opción 3: Windows (sin Make)

Ejecuta el script de inicio:
```bash
.\start.bat
```

O manualmente en dos terminales:

**Terminal 1 - Backend:**
```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
python test.py --clear
uvicorn api.server:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## 📋 Descripción General

Sistema inteligente de **verificación automática de hechos** basado en **Retrieval-Augmented Generation (RAG)** que analiza afirmaciones y determina su veracidad utilizando una base documental específica sobre equipos de fútbol de la Comunidad de Madrid.

El sistema combina técnicas avanzadas de NLP:
- **Búsqueda semántica vectorial** con ChromaDB
- **Reranking** con modelos cross-encoder
- **Generación aumentada** con LLM (Ollama/OpenAI)
- **Procesamiento multilingüe** con traducción automática

### ✨ Características Principales

- ✅ **Verificación precisa**: Clasifica afirmaciones como `VERDADERO`, `FALSO` o `NO SE PUEDE VERIFICAR`
- 🌍 **Soporte multilingüe**: Acepta consultas en español, inglés, francés, alemán, italiano y portugués
- 📚 **Base vectorial ChromaDB**: Almacenamiento y recuperación eficiente con embeddings semánticos
- 🎯 **Reranking inteligente**: Mejora la relevancia con modelos de cross-encoder
- 💾 **Sistema de caché**: Optimiza consultas repetidas para respuestas instantáneas
- 📖 **Citaciones precisas**: Referencias exactas con documento fuente y ubicación
- 📊 **Nivel de confianza**: Puntuación de certeza del veredicto (0-5 estrellas)
- 🔍 **Logging detallado**: Registro completo de operaciones para debugging
- 🎨 **Interfaz web moderna**: Frontend React + Tailwind CSS con experiencia de usuario optimizada

## 🎯 Dominio del Conocimiento

El sistema está especializado en verificar hechos sobre los **equipos de fútbol de la Comunidad de Madrid**:

- ⚪ **Real Madrid CF** - Historia, palmarés, jugadores
- 🔴 **Atlético de Madrid** - Trayectoria, títulos, estadio
- 🔵 **Getafe CF** - Logros, historia reciente
- 💙 **CD Leganés** - Historia, ascensos y descensos
- ⚡ **Rayo Vallecano** - Características, cultura del club

**Base documental**: 11 archivos de texto con información extraída de Wikipedia (actualizada a 2024).

## 🏗️ Arquitectura del Sistema

### 🔄 Flujo de Verificación

```
┌─────────────────────────────────────────────────────────────┐
│  1. ENTRADA DEL USUARIO                                     │
│     "El Real Madrid ganó su primera Champions en 1956"     │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  2. PROCESAMIENTO MULTILINGÜE                               │
│     • Detección automática de idioma (FastText)            │
│     • Traducción a español si es necesario                 │
│     • Normalización y limpieza de texto                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  3. RECUPERACIÓN SEMÁNTICA (RAG)                            │
│     ┌───────────────────────────────────────────────────┐  │
│     │ 3.1 Búsqueda Vectorial (ChromaDB)                 │  │
│     │     • Embeddings: paraphrase-multilingual-mpnet   │  │
│     │     • Similitud coseno                            │  │
│     │     • Top-k = 50 documentos candidatos            │  │
│     └─────────────────┬─────────────────────────────────┘  │
│                       │                                     │
│     ┌─────────────────▼─────────────────────────────────┐  │
│     │ 3.2 Reranking (Cross-Encoder)                     │  │
│     │     • Modelo: ms-marco-MiniLM-L-6-v2              │  │
│     │     • Refinamiento de relevancia                  │  │
│     │     • Top-k = 5 documentos finales                │  │
│     └─────────────────┬─────────────────────────────────┘  │
└───────────────────────┼─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  4. GENERACIÓN CON LLM                                      │
│     • Prompt few-shot con ejemplos                         │
│     • Contexto: Top-5 documentos más relevantes           │
│     • Razonamiento estructurado                            │
│     • Generación JSON con veredicto + explicación         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  5. RESPUESTA ESTRUCTURADA                                  │
│     {                                                       │
│       "veredicto": "VERDADERO",                            │
│       "explicacion": "El Real Madrid ganó...",             │
│       "confianza": 5,                                       │
│       "fuentes": ["Real_Madrid_Club_de_Futbol.txt"],       │
│       "idioma_original": "es"                              │
│     }                                                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  6. TRADUCCIÓN DE RESPUESTA (si aplica)                    │
│     • Traducción al idioma original del usuario            │
│     • Preservación de estructura y fuentes                 │
└─────────────────────────────────────────────────────────────┘
```

### 🧩 Componentes del Sistema

#### **1. Frontend (React + TypeScript + Vite)**
- **Ubicación**: `frontend/`
- **Tecnologías**: React 18, TypeScript, Tailwind CSS, Vite
- **Puerto**: `http://localhost:5174`
- **Funcionalidades**:
  - Interfaz de chat interactiva
  - Entrada de afirmaciones
  - Visualización de veredictos con badges dinámicos
  - Mostrado de fuentes y nivel de confianza
  - Diseño responsive y moderno

#### **2. Backend (FastAPI)**
- **Ubicación**: `api/server.py`
- **Puerto**: `http://localhost:8000`
- **Endpoints**:
  - `POST /verify` - Verificación de afirmaciones
  - `GET /health` - Estado del servicio
  - `GET /stats` - Estadísticas del sistema
- **CORS**: Configurado para desarrollo local

#### **3. Verificador de Hechos (FactChecker)**
- **Ubicación**: `verifier/verifier.py`
- **Funcionalidades**:
  - Gestión de pipeline completo de verificación
  - Integración con ChromaDB
  - Reranking con cross-encoder
  - Generación con LLM (Ollama/OpenAI)
  - Sistema de caché de consultas
  - Logging detallado

#### **4. Procesador Multilingüe**
- **Ubicación**: `language/pipeline_idioma.py`
- **Idiomas soportados**: ES, EN, FR, DE, IT, PT
- **Funcionalidades**:
  - Detección automática con FastText
  - Traducción bidireccional con Deep Translator
  - Validación de calidad de traducción

#### **5. Sistema de Recuperación**
- **Ubicación**: `retriever/`
- **Componentes**:
  - `advanced_retriever.py` - Pipeline avanzado con reranking
  - `rag_retriever.py` - Recuperación básica RAG
- **Técnicas**:
  - Búsqueda vectorial semántica
  - Reranking con cross-encoder
  - Filtrado por relevancia

#### **6. Base de Datos Vectorial**
- **Tecnología**: ChromaDB
- **Embeddings**: `paraphrase-multilingual-mpnet-base-v2` (Sentence Transformers)
- **Dimensión**: 768
- **Almacenamiento**: `data/vector_store/`

#### **7. Procesamiento de Documentos**
- **Ubicación**: `preprocessor/`, `chunker/`
- **Estrategias de chunking**:
  - Chunking semántico con spaCy
  - Chunking híbrido (fijo + semántico)
  - Chunk size: ~500 tokens con overlap de 50
- **Metadatos**: Extracción de entidades, fechas, equipos

#### **8. Ingesta de Datos**
- **Script**: `ingest/ingest_data.py`
- **Proceso**:
  1. Carga documentos desde `data/raw/`
  2. Preprocesamiento y limpieza
  3. Chunking inteligente
  4. Generación de embeddings
  5. Almacenamiento en ChromaDB

## 🚀 Instalación y Ejecución

### Requisitos Previos

- **Python**: 3.9 o superior
- **Node.js**: 16 o superior
- **RAM**: 8GB mínimo (16GB recomendado)
- **Ollama** (opcional): Para usar LLM local
- **OpenAI API Key** (opcional): Para usar GPT-4

### ⚡ Ejecución Rápida con Makefile (Recomendado)

El proyecto incluye un **Makefile** que automatiza todo el proceso:

```bash
# Ejecutar TODO el proyecto (instalar, ingerir, iniciar)
make all
```

Este comando:
1. ✅ Instala todas las dependencias de Python
2. ✅ Descarga el modelo de spaCy (`es_core_news_sm`)
3. ✅ Instala dependencias de Node.js (frontend)
4. ✅ Ingiere los datos a la base vectorial ChromaDB
5. ✅ Inicia el backend (API) en `http://localhost:8000`
6. ✅ Inicia el frontend en `http://localhost:5174`

**¡Y listo!** Abre tu navegador en `http://localhost:5174` para usar el sistema.

### 📚 Otros Comandos Útiles del Makefile

```bash
# Ver todos los comandos disponibles
make help

# Solo instalar dependencias
make install

# Solo ingerir datos (si ya los tienes instalados)
make ingest

# Iniciar solo el backend
make backend

# Iniciar solo el frontend
make frontend

# Limpiar archivos temporales y caché
make clean

# Resetear todo (limpiar, instalar e ingerir)
make reset

# Desarrollo sin Docker
make dev
```

### 🐳 Ejecución con Docker (Alternativa)

```bash
# Construir imagen y levantar contenedores
make docker-build
make docker-up

# En otra terminal, iniciar el frontend
make frontend

# Detener contenedores
make docker-down
```

### 🔧 Instalación Manual (Sin Makefile)

Si prefieres hacerlo paso a paso:

#### 1. Instalar dependencias de Python

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

#### 2. Instalar dependencias del frontend

```bash
cd frontend
npm install
cd ..
```

#### 3. Ingerir datos a ChromaDB

```bash
python test.py --clear
```

#### 4. Iniciar el backend (en una terminal)

```bash
uvicorn api.server:app --reload --port 8000
```

#### 5. Iniciar el frontend (en otra terminal)

```bash
cd frontend
npm run dev
```

#### 6. Abrir en el navegador

Accede a: `http://localhost:5174`

## 💡 Explicación del Código

### Componentes Clave

#### 1. **FactChecker** (`verifier/verifier.py`)

Clase principal que orquesta todo el pipeline de verificación:

```python
class FactChecker:
    def __init__(self, config_path: str = "config.yaml"):
        # Carga configuración desde YAML
        self.config = ConfigManager(config_path)
        
        # Inicializa procesador multilingüe
        self.linguist = PipelineIdioma()
        
        # Carga modelo de embeddings para búsqueda semántica
        self.embeddings = HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-mpnet-base-v2"
        )
        
        # Conecta a la base vectorial ChromaDB
        self.vector_db = Chroma(
            persist_directory="data/vector_store",
            embedding_function=self.embeddings
        )
        
        # Inicializa modelo de reranking
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        # Inicializa LLM (Ollama o OpenAI)
        self.llm = ChatOllama(model="llama3.1", temperature=0.0)
        
    def verify(self, claim_usuario: str) -> Dict[str, Any]:
        """Pipeline completo de verificación"""
        # 1. Detectar idioma y traducir si es necesario
        idioma = self.linguist.detect_language(claim_usuario)
        claim_es = self.linguist.translate(claim_usuario, to_lang="es")
        
        # 2. Búsqueda semántica en ChromaDB (top-50)
        docs = self.vector_db.similarity_search(claim_es, k=50)
        
        # 3. Reranking con cross-encoder (top-5)
        docs_reranked = self.reranker.rank(claim_es, docs, top_k=5)
        
        # 4. Construir contexto
        context = "\n\n".join([doc.page_content for doc in docs_reranked])
        
        # 5. Generar prompt few-shot
        prompt = self.prompts.format(claim=claim_es, context=context)
        
        # 6. Invocar LLM para obtener veredicto
        response = self.llm.invoke(prompt)
        resultado = json.loads(response.content)
        
        # 7. Traducir respuesta al idioma original
        if idioma != "es":
            resultado["explicacion"] = self.linguist.translate(
                resultado["explicacion"], to_lang=idioma
            )
        
        return resultado
```

**Flujo paso a paso**:
1. **Detección de idioma**: FastText identifica el idioma de entrada
2. **Traducción**: Si no es español, traduce a español para búsqueda
3. **Búsqueda vectorial**: ChromaDB recupera los 50 chunks más similares
4. **Reranking**: Cross-encoder refina a los 5 más relevantes
5. **Construcción de contexto**: Concatena los documentos seleccionados
6. **Generación LLM**: Llama al modelo con prompt few-shot
7. **Post-procesamiento**: Traduce la respuesta al idioma original

#### 2. **API FastAPI** (`api/server.py`)

Servidor HTTP que expone el verificador:

```python
from fastapi import FastAPI
from verifier import FactChecker

app = FastAPI(title="FactChecker API")
fact_checker = FactChecker()  # Instancia global reutilizable

@app.post("/verify")
async def verify_claim(request: VerifyRequest):
    """Endpoint principal de verificación"""
    result = fact_checker.verify(request.question)
    return {
        "verdict": _map_verdict_tag(result["veredicto"]),
        "explanation": result["explicacion_corta"],
        "confidence": result["nivel_confianza"],
        "sources": result.get("fuentes", []),
        "language": result.get("idioma_respuesta", "es")
    }
```

**Características**:
- CORS configurado para permitir frontend en localhost:5174
- Instancia global de FactChecker (evita recargar modelos en cada request)
- Transformación de respuesta a formato JSON estándar

#### 3. **Ingesta de Datos** (`ingest/ingest_data.py`)

Script que prepara la base vectorial:

```python
def main():
    # 1. Cargar documentos desde data/raw/
    loader = DirectoryLoader("data/raw/", glob="**/*.txt")
    documents = loader.load()
    
    # 2. Preprocesar (limpiar, normalizar)
    preprocessor = DocumentPreprocessor()
    docs_clean = [preprocessor.clean(doc) for doc in documents]
    
    # 3. Chunking semántico con spaCy
    chunker = SemanticChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split_documents(docs_clean)
    
    # 4. Extraer metadatos (equipos, fechas, entidades)
    extractor = MetadataExtractor()
    chunks_with_metadata = [
        extractor.extract(chunk) for chunk in chunks
    ]
    
    # 5. Generar embeddings y guardar en ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="...")
    db = Chroma.from_documents(
        documents=chunks_with_metadata,
        embedding=embeddings,
        persist_directory="data/vector_store/"
    )
    db.persist()
```

**Proceso**:
1. **Carga**: Lee todos los archivos .txt de `data/raw/`
2. **Preprocesamiento**: Elimina caracteres especiales, normaliza espacios
3. **Chunking**: Divide en fragmentos semánticamente coherentes (~500 tokens)
4. **Metadatos**: Extrae equipos mencionados, fechas, nombres propios
5. **Embeddings**: Genera vectores de 768 dimensiones con Sentence Transformers
6. **Persistencia**: Guarda todo en ChromaDB para búsqueda rápida

#### 4. **Pipeline Multilingüe** (`language/pipeline_idioma.py`)

Gestiona detección y traducción:

```python
class PipelineIdioma:
    SUPPORTED_LANGS = ["es", "en", "fr", "de", "it", "pt"]
    
    def __init__(self):
        # Cargar modelo FastText para detección
        self.detector = fasttext.load_model("data/lid.176.ftz")
        
    def detect_language(self, text: str) -> str:
        """Detecta idioma con FastText"""
        predictions = self.detector.predict(text, k=1)
        lang_code = predictions[0][0].replace("__label__", "")
        return lang_code if lang_code in self.SUPPORTED_LANGS else "es"
    
    def translate(self, text: str, to_lang: str) -> str:
        """Traduce con Google Translator"""
        translator = GoogleTranslator(source='auto', target=to_lang)
        return translator.translate(text)
```

**Características**:
- FastText: 176 idiomas soportados, detección ultrarrápida
- Google Translator: Traducción de alta calidad
- Validación: Solo acepta idiomas en whitelist

#### 5. **Frontend React** (`frontend/src/`)

Interfaz web moderna:

```typescript
// Componente principal de verificación
function VerifyForm() {
  const [claim, setClaim] = useState("");
  const [result, setResult] = useState(null);
  
  const handleSubmit = async () => {
    const response = await fetch("http://localhost:8000/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: claim })
    });
    
    const data = await response.json();
    setResult(data);
  };
  
  return (
    <div className="max-w-4xl mx-auto p-6">
      <textarea 
        value={claim}
        onChange={(e) => setClaim(e.target.value)}
        placeholder="Escribe una afirmación..."
      />
      <button onClick={handleSubmit}>Verificar</button>
      
      {result && (
        <ResultCard 
          verdict={result.verdict}
          explanation={result.explanation}
          sources={result.sources}
          confidence={result.confidence}
        />
      )}
    </div>
  );
}
```

**Tecnologías**:
- **React 18**: Componentes funcionales con hooks
- **TypeScript**: Tipado estático para robustez
- **Tailwind CSS**: Estilos utility-first
- **Vite**: Build tool ultrarrápido

### Archivo de Configuración (`config.yaml`)

```yaml
# Modelos
embeddings:
  model_name: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  device: "cpu"  # o "cuda" si tienes GPU

llm:
  provider: "ollama"  # o "openai"
  model: "llama3.1"
  temperature: 0.0    # Determinista
  max_tokens: 2000

# RAG
retriever:
  k: 50              # Búsqueda inicial
  top_k: 5           # Después de reranking
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking
chunker:
  strategy: "semantic"  # o "hybrid", "fixed"
  chunk_size: 500
  chunk_overlap: 50

# Logging
logging:
  level: "INFO"
  file: "logs/factchecker.log"
```

Este archivo centraliza toda la configuración, permitiendo cambiar modelos y parámetros sin modificar código.

## 📦 Estructura del Proyecto

```
NLP-VERIFICAR_DE_HECHOS-v3/
│
├── 📄 Makefile                    # Automatización de tareas
├── 📄 Dockerfile                  # Contenedor Docker
├── 📄 requirements.txt            # Dependencias Python
├── 📄 config.yaml                 # Configuración global
├── 📄 README.md                   # Este archivo
│
├── 📁 api/                        # API REST FastAPI
│   └── server.py                  # Servidor HTTP
│
├── 📁 verifier/                   # Motor de verificación
│   ├── verifier.py                # FactChecker principal
│   └── prompts.py                 # Templates de prompts
│
├── 📁 retriever/                  # Sistema de recuperación
│   ├── advanced_retriever.py     # Pipeline con reranking
│   ├── rag_retriever.py           # RAG básico
│   └── hyde_retriever.py          # HyDE (Hypothetical Doc Embeddings)
│
├── 📁 language/                   # Procesamiento multilingüe
│   └── pipeline_idioma.py         # Detección y traducción
│
├── 📁 preprocessor/               # Preprocesamiento de textos
│   └── document_preprocessor.py  # Limpieza y normalización
│
├── 📁 chunker/                    # Estrategias de chunking
│   ├── semantic_chunker.py        # Chunking semántico
│   ├── hybrid_chunker.py          # Chunking híbrido
│   └── section_aware.py           # Consciente de secciones
│
├── 📁 extractor/                  # Extracción de metadatos
│   ├── metadata_extractor.py     # Metadatos generales
│   ├── fact_metadata_extractor.py # Metadatos de hechos
│   └── topic_extractor.py         # Extracción de tópicos
│
├── 📁 ingest/                     # Ingesta de documentos
│   └── ingest_data.py             # Pipeline de ingesta
│
├── 📁 frontend/                   # Interfaz web
│   ├── src/                       # Código fuente React
│   ├── package.json               # Dependencias Node.js
│   ├── vite.config.ts             # Configuración Vite
│   └── tailwind.config.js         # Configuración Tailwind
│
├── 📁 data/                       # Datos del sistema
│   ├── raw/                       # Documentos originales (11 archivos .txt)
│   ├── vector_store/              # Base vectorial ChromaDB
│   └── lid.176.ftz                # Modelo FastText para detección de idioma
│
├── 📁 logs/                       # Logs del sistema
├── 📁 evaluations/                # Resultados de evaluaciones
└── 📁 test_parts/                 # Tests unitarios
```

## 🎮 Uso del Sistema

### 💬 Interfaz Web

1. Accede a `http://localhost:5174`
2. Escribe una afirmación en el campo de texto
3. Presiona Enter o haz clic en "Verificar"
4. Observa el veredicto, explicación y fuentes

**Ejemplos de afirmaciones validadas** (85.7% de precisión en evaluación):

```
✅ VERDADERO - Casos verificados correctamente:
- "El Real Madrid fue fundado en 1902"
- "El estadio del Real Madrid se llama Santiago Bernabéu"
- "El Real Madrid ha ganado 15 Copas de Europa"
- "El Atlético de Madrid ganó la Liga en la temporada 2020-21"
- "El Getafe CF juega en el Coliseum Alfonso Pérez"
- "El CD Leganés fue fundado en 1928"
- "El Rayo Vallecano juega en Vallecas"

❌ FALSO - Detección de falsedades:
- "El Real Madrid fue fundado en 1947" (Fundado en 1902, no 1947)
- "El Atlético de Madrid nunca ha ganado la Liga" (Ha ganado 11 veces)
- "El CD Leganés fue fundado en 1900" (Fundado en 1928, no 1900)

🔍 NO VERIFICABLE - Predicciones y afirmaciones fuera de alcance:
- "El Real Madrid ganará la Champions League en 2025" (Predicción futura)
- "Messi es el mejor jugador de la historia" (Opinión subjetiva)
- "El Barcelona es el mejor equipo de España" (Fuera del corpus de Madrid)
```

### 🔌 API REST

#### **Verificar una afirmación**

```bash
curl -X POST "http://localhost:8000/api/verify" \
  -H "Content-Type: application/json" \
  -d '{"question": "El Real Madrid fue fundado en 1902"}'
```

**Respuesta**:

```json
{
  "verdict": "true",
  "explanation": "La afirmación es VERDADERA. Confirma la fundación en 1902. El Real Madrid fue registrado oficialmente como club de fútbol el 6 de marzo de 1902 en una Junta General Extraordinaria.",
  "confidence": 3,
  "sources": [
    {
      "document": "Historia_del_Real_Madrid_Club_de_Fútbol.txt",
      "snippet": "legalizaron oficialmente la nueva asociación el 6 de marzo de 1902 en una Junta General Extraordinaria"
    }
  ],
  "language": "es",
  "retrieval_time_ms": 234,
  "llm_time_ms": 6722
}
```

#### **Estado del servicio**

```bash
curl http://localhost:8000/health
```

#### **Estadísticas**

```bash
curl http://localhost:8000/stats
```

### 🐍 Uso Programático (Python)

Puedes usar el verificador directamente en tus scripts Python:

```python
from verifier.verifier import FactChecker

# Inicializar verificador (carga modelos una sola vez)
fact_checker = FactChecker()

# Ejemplos validados (85.7% de precisión)
test_cases = [
    "El Real Madrid fue fundado en 1902",
    "El Atlético de Madrid ganó la Liga en la temporada 2020-21",
    "El Getafe CF juega en el Coliseum Alfonso Pérez",
    "El CD Leganés fue fundado en 1928",
    "El Rayo Vallecano juega en Vallecas",
]

for claim in test_cases:
    result = fact_checker.verify(claim)
    print(f"Afirmación: {claim}")
    print(f"Veredicto: {result['veredicto']}")
    print(f"Confianza: {result['nivel_confianza']}/5")
    print(f"Explicación: {result['explicacion_corta']}")
    print(f"Fuente: {result['fuente_documento']}")
    print("-" * 80)
```

**Salida esperada**:

```
Afirmación: El Real Madrid fue fundado en 1902
Veredicto: VERDADERO
Confianza: 3/5
Explicación: Confirma la fundación en 1902
Fuente: Historia_del_Real_Madrid_Club_de_Fútbol.txt
--------------------------------------------------------------------------------
Afirmación: El Atlético de Madrid ganó la Liga en la temporada 2020-21
Veredicto: VERDADERO
Confianza: 4/5
Explicación: Confirma que ganó la Liga en 2020-21
Fuente: Anexo-Palmarés_del_Club_Atlético_de_Madrid.txt
--------------------------------------------------------------------------------
...
```

## ⚙️ Configuración

### Archivo `config.yaml`

El sistema se configura mediante `config.yaml`:

```yaml
# Modelo de embeddings
embeddings:
  model_name: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  device: "cpu"  # o "cuda" si tienes GPU

# LLM (Ollama o OpenAI)
llm:
  provider: "ollama"  # o "openai"
  model: "llama3.1"   # o "gpt-4"
  temperature: 0.0
  max_tokens: 2000

# Recuperación
retriever:
  k: 50              # Documentos iniciales
  top_k: 5           # Documentos finales tras reranking
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking
chunker:
  strategy: "semantic"  # o "hybrid", "fixed"
  chunk_size: 500
  chunk_overlap: 50

# Logging
logging:
  level: "INFO"
  file: "logs/factchecker.log"
```

### Variables de Entorno

Para usar OpenAI:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."

# Linux/Mac
export OPENAI_API_KEY="sk-..."
```

## 🧪 Testing y Evaluación

### Ejecutar tests

```bash
# Test completo del verificador
python test_verifier_simple.py

# Test de embeddings
python test_embedding_upgrade.py

# Test de recuperación
python test_retrieval_debug.py

# Test de confianza semántica
python test_confidence.py
```

### Evaluación con métricas

```bash
python evaluate.py
```

Métricas implementadas:
- **Exactitud**: % de veredictos correctos
- **Precisión/Recall**: Por clase (VERDADERO/FALSO/NO VERIFICABLE)
- **BERT Score**: Similitud semántica de explicaciones
- **ROUGE**: Calidad de generación de texto
- **Latencia**: Tiempo de respuesta

## 📊 Requisitos Funcionales Cumplidos

Según el enunciado del proyecto, el sistema cumple con:

### ✅ Requisitos Obligatorios

1. **Respuesta clara sobre veracidad**  
   ✓ Veredictos explícitos: `VERDADERO`, `FALSO`, `NO SE PUEDE VERIFICAR`

2. **Citación de fuentes precisas**  
   ✓ Nombre del documento fuente  
   ✓ Ubicación específica (sección/página)  
   ✓ Snippet de evidencia citado

3. **Manejo de información insuficiente**  
   ✓ Responde "NO SE PUEDE VERIFICAR" cuando no hay evidencia  
   ✓ No inventa información

4. **Respuesta en idioma original**  
   ✓ Detección automática del idioma de entrada  
   ✓ Traducción de respuesta al idioma detectado

5. **Base documental específica**  
   ✓ 11 documentos sobre equipos de fútbol de Madrid  
   ✓ Información actualizada (Wikipedia 2024)

### ✨ Características Adicionales

- 🔄 **Reranking** con cross-encoder para mejor precisión
- 💾 **Sistema de caché** para optimización
- 📊 **Nivel de confianza** cuantificado (0-5)
- 🎯 **Chunking semántico** inteligente
- 🌍 **6 idiomas** soportados
- 🎨 **Interfaz web** moderna y responsive
- 🔍 **Logging detallado** para debugging
- 📈 **Métricas de evaluación** completas

## 🛠️ Tecnologías Utilizadas

### Backend
- **Python 3.11** - Lenguaje principal
- **FastAPI** - Framework web
- **LangChain** - Orquestación RAG
- **ChromaDB** - Base de datos vectorial
- **Sentence Transformers** - Embeddings
- **Ollama / OpenAI** - Modelos LLM
- **spaCy** - NLP y chunking semántico
- **FastText** - Detección de idioma
- **Deep Translator** - Traducción

### Frontend
- **React 18** - Biblioteca UI
- **TypeScript** - Tipado estático
- **Vite** - Build tool
- **Tailwind CSS** - Estilos

### DevOps
- **Docker** - Contenedores
- **Make** - Automatización
- **Git** - Control de versiones

## 📝 Logging y Debugging

El sistema genera logs detallados en `logs/`:

```
logs/
├── factchecker.log       # Log principal
├── retriever.log         # Logs de recuperación
├── llm.log               # Logs del LLM
└── api.log               # Logs de la API
```

Nivel de detalle configurable en `config.yaml`:
- `DEBUG`: Todo el detalle (desarrollo)
- `INFO`: Información general (recomendado)
- `WARNING`: Solo advertencias
- `ERROR`: Solo errores

## 🐛 Solución de Problemas

### El sistema no encuentra documentos

```bash
# Re-ingestar la base de datos
python test.py --clear
```

### Error de memoria

- Reducir `retriever.k` en `config.yaml`
- Usar embeddings más pequeños
- Aumentar RAM disponible

### Frontend no conecta con backend

- Verificar que el backend esté en `http://localhost:8000`
- Revisar CORS en `api/server.py`
- Verificar puertos en uso

### Frontend no inicia en Ubuntu (vite: Permission denied)

```bash
# Solución: Dar permisos al ejecutable de vite
cd frontend
chmod +x node_modules/.bin/vite
npm run dev
```

**Alternativa**: Reinstalar dependencias del frontend
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Ollama no responde

```bash
# Verificar que Ollama esté corriendo
ollama list

# Iniciar Ollama
ollama serve

# Descargar modelo
ollama pull llama3.1
```

## 📚 Referencias y Documentación

- **LangChain**: https://python.langchain.com/
- **ChromaDB**: https://docs.trychroma.com/
- **Sentence Transformers**: https://www.sbert.net/
- **FastAPI**: https://fastapi.tiangolo.com/
- **React**: https://react.dev/
- **Ollama**: https://ollama.ai/

## 👥 Autores

**Proyecto Final - Máster IA Aplicada UC3M**  
Curso 2024/2025

## 📄 Licencia

Este proyecto es con fines académicos.

---

**🎯 ¡Listo para usar!** Ejecuta `make all` y comienza a verificar hechos.


# Análisis Profundo: Problema del RAG y Soluciones Gratuitas

## 🔬 Diagnóstico del Problema Real

### **¿Por qué el RAG no funciona para fact-checking?**

Analizando los logs y resultados:

```
CLAIM: "fundado en 1903"
RECUPERA: Documentos sobre 1940-1941, estadios, etc.
NO RECUPERA: "fue registrada oficialmente... el 6 de marzo de 1902"
```

**Causas raíces identificadas:**

1. **Embeddings semánticos son "literales"**
   - Buscan matching semántico directo
   - "fundado 1903" → busca chunks que mencionen "1903"
   - NO buscan "contradicciones" o "hechos relacionados pero diferentes"

2. **Chunks son demasiado largos** (1000 caracteres)
   - Un chunk puede contener múltiples hechos/fechas
   - Dilución de la señal: "1902" se pierde entre otras 100 palabras
   - El embedding promedia todo el contenido

3. **El modelo de embeddings no entiende "relaciones numéricas"**
   - Para el embedding, "1902" y "1903" son casi idénticos
   - No entiende que son fechas contradictorias
   - Son solo tokens similares

4. **Falta de indexación por entidades específicas**
   - No hay índice de "hechos verificables"
   - No hay estructura: SUJETO → PREDICADO → OBJETO → FECHA

---

## ✅ SOLUCIONES PRÁCTICAS (100% Gratuitas)

### **SOLUCIÓN 1: Dual-Index RAG (Recomendada)**

Crear **DOS índices diferentes** para búsqueda:

#### **Índice A: Embeddings Semánticos** (ya existe)
- Para comprensión general del tema
- Chunks de 1000 caracteres

#### **Índice B: Índice de Hechos con BM25** (NUEVO)
- Búsqueda keyword-based ultra rápida
- Indexa "hechos atómicos": entidad + acción + fecha
- Gratuito: usa Tantivy o Whoosh (Python puro)

**Ejemplo de hechos indexados:**
```json
{
  "entidad": "Real Madrid",
  "accion": "fundado",
  "fecha": "1902",
  "texto": "fue registrada oficialmente como club de fútbol por sus socios el 6 de marzo de 1902"
}
```

**Ventaja**: Búsqueda exacta por keywords + fechas
- Query: "Real Madrid fundado 1903"
- BM25 encuentra: TODOS los docs con "Real Madrid" + "fundado" + cualquier fecha
- Incluye el doc con 1902 → LLM puede contradecir

---

### **SOLUCIÓN 2: Propositions/Atomic Facts Chunking** (MÁS IMPACTO)

En lugar de chunks de 1000 caracteres, crear **"proposiciones atómicas"**:

**Chunk actual (1000 chars)**:
```
"El Real Madrid Club de Fútbol, más conocido simplemente como Real Madrid, 
es una entidad polideportiva con sede en Madrid, España. Fue registrada 
oficialmente como club de fútbol por sus socios el 6 de marzo de 1902 con 
el objeto de la práctica y desarrollo de este deporte —si bien sus orígenes 
datan del año 1900,​ y su denominación..."
```

**Proposiciones atómicas (nuevo)**:
```
P1: "Real Madrid es una entidad polideportiva con sede en Madrid, España"
P2: "Real Madrid fue registrada oficialmente el 6 de marzo de 1902"
P3: "Los orígenes del Real Madrid datan del año 1900"
P4: "La denominación (Sociedad) Madrid Foot-ball Club es de octubre de 1901"
```

**Ventaja**:
- Cada embedding representa UN HECHO específico
- "fundado 1903" → match con P2 (menciona fundación + fecha)
- Mucho más preciso para fact-checking

**Implementación gratuita:**
- Usar spaCy (ya instalado) para sentence splitting
- Regex para detectar "hechos verificables" (fecha + verbo)
- O usar LLM local (Ollama llama3.2) para extraer proposiciones

---

### **SOLUCIÓN 3: Metadata-Rich Indexing** (MÁS SIMPLE)

Enriquecer los metadatos de cada chunk con **entidades extraídas**:

**Metadata actual:**
```python
{
  "source": "Real_Madrid.txt",
  "chunk_id": 1,
  "chunk_size": 1000
}
```

**Metadata mejorada:**
```python
{
  "source": "Real_Madrid.txt",
  "chunk_id": 1,
  "chunk_size": 1000,
  "entidades": ["Real Madrid", "Madrid", "España"],
  "fechas": ["1902", "1900", "1901"],
  "hechos_clave": ["fundación: 1902", "orígenes: 1900"],
  "verbos_accion": ["registrada", "datan"]
}
```

**Búsqueda mejorada:**
1. Query: "Real Madrid fundado 1903"
2. Extraer: entidad="Real Madrid", fecha="1903", verbo="fundado"
3. Filtrar chunks por metadata:
   - chunks con entidad="Real Madrid" AND "fundación" in hechos_clave
4. Recuperar TODOS (incluye el de 1902)

**Implementación:**
- Usar spaCy para NER (entidades)
- Regex para fechas
- Template matching para hechos

---

### **SOLUCIÓN 4: Query Decomposition + Multi-Retrieval**

Descomponer queries complejas en sub-queries:

**Query original:**
```
"El Real Madrid fue fundado en 1903"
```

**Descomposición:**
```
Q1: "¿Cuándo fue fundado el Real Madrid?"  ← Sin la fecha incorrecta
Q2: "Real Madrid fundación fecha"          ← Keywords genéricos
Q3: "Real Madrid 1903"                     ← Fecha específica (para verificar)
```

**Proceso:**
1. Recuperar docs con Q1 (sin fecha) → Encuentra doc con 1902
2. Recuperar docs con Q2 → Más docs sobre fundación  
3. Recuperar docs con Q3 → Verifica si existe 1903
4. Combinar resultados
5. LLM compara: "Los docs sobre fundación dicen 1902, no 1903"

**Implementación:**
- Usar LLM local (Ollama) para generar sub-queries
- Hacer múltiples retrievals
- Combinar con voting/ranking

---

## 📊 Comparación de Soluciones

| Solución | Complejidad | Impacto | Costo Computacional | Gratuito |
|----------|-------------|---------|---------------------|----------|
| **Dual-Index (BM25 + Embeddings)** | Media | ⭐⭐⭐⭐ | Bajo | ✅ Sí |
| **Atomic Facts Chunking** | Alta | ⭐⭐⭐⭐⭐ | Medio | ✅ Sí |
| **Metadata-Rich Indexing** | Baja | ⭐⭐⭐ | Bajo | ✅ Sí |
| **Query Decomposition** | Media | ⭐⭐⭐ | Medio | ✅ Sí |

---

## 🎯 RECOMENDACIÓN: Implementación por Fases

### **FASE 1: Quick Win (1-2 horas)**
**Metadata-Rich Indexing + Query Decomposition**

Razón:
- Fácil de implementar
- No requiere re-indexar todo
- Mejora inmediata del 30-40%

Implementar:
1. Extraer fechas/entidades en metadata al ingestar
2. Filtrar por metadata antes de similarity search
3. Generar 3 sub-queries por cada claim

### **FASE 2: Mejora Estructural (1-2 días)**
**Dual-Index: BM25 + Embeddings**

Razón:
- Balance perfecto: precisión + recall
- BM25 es rápido y gratuito
- Complementa embeddings semánticos

Implementar:
1. Instalar `whoosh` o `rank_bm25` (Python puro)
2. Indexar chunks con BM25
3. Búsqueda híbrida: 50% BM25 + 50% embeddings
4. Reciprocal Rank Fusion para combinar

### **FASE 3: Solución Ideal (3-5 días)**
**Atomic Facts Chunking**

Razón:
- Máxima precisión para fact-checking
- Cada embedding = 1 hecho verificable
- Solución a largo plazo

Implementar:
1. Usar Ollama llama3.2 para extraer proposiciones
2. Re-chunking del corpus en hechos atómicos
3. Re-indexar base de datos vectorial

---

## 💻 CÓDIGO: Implementación Fase 1 (Quick Win)

### **1. Metadata-Rich Extractor**

```python
import re
import spacy

class FactMetadataExtractor:
    """Extrae metadata rica para fact-checking."""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("es_core_news_sm")
        except:
            self.nlp = None
    
    def extract_dates(self, text: str) -> List[str]:
        """Extrae todas las fechas del texto."""
        # Regex para años
        years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', text)
        # Regex para fechas completas
        full_dates = re.findall(
            r'\b(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})\b',
            text,
            re.IGNORECASE
        )
        return list(set(years + full_dates))
    
    def extract_key_facts(self, text: str) -> List[str]:
        """Extrae hechos clave (acción + fecha)."""
        facts = []
        
        # Patrones para hechos verificables
        patterns = [
            r'(fundad[oa]|cread[oa]|establecid[oa]|registrad[oa])\s+.*?(\d{4})',
            r'(gan[óo]|consigui[óo]|logr[óo])\s+.*?(\d{4})',
            r'(naci[óo]|muri[óo])\s+.*?(\d{4})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                facts.append(f"{match[0]}: {match[1]}")
        
        return facts
    
    def enrich_metadata(self, doc: Document) -> Document:
        """Enriquece metadata de un documento."""
        text = doc.page_content
        
        # Extraer fechas
        dates = self.extract_dates(text)
        
        # Extraer entidades con spaCy
        entities = []
        if self.nlp:
            doc_nlp = self.nlp(text[:500])  # Solo primeros 500 chars
            entities = [ent.text for ent in doc_nlp.ents 
                       if ent.label_ in ['PER', 'ORG', 'LOC']]
        
        # Extraer hechos clave
        key_facts = self.extract_key_facts(text)
        
        # Agregar a metadata
        doc.metadata.update({
            'fechas': dates,
            'entidades': entities,
            'hechos_clave': key_facts,
            'tiene_fechas': len(dates) > 0,
            'es_sobre_fundacion': any(
                word in text.lower() 
                for word in ['fundado', 'fundación', 'registrado', 'creado']
            )
        })
        
        return doc
```

### **2. Query Decomposer**

```python
from langchain_ollama import ChatOllama

class QueryDecomposer:
    """Descompone queries en sub-queries para mejor retrieval."""
    
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2", temperature=0.0)
    
    def decompose(self, query: str) -> List[str]:
        """Genera 3 sub-queries variadas."""
        
        # Extraer componentes manualmente
        import re
        
        # Extraer fecha
        dates = re.findall(r'\b\d{4}\b', query)
        
        # Extraer entidades principales
        entities = []
        for entity in ['Real Madrid', 'Barcelona', 'Atlético']:
            if entity.lower() in query.lower():
                entities.append(entity)
        
        # Extraer verbo de acción
        action_words = ['fundado', 'ganó', 'jugó', 'nació']
        action = None
        for word in action_words:
            if word in query.lower():
                action = word
                break
        
        # Generar sub-queries
        sub_queries = [query]  # Original siempre incluida
        
        if entities and action:
            # Sin fecha (CLAVE para encontrar info contradictoria)
            sub_queries.append(f"{entities[0]} {action}")
            
            # Solo keywords
            sub_queries.append(f"{entities[0]} {action} fecha")
        
        return sub_queries[:3]
```

### **3. Integrar en AdvancedRetriever**

```python
# En advanced_retriever.py, modificar retrieve():

def retrieve_with_metadata_filter(self, query: str) -> List[Document]:
    """Retrieval mejorado con filtros de metadata."""
    
    # 1. Extraer componentes de la query
    dates_in_query = re.findall(r'\b\d{4}\b', query)
    is_about_foundation = any(
        word in query.lower() 
        for word in ['fundado', 'fundación', 'creado']
    )
    
    # 2. Búsqueda vectorial normal
    docs = self.vector_db.similarity_search(query, k=100)
    
    # 3. PRE-FILTRO por metadata relevante
    if is_about_foundation:
        # Priorizar docs sobre fundación
        filtered = [
            doc for doc in docs
            if doc.metadata.get('es_sobre_fundacion', False)
        ]
        if filtered:
            docs = filtered[:50] + docs[:50]  # Combinar
    
    # 4. Si query tiene fecha, incluir docs con fechas cercanas
    if dates_in_query:
        query_year = int(dates_in_query[0])
        # Buscar docs con fechas en rango ±10 años
        date_relevant_docs = [
            doc for doc in docs
            if any(
                abs(int(date) - query_year) <= 10
                for date in doc.metadata.get('fechas', [])
                if date.isdigit()
            )
        ]
        if date_relevant_docs:
            docs = date_relevant_docs[:30] + docs[:30]
    
    return docs[:self.config.k_initial]
```

---

## 🚀 Plan de Acción Inmediato

### **HOY (2-3 horas):**

1. ✅ Implementar `FactMetadataExtractor`
2. ✅ Modificar `ingest_data.py` para usar el extractor
3. ✅ Re-ingestar el corpus con metadata rica
4. ✅ Probar si mejora el retrieval

### **MAÑANA (3-4 horas):**

1. ✅ Implementar `QueryDecomposer`
2. ✅ Modificar `AdvancedRetriever` para usar sub-queries
3. ✅ Implementar filtros de metadata en retrieval
4. ✅ Probar test completo

### **ESTA SEMANA (si hay tiempo):**

1. Investigar BM25 en Python (`rank_bm25` library)
2. Implementar dual-index
3. Comparar resultados

---

## 📚 Recursos Gratuitos

### **Para BM25:**
- `rank_bm25`: https://pypi.org/project/rank-bm25/
- Tutorial: https://www.pinecone.io/learn/bm25/

### **Para Atomic Facts:**
- Paper: "Enabling Large Language Models to Generate Text with Citations"
- Usar Ollama llama3.2 (ya instalado) para extracción

### **Para Query Decomposition:**
- DSPy framework (gratuito)
- Langchain tiene built-in query decomposition

---

## 🎯 Resultado Esperado Tras Fase 1:

**ANTES:**
```
"fundado 1903" → NO encuentra doc con 1902 → NO SE PUEDE VERIFICAR
```

**DESPUÉS:**
```
"fundado 1903" 
  → Query Decomp: ["Real Madrid fundado", "fundado 1903", "Real Madrid fundación fecha"]
  → Metadata Filter: chunks con es_sobre_fundacion=True
  → Encuentra doc con 1902
  → LLM: FALSO (dice 1902, no 1903)
```

**Mejora esperada: 50-70% en tests**

---

¿Quieres que implemente la **Fase 1** ahora? Es lo más rápido y efectivo.

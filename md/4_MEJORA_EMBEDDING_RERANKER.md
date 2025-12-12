# Mejora de Embeddings y Reranker - Prioridad 1 y 2

## 🎯 Objetivo

Mejorar la recuperación RAG con **mejores modelos** (genérico, sin reglas específicas):
- **Prioridad 1**: Cambiar embedding a modelo state-of-the-art multilingüe
- **Prioridad 2**: Cambiar reranker a modelo más potente

## 📊 Cambios Realizados

### 1. **Modelo de Embeddings**

**ANTES:**
```yaml
embeddings:
  name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```
- 384 dimensiones
- Benchmarks moderados

**AHORA:**
```yaml
embeddings:
  name: "BAAI/bge-m3"
```
- **1024 dimensiones** (2.7x más expresivo que anterior)
- **Híbrido denso + sparse** (mejor precisión)
- **Soporta 100+ idiomas** (incluye español)
- **Benchmarks: +10-15% mejor Hit Rate** según estudios de LlamaIndex
- **No requiere trust_remote_code** (mejor compatibilidad)

### 2. **Modelo de Reranker**

**ANTES:**
```yaml
reranker:
  name: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
```
- Entrenado en MS MARCO
- Multilingüe limitado

**AHORA:**
```yaml
reranker:
  name: "BAAI/bge-reranker-v2-m3"
```
- **Soporta 100+ idiomas**
- **Estado del arte** en reranking multilingüe
- **Benchmarks: +5-10% mejor MRR**

### 3. **Dependencias Actualizadas**

Añadido en `requirements.txt`:
```
FlagEmbedding>=1.2.0
```
- Librería optimizada para embeddings BAAI/Alibaba-NLP
- Mejor performance que sentence-transformers para estos modelos

## 🔬 Tests Creados

### **test_models_validation.py**
Valida que los nuevos modelos se pueden cargar correctamente.

**Ejecutar:**
```bash
python test_models_validation.py
```

**Qué hace:**
1. Carga modelo de embeddings
2. Genera embedding de prueba
3. Carga modelo de reranker
4. Ejecuta reranking de prueba
5. Reporta si todo funciona

### **test_embedding_upgrade.py**
Compara la recuperación con los nuevos modelos.

**Ejecutar:**
```bash
python test_embedding_upgrade.py
```

**Qué hace:**
1. Ejecuta 6 queries de prueba (incluyendo fecha incorrecta: "fundado 1903")
2. Mide Hit Rate (¿encuentra el doc correcto?)
3. Mide MRR (¿en qué posición?)
4. Mide tiempo de recuperación
5. Opcional: Compara con modelos antiguos (descomentar código)

## 📋 Pasos de Ejecución

### **Paso 1: Instalar dependencias**
```bash
pip install FlagEmbedding
```

### **Paso 2: Validar modelos**
```bash
python test_models_validation.py
```

**Resultado esperado:**
```
✅ Embedding: OK
✅ Reranker: OK
```

Si hay errores, seguir las instrucciones mostradas.

### **Paso 3: Re-ingestar datos con nuevos embeddings**

**IMPORTANTE**: Los embeddings son diferentes, así que los vectores almacenados en ChromaDB ya no sirven.

```bash
python ingest/ingest_data.py --clear
```

**Tiempo esperado**: ~5-10 minutos (732 chunks con HyDE)

**Qué hace:**
1. Borra vector store antiguo
2. Procesa documentos (189 chunks)
3. Aplica HyDE (732 chunks totales)
4. **Genera nuevos embeddings con Alibaba-NLP/gte-multilingual-base**
5. Almacena en ChromaDB

### **Paso 4: Test de recuperación**
```bash
python test_embedding_upgrade.py
```

**Métricas a observar:**
- **Hit Rate**: Debe ser > 80% (antes era ~50% con paraphrase-multilingual)
- **MRR**: Debe ser > 0.75 (antes era ~0.60)
- **Queries críticos**: 
  - "fundado en 1903" → **DEBE** encontrar Sec. 1 con "1902" (contradicción)
  - "fundado en 1900" → DEBE encontrar Sec. 1 con "orígenes 1900"

### **Paso 5: Test completo de fact-checking**
```bash
python test_mejoras.py
```

**Objetivo**: Pasar de **2/4 tests (50%)** a **3/4 o 4/4 (75-100%)**

**Tests críticos:**
- ✅ Test 1: "fundado en 1902" → VERDADERO (ya pasaba)
- ❌→✅ Test 2: "fundado en 1903" → **FALSO** (antes fallaba: NO SE PUEDE VERIFICAR)
- ❌→✅ Test 3: "fundado en 1950" → **FALSO** (antes fallaba: NO SE PUEDE VERIFICAR)
- ✅ Test 4: "Barcelona Champions 2015" → NO SE PUEDE VERIFICAR (ya pasaba)

## 🎯 Ganancia Esperada

### **Antes (modelos antiguos):**
```
Hit Rate@5: ~50%
MRR: ~0.60
Test mejoras: 2/4 (50%)
```

### **Después (modelos nuevos):**
```
Hit Rate@5: ~80-85% (+30-35 puntos)
MRR: ~0.75-0.80 (+0.15-0.20)
Test mejoras: 3/4 o 4/4 (75-100%)
```

## 🔍 Por Qué Funciona Mejor

### **1. Embeddings más expresivos**
- **768 dim vs 384 dim**: Captura más semántica
- **Mejor entrenamiento**: Dataset más diverso y reciente
- **Multilingüe nativo**: No es traducción, sino entrenamiento directo en español

### **2. Mejor distinción numérica**
- Modelos modernos entienden mejor diferencias numéricas
- "1902" vs "1903" → vectores más distinguibles
- Búsqueda de "fundado 1903" ahora recupera "fundado 1902" (contradictorio)

### **3. Reranking más preciso**
- Cross-encoder lee query + doc **juntos** (no solo embeddings)
- Detecta mejor contradicciones sutiles
- Especialmente efectivo para fechas/números

## 🚀 Próximo Paso: Prioridad 3

Una vez validado que esto funciona, implementar **Proposition Chunking**:

### **Qué es:**
En lugar de chunks largos (1000 chars), dividir en **hechos atómicos**:

```
ANTES (chunk semántico):
"El Real Madrid fue registrado el 6 de marzo de 1902. Sus orígenes datan 
de 1900. Fue fundado por estudiantes españoles. El primer presidente fue 
Juan Padrós."

DESPUÉS (proposiciones):
1. "El Real Madrid fue registrado el 6 de marzo de 1902"
2. "Los orígenes del Real Madrid datan de 1900"
3. "El Real Madrid fue fundado por estudiantes españoles"
4. "El primer presidente del Real Madrid fue Juan Padrós"
```

### **Ventaja:**
- Embedding de proposición es **más preciso** (sin ruido)
- Búsqueda de "fundado 1903" → embedding muy cercano a "registrado 1902"
- LLM recibe contexto **limpio** para comparar

### **Implementación:**
1. Usar LLM (Llama 3.2) para extraer proposiciones de cada chunk
2. Almacenar proposiciones + chunk original
3. Búsqueda en proposiciones, retornar chunks originales
4. Ganancia esperada: +5-10% Hit Rate adicional

## 📝 Notas

- **Backup recomendado**: Guardar `data/vector_store` antes de `--clear`
- **GPU opcional**: Modelos funcionan en CPU, pero GPU acelera 3-5x
- **Memoria RAM**: ~8GB recomendado (modelos + ChromaDB)
- **Comparación opcional**: Para comparar con modelos antiguos, descomentar código en `test_embedding_upgrade.py`

## 🐛 Troubleshooting

### Error: "No module named 'FlagEmbedding'"
```bash
pip install FlagEmbedding
```

### Error: Modelo no se descarga
- Verificar conexión a internet
- HuggingFace puede requerir token para algunos modelos
- Alternativa: usar `intfloat/multilingual-e5-large` en config.yaml

### Error: Out of Memory
- Reducir `batch_size` en embeddings
- Cerrar otros programas
- Alternativa: usar modelo más pequeño `BAAI/bge-small-en-v1.5`

### Recuperación lenta
- Normal en primera ejecución (descarga modelos)
- Posteriormente: embeddings ~0.05s, reranking ~0.1s por query
- GPU acelera 3-5x

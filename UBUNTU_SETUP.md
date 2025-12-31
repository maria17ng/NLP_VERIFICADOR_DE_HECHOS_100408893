# 🐧 Guía de Instalación y Troubleshooting para Ubuntu 24.04

## ⚡ Inicio Rápido

### Opción 1: Docker (Recomendado - Instalación Automática)

```bash
# Docker se instala automáticamente si no está presente
make all

# Si aparece error de permisos:
sudo usermod -aG docker $USER
newgrp docker
make all
```

**Frontend:** http://localhost:5174  
**Backend:** http://localhost:8000

**Detener:** `make docker-down`

### Opción 2: Desarrollo Local (Requiere Dependencias)

```bash
# 1. Instalar dependencias del sistema
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip nodejs npm

# 2. Configurar entorno Python
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Iniciar en modo desarrollo
make dev
```

**Frontend:** http://localhost:5174  
**Backend:** http://localhost:8000

---

## 🔧 Problemas Comunes y Soluciones

### ❌ Error: `vite: Permission denied`

**Problema:** El ejecutable de Vite no tiene permisos de ejecución.

**Soluciones:**

1. **Opción A - Usar npx (recomendado):**
   ```bash
   cd frontend
   npx vite
   ```

2. **Opción B - Dar permisos:**
   ```bash
   chmod +x frontend/node_modules/.bin/vite
   npm run dev
   ```

3. **Opción C - Reinstalar dependencias:**
   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   npm run dev
   ```

### ❌ Error: `Module not found: chromadb`

**Problema:** ChromaDB no está instalado o el entorno virtual no está activado.

**Solución:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### ❌ Error: `OPENAI_API_KEY not found`

**Problema:** Falta la API key de OpenAI.

**Solución:**
```bash
# Crear archivo .env
cat > .env << EOF
OPENAI_API_KEY=tu-api-key-aqui
EOF

# O exportar temporalmente
export OPENAI_API_KEY="tu-api-key-aqui"
```

### ❌ Error: `Port 8000 already in use`

**Problema:** El puerto 8000 está ocupado.

**Solución:**
```bash
# Encontrar proceso
lsof -i :8000

# Matar proceso
kill -9 <PID>

# O usar otro puerto
uvicorn api.server:app --port 8001
```

### ❌ Frontend no carga en localhost:5174

**Problema:** Frontend no inició correctamente.

**Solución:**
```bash
# Verificar que el backend esté corriendo
curl http://localhost:8000/api/health

# Iniciar frontend manualmente
cd frontend
npm install  # Si es primera vez
npx vite --host 0.0.0.0 --port 5174
```

---

## 🧪 Diagnóstico del Atlético de Madrid

Si el sistema RAG no encuentra datos del Atlético de Madrid:

```bash
# 1. Ejecutar diagnóstico
python debug_atletico.py

# 2. Si faltan datos, re-ingestar
python test.py --clear

# 3. Verificar resultado
python debug_atletico.py
```

**Síntomas:**
- Tests `ATM_1`, `ATM_2`, `ATM_3` fallan con "NO SE PUEDE VERIFICAR"
- El sistema no encuentra la fecha de fundación (1903)

**Causa probable:**
- Documentos del Atlético no están en ChromaDB
- Chunking separó la información clave
- Embeddings no capturan semántica de "fundado en 1903"

**Solución:**
```bash
# Re-ingestar con chunking más grande
python test.py --clear
```

---

## 📊 Comparación RAG vs Baseline

El frontend **ya incluye** la comparación con el baseline:

1. ✅ Marca el checkbox "Comparar con baseline" en la UI
2. ✅ El backend ejecuta ambos sistemas
3. ✅ Muestra diferencias de accuracy, confianza y tiempo

**Endpoint API:**
```bash
curl -X POST http://localhost:8000/api/verify \
  -H "Content-Type: application/json" \
  -d '{
    "question": "El Real Madrid fue fundado en 1902",
    "compare_baseline": true
  }'
```

**Comparación desde terminal:**
```bash
python compare_systems.py
```

**Resultados esperados:**
- 🎯 **RAG**: ~78% accuracy (11/14)
- 📉 **Baseline**: ~71% accuracy (10/14)
- 🏆 **Ventaja RAG**: +7% accuracy

---

## 🚀 Comandos Útiles

### Desarrollo

```bash
# Iniciar todo (backend + frontend)
make all

# Solo backend
make backend

# Solo frontend
make frontend

# Desarrollo con logs
make dev
```

### Testing

```bash
# Test completo
python test_comprehensive.py

# Comparación RAG vs Baseline
python compare_systems.py

# Test individual
python -m pytest test_fase1.py -v
```

### Mantenimiento

```bash
# Limpiar caché y logs
make clean

# Resetear proyecto completo
make reset

# Ver ayuda
make help
```

---

## 🐳 Docker (Alternativa)

Si prefieres usar Docker:

```bash
# Construir imagen
make docker-build

# Iniciar servicios
make docker-up

# En otra terminal, iniciar frontend
make frontend

# Detener servicios
make docker-down
```

---

## 📝 Verificar Instalación

```bash
# 1. Python y venv
python3.12 --version
source .venv/bin/activate
python --version  # Debe ser 3.12.x

# 2. Dependencias Python
pip list | grep -E "chromadb|openai|fastapi"

# 3. Node.js y npm
node --version  # Debe ser >= 18
npm --version

# 4. Dependencias frontend
cd frontend
npm list vite react

# 5. Variables de entorno
echo $OPENAI_API_KEY

# 6. Archivos de datos
ls -lh data/raw/*.txt | wc -l  # Debe ser >= 11
ls -lh data/vector_store/  # Debe existir

# 7. Backend health
curl http://localhost:8000/api/health

# 8. Frontend
curl http://localhost:5174
```

---

## 🔍 Logs y Debug

```bash
# Ver logs del backend
tail -f logs/factchecker.log

# Ver logs de ChromaDB
tail -f logs/chromadb.log

# Debug de embeddings
python debug_chromadb.py

# Debug de retrieval
python test_retrieval_debug.py
```

---

## 💡 Tips de Performance

### 1. Caché de Embeddings
ChromaDB cachea embeddings automáticamente. Si cambias documentos:
```bash
python test.py --clear  # Regenera toda la base
```

### 2. Modo Producción
Para producción, desactiva logs DEBUG:
```python
# settings/config.py
LOG_LEVEL = "INFO"  # En lugar de "DEBUG"
```

### 3. Paralelización
El sistema usa threading para recuperación. Ajusta workers:
```python
# retriever/advanced_retriever.py
MAX_WORKERS = 4  # Aumentar en servidores potentes
```

---

## 🆘 Soporte

Si persisten los problemas:

1. **Revisar logs:**
   ```bash
   tail -f logs/factchecker.log
   ```

2. **Ejecutar diagnósticos:**
   ```bash
   python debug_chromadb.py
   python debug_atletico.py
   ```

3. **Resetear proyecto:**
   ```bash
   make reset
   ```

4. **Verificar versiones:**
   ```bash
   python --version  # 3.12.x
   node --version    # >= 18.x
   npm --version     # >= 9.x
   ```

---

## ✅ Checklist de Instalación Completa

- [ ] Python 3.12 instalado
- [ ] Node.js >= 18 instalado
- [ ] Entorno virtual creado y activado
- [ ] `OPENAI_API_KEY` configurado en `.env`
- [ ] Dependencias Python instaladas (`make install`)
- [ ] Dependencias frontend instaladas (`cd frontend && npm install`)
- [ ] Datos ingestados (`make ingest`)
- [ ] Backend corriendo en http://localhost:8000
- [ ] Frontend corriendo en http://localhost:5174
- [ ] ChromaDB con 4472+ documentos
- [ ] Test básico exitoso (`python test.py`)
- [ ] Comparación exitosa (`python compare_systems.py`)

---

**Última actualización:** 2025-12-31

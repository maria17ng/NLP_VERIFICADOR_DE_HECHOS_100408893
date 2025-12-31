# ⚡ Guía de Inicio Rápido

## 🎯 Objetivo

Verificar automáticamente la veracidad de afirmaciones sobre equipos de fútbol de Madrid usando RAG (Retrieval-Augmented Generation).

---

## 🚀 Ejecución en 1 Comando

### Opción 1: Docker (Recomendado)

```bash
make all
```

Luego abre: **http://localhost:5174**

¡Listo! 🎉

**🐳 ¿Qué hace ese comando?**

1. Verifica e instala Docker si es necesario (Ubuntu)
2. Ingiere 11 documentos de Wikipedia a ChromaDB
3. Construye imágenes Docker (backend + frontend)
4. Inicia ambos servicios en contenedores
5. Backend: http://localhost:8000
6. Frontend: http://localhost:5174

**Detener:** `make docker-down`

### Opción 2: Desarrollo Local (Sin Docker)

```bash
make dev
```

**📝 ¿Qué hace ese comando?**

1. Instala dependencias Python y Node.js
2. Descarga modelo de spaCy para español
3. Ingiere 11 documentos de Wikipedia a ChromaDB
4. Inicia backend con hot-reload en puerto 8000
5. Inicia frontend con hot-reload en puerto 5174

---

## 🪟 Sin Make (Windows)

```bash
# Opción 1: Script automatizado
.\start.bat

# Opción 2: Manual
# Terminal 1
uvicorn api.server:app --reload --port 8000

# Terminal 2
cd frontend
npm run dev
```

---

## ✅ Ejemplos de Uso

### En la interfaz web (http://localhost:5174)

Escribe afirmaciones como:

**✅ VERDADERO:**
- "El Real Madrid fue fundado en 1902"
- "El Atlético de Madrid ganó La Liga en 2021"

**❌ FALSO:**
- "El Real Madrid nunca ha ganado la Champions"
- "El Getafe juega en el Bernabéu"

**❓ NO SE PUEDE VERIFICAR:**
- "El Madrid ganará la Champions en 2030"

### Por API

```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"question": "El Real Madrid fue fundado en 1902"}'
```

---

## 🛠️ Comandos Útiles

```bash
# Ver todos los comandos disponibles
make help

# Solo instalar dependencias
make install

# Solo ingerir datos
make ingest

# Solo backend
make backend

# Solo frontend
make frontend

# Limpiar todo
make clean
```

---

## 🌍 Idiomas Soportados

- 🇪🇸 Español
- 🇬🇧 Inglés
- 🇫🇷 Francés
- 🇩🇪 Alemán
- 🇮🇹 Italiano
- 🇵🇹 Portugués

---

## 📚 ¿Qué documentos tiene?

11 archivos sobre equipos de Madrid:
- Real Madrid (historia y palmarés)
- Atlético de Madrid (historia y palmarés)
- Getafe CF (trayectoria)
- CD Leganés (historia)
- Rayo Vallecano

---

## 🐛 Problemas Comunes

### No encuentra documentos
```bash
python test.py --clear
```

### Puerto ocupado
Cambiar en `config.yaml` o usar otro puerto:
```bash
uvicorn api.server:app --port 8001
```

### Frontend no conecta
Verificar CORS en `api/server.py` (debe incluir `http://localhost:5174`)

---

## 📖 Documentación Completa

Lee `README.md` para detalles técnicos, arquitectura y configuración avanzada.

---

**💡 Consejo**: Ejecuta `make all` y espera 2-3 minutos la primera vez (descarga modelos).

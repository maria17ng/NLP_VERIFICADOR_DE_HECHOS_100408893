.PHONY: help install ingest backend frontend dev clean docker-build docker-up docker-down all

# Variables
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
IMAGE_NAME := factchecker-api
BACKEND_PORT := 8000
FRONTEND_PORT := 5174

# Colores para mensajes
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Mostrar esta ayuda
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)    Sistema de Verificación de Hechos - Comandos Make    $(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(YELLOW)Comandos disponibles:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Uso rápido:$(NC)"
	@echo "  make all              # Ejecutar todo el proyecto (recomendado)"
	@echo "  make dev              # Desarrollo sin Docker"
	@echo ""

venv: ## Crear entorno virtual
	@test -d $(VENV) || python3 -m venv $(VENV)
	@echo "✅ Entorno virtual listo"

install: venv ## Instalar dependencias de Python y Node.js
	@echo "📦 Instalando dependencias de Python..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "📦 Descargando modelo de spaCy..."
	$(PYTHON) -m spacy download es_core_news_sm
	@echo "📦 Instalando dependencias de Node.js..."
	cd frontend && npm install
	@echo "✅ Dependencias instaladas correctamente"

ingest: ## Ingerir datos a la base vectorial (con --clear)
	@echo "$(BLUE)📥 Ingiriendo datos a ChromaDB...$(NC)"
	$(PYTHON) test.py --clear
	@echo "$(GREEN)✅ Datos ingresados correctamente$(NC)"

backend: ## Iniciar servidor backend (API FastAPI)
	@echo "$(BLUE)🚀 Iniciando servidor backend en puerto $(BACKEND_PORT)...$(NC)"
	@echo "$(YELLOW)📍 API disponible en: http://localhost:$(BACKEND_PORT)$(NC)"
	$(PYTHON) -m uvicorn api.server:app --reload --port $(BACKEND_PORT)

frontend: ## Iniciar servidor frontend (Vite + React)
	@echo "$(BLUE)🚀 Iniciando servidor frontend en puerto $(FRONTEND_PORT)...$(NC)"
	@echo "$(YELLOW)📍 Frontend disponible en: http://localhost:$(FRONTEND_PORT)$(NC)"
	cd frontend && npm run dev

docker-build: ## Construir imagen Docker
	@echo "$(BLUE)🐳 Construyendo imagen Docker...$(NC)"
	$(DOCKER) build -t $(IMAGE_NAME) .
	@echo "$(GREEN)✅ Imagen Docker construida: $(IMAGE_NAME)$(NC)"

docker-up: docker-build ingest ## Iniciar servicios con Docker Compose
	@echo "$(BLUE)🐳 Iniciando contenedores...$(NC)"
	@echo "$(YELLOW)Nota: Después de que el backend esté listo, abre otra terminal y ejecuta 'make frontend'$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✅ Backend corriendo en: http://localhost:$(BACKEND_PORT)$(NC)"
	@echo "$(YELLOW)⚠️  Ejecuta 'make frontend' en otra terminal para iniciar el frontend$(NC)"

docker-down: ## Detener servicios Docker
	@echo "$(BLUE)🐳 Deteniendo contenedores...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✅ Contenedores detenidos$(NC)"

dev: install ingest ## Iniciar desarrollo (sin Docker)
	@echo "$(GREEN)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)    Iniciando Sistema de Verificación de Hechos         $(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(YELLOW)📍 Backend:  http://localhost:$(BACKEND_PORT)$(NC)"
	@echo "$(YELLOW)📍 Frontend: http://localhost:$(FRONTEND_PORT)$(NC)"
	@echo ""
	@echo "$(BLUE)Iniciando servicios...$(NC)"
	@echo "$(YELLOW)⚠️  Presiona Ctrl+C para detener todos los servicios$(NC)"
	@echo ""
	cd frontend && npx vite &
	$(PYTHON) -m uvicorn api.server:app --reload --port $(BACKEND_PORT)


all: install ingest ## Ejecutar todo el proyecto (instalar, ingerir e iniciar)
	@echo "$(GREEN)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)    Sistema de Verificación de Hechos - INICIANDO        $(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(YELLOW)📍 Backend:  http://localhost:$(BACKEND_PORT)$(NC)"
	@echo "$(YELLOW)📍 Frontend: http://localhost:$(FRONTEND_PORT)$(NC)"
	@echo ""
	@echo "$(BLUE)🚀 Iniciando backend y frontend...$(NC)"
	@echo "$(YELLOW)⚠️  Abre http://localhost:$(FRONTEND_PORT) en tu navegador$(NC)"
	@echo "$(YELLOW)⚠️  Presiona Ctrl+C para detener todos los servicios$(NC)"
	@echo ""
	cd frontend && npx vite &
	$(PYTHON) -m uvicorn api.server:app --reload --port $(BACKEND_PORT)


clean: ## Limpiar archivos temporales y caché
	@echo "$(BLUE)🧹 Limpiando archivos temporales...$(NC)"
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist logs rmdir /s /q logs
	@if exist data\vector_store rmdir /s /q data\vector_store
	@for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@echo "$(GREEN)✅ Limpieza completada$(NC)"

reset: clean install ingest ## Resetear proyecto (limpiar, instalar e ingerir)
	@echo "$(GREEN)✅ Proyecto reseteado correctamente$(NC)"

FROM python:3.11-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Descargar modelo de spaCy
RUN python -m spacy download es_core_news_sm

# Copiar el código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p data/vector_store logs

# Exponer puerto para la API
EXPOSE 8000

# Comando por defecto (se sobrescribe en docker-compose)
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]

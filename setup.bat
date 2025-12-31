@echo off
echo ===============================================
echo   Sistema de Verificacion de Hechos - Setup
echo ===============================================
echo.

echo [1/5] Instalando dependencias de Python...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo [2/5] Descargando modelo de spaCy...
python -m spacy download es_core_news_sm

echo.
echo [3/5] Instalando dependencias de Node.js...
cd frontend
call npm install
cd ..

echo.
echo [4/5] Ingiriendo datos a ChromaDB...
python test.py --clear

echo.
echo [5/5] Setup completado!
echo.
echo ===============================================
echo   Para iniciar el sistema:
echo.
echo   Opcion 1 - Docker (Recomendado):
echo   make all
echo.
echo   Opcion 2 - Desarrollo local:
echo   make dev
echo
echo   Opcion 3 - Manual (Windows):
echo   start.bat
echo ===============================================
pause

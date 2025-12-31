@echo off
echo ===============================================
echo   Sistema de Verificacion de Hechos
echo   Modo: Desarrollo Local (Sin Docker)
echo ===============================================
echo.
echo Iniciando backend y frontend...
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5174
echo.
echo Para usar Docker: make all
echo Presiona Ctrl+C para detener todos los servicios
echo ===============================================
echo.

:: Iniciar frontend en segundo plano
start /B cmd /c "cd frontend && npm run dev"

:: Esperar 5 segundos para que el frontend inicie
timeout /t 5 /nobreak >nul

:: Iniciar backend (en primer plano)
uvicorn api.server:app --reload --port 8000

@echo off
title Multimodal E-Commerce Engine Launcher
echo ============================================================
echo   Launching Multimodal Search Engine (FastAPI + Streamlit)
echo ============================================================
cd /d "%~dp0"

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "FastAPI Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000"

echo Waiting 5 seconds for CLIP & FAISS models to initialize...
timeout /t 5 /nobreak >nul

echo [2/2] Starting Streamlit UI on http://localhost:8501 ...
start "Streamlit Frontend" cmd /k ".venv\Scripts\python.exe -m streamlit run app.py"

echo ============================================================
echo   SUCCESS! Both servers launched in separate windows.
echo   • API Backend:  http://127.0.0.1:8000
echo   • Streamlit UI: http://localhost:8501
echo ============================================================

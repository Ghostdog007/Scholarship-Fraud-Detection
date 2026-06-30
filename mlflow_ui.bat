@echo off
REM Launch the MLflow UI backed by this project's SQLite database.
REM Run from any location:  mlflow_ui.bat
REM Then open:              http://localhost:5000

set PROJECT_DIR=%~dp0
echo Starting MLflow UI for NIC Fraud Detection V3...
echo Tracking DB    : %PROJECT_DIR%mlflow.db
echo Artifacts root : %PROJECT_DIR%mlruns
echo Open browser   : http://localhost:5000
echo Press Ctrl+C to stop.
echo.

"%PROJECT_DIR%.venv\Scripts\python.exe" -m mlflow ui ^
    --backend-store-uri "sqlite:///%PROJECT_DIR%mlflow.db" ^
    --default-artifact-root "%PROJECT_DIR%mlruns" ^
    --port 5000

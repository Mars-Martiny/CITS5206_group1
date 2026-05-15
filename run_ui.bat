@echo off
setlocal enabledelayedexpansion

set MARKER=.installed

echo ============================================
echo   Incident Classifier -- Web UI
echo   [Windows - AI-converted, not tested]
echo ============================================

if not exist "%MARKER%" (
    echo [Run] Not installed yet -- running installer first...
    echo.
    call install.bat
    if errorlevel 1 exit /b 1
    echo.
)

rem --- Read marker ---
for /f "tokens=1,2 delims==" %%a in (%MARKER%) do (
    if "%%a"=="env_type" set ENV_TYPE=%%b
    if "%%a"=="env_name" set ENV_NAME=%%b
    if "%%a"=="gpu" set GPU=%%b
)

echo [Run] GPU backend  : !GPU!
echo [Run] Environment  : !ENV_TYPE! ^(!ENV_NAME!^)

rem --- Activate environment ---
if "!ENV_TYPE!"=="conda" (
    for /f "tokens=*" %%i in ('conda info --base 2^>nul') do set CONDA_BASE=%%i
    call "!CONDA_BASE!\Scripts\activate.bat" "!CONDA_BASE!"
    call conda activate !ENV_NAME!
) else (
    call .venv\Scripts\activate.bat
)

echo [Run] Starting Streamlit app...
echo.
streamlit run app/app.py

endlocal

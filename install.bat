@echo off
setlocal enabledelayedexpansion

set ENV_NAME=hs_classifier
set MARKER=.installed
set FORCE=false

for %%a in (%*) do (
    if /i "%%a"=="--force" set FORCE=true
)

echo ============================================
echo   Incident Classifier -- Installer
echo   [Windows - AI-converted, not tested]
echo ============================================

if exist "%MARKER%" if "!FORCE!"=="false" (
    echo [Install] Already installed. Run with --force to reinstall.
    exit /b 0
)

rem --- Python version check ---
echo [Install] Checking Python version...
where python >nul 2>&1
if errorlevel 1 (
    echo [Error] python not found. Please install Python 3.12+.
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
for /f "tokens=1,2 delims=." %%a in ("!PYTHON_VERSION!") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

set VERSION_OK=false
if !PYTHON_MAJOR! GTR 3 set VERSION_OK=true
if !PYTHON_MAJOR! EQU 3 if !PYTHON_MINOR! GEQ 12 set VERSION_OK=true

if "!VERSION_OK!"=="false" (
    echo [Error] Python 3.12+ required. Found: !PYTHON_VERSION!
    exit /b 1
)
echo [Install] Python !PYTHON_VERSION! OK

rem --- GPU detection ---
echo [Install] Detecting GPU...
set TORCH_INDEX=
set GPU_TYPE=cpu

where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        set GPU_TYPE=nvidia
        set TORCH_INDEX=https://download.pytorch.org/whl/cu130
        echo [Install] NVIDIA GPU detected -- will install CUDA 13.0 build
        goto :env_setup
    )
)

where rocm-smi >nul 2>&1
if not errorlevel 1 (
    rocm-smi >nul 2>&1
    if not errorlevel 1 (
        set GPU_TYPE=amd
        set TORCH_INDEX=https://download.pytorch.org/whl/rocm7.0
        echo [Install] AMD GPU detected -- will install ROCm 7.0 build
        goto :env_setup
    )
)

echo [Install] No GPU detected -- using CPU-only build

:env_setup
rem --- Environment setup ---
set USE_CONDA=false

where conda >nul 2>&1
if not errorlevel 1 (
    set USE_CONDA=true
    echo [Install] conda found -- setting up environment '!ENV_NAME!'

    for /f "tokens=*" %%i in ('conda info --base 2^>nul') do set CONDA_BASE=%%i
    call "!CONDA_BASE!\Scripts\activate.bat" "!CONDA_BASE!"

    conda env list 2>nul | findstr /r /b "!ENV_NAME! " >nul 2>&1
    if errorlevel 1 (
        echo [Install] Creating conda environment '!ENV_NAME!' with Python 3.12...
        call conda create -n !ENV_NAME! python=3.12 -y
        if errorlevel 1 exit /b 1
    ) else (
        echo [Install] Conda environment '!ENV_NAME!' already exists -- reusing
    )

    call conda activate !ENV_NAME!
    echo [Install] Conda environment '!ENV_NAME!' activated
) else (
    echo [Install] conda not found -- falling back to venv ^(.venv^)

    if not exist ".venv" (
        echo [Install] Creating .venv...
        python -m venv .venv
        if errorlevel 1 exit /b 1
    )

    call .venv\Scripts\activate.bat
    echo [Install] venv activated
)

rem --- Base dependencies (torch handled separately for GPU index) ---
echo [Install] Installing base dependencies...
python -m pip install --upgrade pip -q

set TMPFILE=%TEMP%\reqs_notorch_%RANDOM%.txt
findstr /v /b "torch" requirements.txt > "!TMPFILE!"
pip install -r "!TMPFILE!"
if errorlevel 1 exit /b 1
del "!TMPFILE!"

rem --- PyTorch (GPU-appropriate) ---
echo [Install] Installing PyTorch ^(!GPU_TYPE!^)...
if not "!TORCH_INDEX!"=="" (
    pip install torch torchvision --index-url !TORCH_INDEX!
) else (
    pip install torch torchvision
)
if errorlevel 1 exit /b 1

rem --- spaCy language model ---
echo [Install] Downloading spaCy model ^(en_core_web_sm^)...
python -m spacy download en_core_web_sm
if errorlevel 1 exit /b 1

rem --- Write marker ---
if "!USE_CONDA!"=="true" (set ENV_TYPE_VAL=conda) else (set ENV_TYPE_VAL=venv)
(
    echo env_type=!ENV_TYPE_VAL!
    echo env_name=!ENV_NAME!
    echo gpu=!GPU_TYPE!
) > "!MARKER!"

echo.
echo ============================================
echo   Installation complete.
echo   GPU backend : !GPU_TYPE!
if "!USE_CONDA!"=="true" (
    echo   Environment : conda ^(!ENV_NAME!^)
) else (
    echo   Environment : venv ^(.venv^)
)
echo   Run the app : run_ui.bat
echo ============================================

endlocal

@echo off
setlocal enabledelayedexpansion

set MARKER=.installed

echo [Docs] Building Sphinx documentation...
echo [Docs] [Windows - AI-converted, not tested]

if exist "%MARKER%" (
    for /f "tokens=1,2 delims==" %%a in (%MARKER%) do (
        if "%%a"=="env_type" set ENV_TYPE=%%b
        if "%%a"=="env_name" set ENV_NAME=%%b
    )

    if "!ENV_TYPE!"=="conda" (
        for /f "tokens=*" %%i in ('conda info --base 2^>nul') do set CONDA_BASE=%%i
        call "!CONDA_BASE!\Scripts\activate.bat" "!CONDA_BASE!"
        call conda activate !ENV_NAME!
    ) else (
        call .venv\Scripts\activate.bat
    )
) else (
    echo [Docs] Warning: .installed not found -- using current Python environment
)

cd docs
call make.bat html
if errorlevel 1 exit /b 1

echo.
echo [Docs] Build complete. Output: docs\build\html\index.html

endlocal

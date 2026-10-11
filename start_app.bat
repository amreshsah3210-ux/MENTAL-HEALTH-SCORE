@echo off
setlocal
cd /d "%~dp0"
set "APP_PORT=%~1"
if "%APP_PORT%"=="" set "APP_PORT=8000"

if not exist ".venv312\Scripts\python.exe" (
    echo Creating the Python 3.12 environment...
    py -3.12 -m venv .venv312
    if errorlevel 1 goto setup_failed
)

echo Installing project requirements in the Python 3.12 environment...
".venv312\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto setup_failed

".venv312\Scripts\python.exe" -c "import sklearn; print('Using scikit-learn', sklearn.__version__); assert sklearn.__version__ == '1.6.1', 'Expected scikit-learn 1.6.1'"
if errorlevel 1 goto setup_failed

".venv312\Scripts\python.exe" -m uvicorn main:app --port %APP_PORT%
exit /b %errorlevel%

:setup_failed
echo.
echo Setup failed. Make sure Python 3.12 is installed and available as "py -3.12".
pause
exit /b 1

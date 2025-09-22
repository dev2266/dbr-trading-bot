@echo off
echo =========================================
echo Enhanced Trading Bot - Azure Deployment
echo =========================================

:: Set error handling
setlocal enabledelayedexpansion

:: Upgrade pip and install packages
echo Installing Python packages...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ERROR: Failed to upgrade pip
    exit /b %errorlevel%
)

python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    exit /b %errorlevel%
)

:: Install additional ML packages if needed
python -m pip install --no-deps ta pandas-ta

:: Create necessary directories
echo Creating application directories...
if not exist "logs" mkdir logs
if not exist "data" mkdir data  
if not exist "temp" mkdir temp
if not exist "cache" mkdir cache
if not exist "models" mkdir models

:: Set file permissions
echo Setting up file permissions...

:: Verify essential files
echo Verifying deployment files...
if exist "bot.py" (
    echo ✅ bot.py found
) else (
    echo ❌ bot.py missing!
    exit /b 1
)

if exist "your_analysis_module.py" (
    echo ✅ Professional analysis module found
) else (
    echo ⚠️ Professional analysis module missing
)

if exist "symbol_mapper.py" (
    echo ✅ Symbol mapper found
) else (
    echo ⚠️ Symbol mapper missing
)

:: Log successful deployment
echo %date% %time% - Enhanced deployment completed >> deployment.log

echo =========================================
echo Enhanced Trading Bot Deployment Complete
echo =========================================

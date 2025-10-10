@echo off
REM PyTradePath Release Script for Windows

echo PyTradePath Release Script
echo ========================

REM Check if we're in the right directory
if not exist "setup.py" (
    echo Error: setup.py not found. Please run this script from the project root directory.
    pause
    exit /b 1
)

REM Create a zip archive of the project
echo Creating zip archive...
python release.py --package-only

echo.
echo Release archive created: pytradepath-release.zip
echo.

echo To use this release:
echo 1. Extract the zip file
echo 2. Run examples: python examples/simple_backtest.py
echo 3. Run CLI: python -m pytradepath.cli --help
echo.

echo Release preparation completed successfully!
pause
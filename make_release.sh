#!/bin/bash
# PyTradePath Release Script for Unix/Linux/MacOS

echo "PyTradePath Release Script"
echo "========================"

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "Error: setup.py not found. Please run this script from the project root directory."
    exit 1
fi

# Create a zip archive of the project
echo "Creating zip archive..."
zip -r pytradepath-release.zip . -x "*.git*" "dist/*" "build/*" "__pycache__" "*.pyc" "logs/*" "cache/*" "temp/*" "tmp/*" "exported_data/*"

echo ""
echo "Release archive created: pytradepath-release.zip"
echo ""

echo "To use this release:"
echo "1. Extract the zip file"
echo "2. Run examples: python examples/simple_backtest.py"
echo "3. Run CLI: python -m pytradepath.cli --help"
echo ""

echo "Release preparation completed successfully!"
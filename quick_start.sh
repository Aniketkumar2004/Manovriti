#!/bin/bash

# Quick Start Script for Toxic Comment Classification Project
# This script sets up the environment and runs a basic example

set -e  # Exit on error

echo "🛡️ Toxic Comment Classification - Quick Start"
echo "=============================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install basic requirements
echo "📚 Installing basic requirements..."
pip install pandas numpy scikit-learn nltk matplotlib seaborn

# Download NLTK data
echo "📥 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# Run simple example
echo ""
echo "🚀 Running simple example..."
echo "=============================================="
python examples/simple_example.py

echo ""
echo "🎉 Quick start completed successfully!"
echo ""
echo "Next steps:"
echo "1. Install additional dependencies for full functionality:"
echo "   pip install -r requirements.txt"
echo ""
echo "2. Train models with your own data:"
echo "   python src/train_evaluate.py --data your_data.csv"
echo ""
echo "3. Start the API server:"
echo "   python src/api.py"
echo ""
echo "4. Launch the web interface:"
echo "   streamlit run src/web_app.py"
echo ""
echo "5. Run API client examples:"
echo "   python examples/api_client.py"
echo ""
echo "For more information, see README.md"
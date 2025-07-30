#!/bin/bash

# Setup script for the ML environment

echo "Setting up ML environment..."

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install pre-commit hooks (optional)
# pip install pre-commit
# pre-commit install

echo "Environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"

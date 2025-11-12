#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies if not already installed
if [ ! -f .deps_installed ]; then
    echo "Installing dependencies..."
    uv pip install -r requirements.txt
    touch .deps_installed
fi

# Start the MkDocs development server
echo "Starting development server at http://127.0.0.1:8000"
echo "Press Ctrl+C to stop the server"
uv run mkdocs serve

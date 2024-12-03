#!/bin/bash
set -e

echo "Navigating to /home/ubuntu/app..."
cd /home/ubuntu/app

echo "Ownership and permissions of /home/ubuntu/app before creating venv:"
ls -ld /home/ubuntu/app

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Virtual environment setup complete. Python and pip versions:"
which python
python --version
which pip
pip --version

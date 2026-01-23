#!/usr/bin/env bash
###
# Run pytest with optional snapshot update
# Usage:
#   bash scripts/test.sh ## Run tests without snapshot update
#   bash scripts/test.sh ## Run tests with snapshot update
###
set -euo pipefail

# Get the project root directory
SELF=$(readlink -f "${BASH_SOURCE[0]}")
DIR=${SELF%/*/*}

cd -- "$DIR"


# Setup virtual environment if needed
if [[ ! -e ./venv ]]; then
  echo "Creating virtual environment..."
  bash ./scripts/install_in_venv.sh
fi

# Activate the virtual environment
source ./venv/bin/activate

# Handle optional --update flag
if [[ "${1:-}" == "--update" ]]; then
  echo "Running tests with snapshot update..."
  pytest --snapshot-update --import-mode=importlib --cov=. --cov-report=xml --cov-report=term test/
else
  echo "Running tests..."
  pytest -vv --import-mode=importlib --cov=. --cov-report=xml --cov-report=term test/
fi
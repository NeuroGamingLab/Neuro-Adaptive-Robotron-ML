#!/usr/bin/env bash
# Start Robotron:2026 with ML using the project venv
# Usage: ./start.sh   (or: bash start.sh)

set -e
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "No .venv found. Create one with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

source .venv/bin/activate
exec streamlit run app.py

#!/bin/bash
# Kill whatever is on port 8502, then start Streamlit
cd "$(dirname "$0")"
lsof -ti:8502 | xargs kill -9 2>/dev/null || true
sleep 1
source .venv/bin/activate
streamlit run app.py --server.port=8502

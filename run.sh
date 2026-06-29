#!/bin/bash
# Build frontend and start backend server

cd "$(dirname "$0")"

# Build frontend
cd frontend && npm run build && cd ..

# Run backend with env file
cd backend && set -a && source .env && set +a && uv run --python 3.13 --extra qa --extra nninteractive python -m uvicorn server:app --app-dir src --host 0.0.0.0 --port 5990

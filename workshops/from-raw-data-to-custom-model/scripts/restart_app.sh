#!/usr/bin/env bash
# Restart the Streamlit demo. Usage: scripts/restart_app.sh [run_name]   (default: demo)
cd "$(dirname "$0")/.."
for p in $(pgrep -f "bin/[s]treamlit run app/app.py"); do kill "$p"; done
sleep 2
OCTANK_APP_RUN="${1:-demo}" nohup ~/.venvs/gift_concierge/bin/streamlit run app/app.py --server.port 8501 \
  --server.headless true --server.enableCORS false --server.enableXsrfProtection false \
  > work/streamlit.log 2>&1 &
sleep 10
curl -s -o /dev/null -w "streamlit http %{http_code}\n" http://localhost:8501/

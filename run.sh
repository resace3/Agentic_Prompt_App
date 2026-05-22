#!/usr/bin/env sh
set -eu

export PORT="${PORT:-5000}"
export CHAT_STORE_PATH="${CHAT_STORE_PATH:-/data/chat_history.json}"
export SENSOR_MAP_PATH="${SENSOR_MAP_PATH:-/data/sensor_map.json}"
export HOME_ASSISTANT_API_URL="${HOME_ASSISTANT_API_URL:-http://supervisor/core/api}"

exec python /app/app.py

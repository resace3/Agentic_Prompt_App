#!/usr/bin/with-contenv sh
# shellcheck shell=sh
set -eu

export PORT="${PORT:-5000}"
export CHAT_STORE_PATH="${CHAT_STORE_PATH:-/data/chat_history.json}"
export SENSOR_MAP_PATH="${SENSOR_MAP_PATH:-/data/sensor_map.json}"

if command -v python3 >/dev/null 2>&1; then
    exec python3 /app/app.py
fi

exec python /app/app.py

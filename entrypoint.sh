#!/bin/bash
set -e

echo "Starting CUA2 Application..."

# Start nginx in the background using the config from the app directory
nginx -c $HOME/app/nginx.conf -g 'daemon off;' &

sleep 2

# Check if nginx is running
if ! pgrep nginx > /dev/null; then
    echo "Error: nginx failed to start"
    exit 1
fi

echo "nginx started successfully"

cd $HOME/app/cua2-core

# Set default number of workers if not specified
NUM_WORKERS=${NUM_WORKERS:-1}

echo "Starting backend with $WORKERS worker(s)..."

# Use uv to run the application
echo "uv run uvicorn cua2_core.main:app --host 0.0.0.0 --port 8000 --workers $NUM_WORKERS --log-level error > /dev/null"
exec uv run uvicorn cua2_core.main:app --host 0.0.0.0 --port 8000 --workers $NUM_WORKERS --log-level error > /dev/null

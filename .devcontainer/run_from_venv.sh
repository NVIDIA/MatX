#!/bin/bash
set -e

# Step 1: Activate the virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
  echo "Activating venv"
  source "/opt/nvidia/venv/bin/activate"
fi

# Step 2: Execute the final command (passed as args)
exec "$@"

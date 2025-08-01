#!/bin/bash
set -e

# Step 1: Activate the virtual environment
source "/opt/nvidia/venv/bin/activate"

# Step 2: Execute the final command (passed as args)
exec "$@"

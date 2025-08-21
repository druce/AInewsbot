#!/bin/bash

# get vars from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Make sure we are in the right directory
cd "${HOMEDIR:?HOMEDIR environment variable not set}"
echo "Working directory: $HOMEDIR"

echo "Using Firefox profile: $FIREFOX_PROFILE_PATH"

echo "Activating conda environment: $ENV_NAME"
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || {
    echo "Failed to activate conda environment: $ENV_NAME"
    exit 1
}

# Windows
# source "$CONDA_PREFIX/etc/profile.d/conda.sh"  # Uncomment if using Git Bash
# eval "$(conda shell.bash hook)"  # Uncomment if using Git Bash

# Linux/Mac
# shellcheck source=/dev/null
# source "${CONDA_PREFIX:-$HOME/anaconda3}/etc/profile.d/conda.sh"

# Run the Python script
echo "Running AInewsbot.py..."
python "$HOMEDIR/AInewsbot.py" > "$HOMEDIR/AInewsbot.out" 2>&1

#!/bin/zsh

cd "${0:a:h}"

# Ensure logs and input directories exist at startup
mkdir -p logs
mkdir -p input

# Start the Python summariser
source myenv/bin/activate

# --- Modified Redirection ---
# Redirect both stdout (1) and stderr (2) to logs/history.log
nohup python3 summarise.py $1 $2 >> logs/history.log 2>&1 &
# --- End Modified Redirection ---

PYTHON_PID=$!
echo $PYTHON_PID > logs/process.pid
deactivate

echo "Paper summariser started with PID ${PYTHON_PID}. Output logged to logs/history.log"
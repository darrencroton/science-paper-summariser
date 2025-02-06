#!/bin/zsh

cd "${0:a:h}"

# Ensure logs and input directories exist at startup
mkdir -p logs
mkdir -p input

# Start the Python summariser
source myenv/bin/activate
nohup python3 summarise.py > /dev/null 2>&1 &
PYTHON_PID=$!
echo $PYTHON_PID > logs/process.pid
deactivate
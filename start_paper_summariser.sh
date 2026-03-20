#!/bin/zsh

cd "${0:a:h}"

# Ensure logs and input directories exist at startup
mkdir -p logs
mkdir -p input

# Start the Python summariser
source myenv/bin/activate

# Default provider is "claude" (CLI-first: uses Claude Code CLI if available,
# falls back to Anthropic API). Override with: ./start_paper_summariser.sh gemini
# Optionally specify a model: ./start_paper_summariser.sh claude claude-opus-4-6
nohup python3 summarise.py $1 $2 >> logs/history.log 2>&1 &

PYTHON_PID=$!
echo $PYTHON_PID > logs/process.pid
deactivate

echo "Paper summariser started with PID ${PYTHON_PID}. Output logged to logs/history.log"

#!/bin/zsh

cd "${0:a:h}"

# Ensure logs and input directories exist at startup
mkdir -p logs
mkdir -p input

# Usage:
#   ./start_paper_summariser.sh
#   ./start_paper_summariser.sh cli gemini
#   ./start_paper_summariser.sh cli claude --effort high
#   ./start_paper_summariser.sh api openai gpt-5.2
# Default with no arguments: cli claude

# Start the Python summariser
source myenv/bin/activate

if ! python3 - "$@" <<'PY'
import sys

from summarise import validate_startup_selection

try:
    validate_startup_selection(sys.argv[1:])
except Exception as exc:
    print(exc, file=sys.stderr)
    sys.exit(2)
PY
then
    deactivate
    exit 2
fi

display_args=("$@")
if [[ $# -eq 0 ]]; then
    display_args=(cli claude)
fi

nohup python3 summarise.py "$@" >> logs/history.log 2>&1 &

PYTHON_PID=$!
echo $PYTHON_PID > logs/process.pid
deactivate

echo "Paper summariser started with PID ${PYTHON_PID} using: python3 summarise.py ${display_args[*]}"
echo "Output logged to logs/history.log"

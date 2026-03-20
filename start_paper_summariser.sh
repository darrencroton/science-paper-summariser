#!/bin/zsh

cd "${0:a:h}"

# Ensure logs and input directories exist at startup
mkdir -p logs
mkdir -p input

# Usage:
#   ./start_paper_summariser.sh
#   ./start_paper_summariser.sh cli gemini
#   ./start_paper_summariser.sh api openai gpt-5.2
# Default with no arguments: cli claude
if [[ $# -ne 0 && $# -ne 2 && $# -ne 3 ]]; then
    echo "Usage: ./start_paper_summariser.sh [mode provider [model]]" >&2
    echo "Examples:" >&2
    echo "  ./start_paper_summariser.sh" >&2
    echo "  ./start_paper_summariser.sh cli gemini" >&2
    echo "  ./start_paper_summariser.sh api openai gpt-5.2" >&2
    echo "Old one-argument forms such as './start_paper_summariser.sh gemini' are no longer supported." >&2
    exit 2
fi

# Start the Python summariser
source myenv/bin/activate

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

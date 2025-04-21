#!/bin/zsh

# Change to the script's directory to ensure relative paths work
cd "${0:a:h}"

# Stop the main paper summarisation process using its PID file
if [ -f logs/process.pid ]; then
    MAIN_PID=$(cat logs/process.pid)
    if [ -n "$MAIN_PID" ] && ps -p $MAIN_PID > /dev/null; then
        echo "Stopping paper summariser process (PID: $MAIN_PID)..."
        # Send SIGTERM (graceful shutdown signal)
        kill $MAIN_PID 2>/dev/null
        # Allow a short time for graceful shutdown before removing PID file
        sleep 2
    else
        echo "PID file found, but process $MAIN_PID not running or PID invalid."
    fi
    # Remove the PID file regardless of whether the process was running
    rm logs/process.pid
    echo "PID file removed."
else
    echo "No active PID file found (logs/process.pid). Is the summariser running?"
    # Optional: You could add the ps aux grep kill loop here as a fallback
    # if you suspect orphaned processes might occur often, but try without first.
fi

# The Python script's signal handler should log the actual shutdown status to history.log
echo "Stop script finished."
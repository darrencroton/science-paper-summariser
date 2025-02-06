#!/bin/zsh

cd "${0:a:h}"  # Change to script directory using zsh syntax

# Stop paper summarisation
if [ -f logs/process.pid ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Stopping paper summarisation" >> logs/history.log
    kill $(cat logs/process.pid)
    rm logs/process.pid
fi
